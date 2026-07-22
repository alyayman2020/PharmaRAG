"""Cross-encoder reranking (ADR-024, ADR-025).

ADR-025 is the rule that matters: NEVER SKIP. Reranking here is not a ranking
optimization — it is the SCORING stage that feeds abstention. Skip it and those
queries reach generation with no relevance gate at all, and the ≤8-candidate
case is exactly where the corpus is thinnest and abstention matters most.

Scores are NOT calibrated probabilities. A raw 0.7 does not mean 70% likely
relevant; the scale shifts across query types. Thresholding raw scores is an
arbitrary rule dressed up as a principled one. The Platt calibrator (ADR-026)
lands at milestone C5 — until then, do not describe this as "calibrated".
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

from pharmarag.config import RERANKER_CPU, RERANKER_GPU, settings


@dataclass(slots=True)
class ScoredCandidate:
    chunk_id: str
    raw_score: float
    calibrated_score: float
    payload: dict[str, Any]


@functools.lru_cache(maxsize=2)
def _load(device: str) -> Any:
    from sentence_transformers import CrossEncoder

    model_name = RERANKER_GPU if device == "cuda" else RERANKER_CPU
    kwargs: dict[str, Any] = {"device": device}
    if device == "cuda":
        kwargs["model_kwargs"] = {"torch_dtype": "float16"}
    return CrossEncoder(model_name, **kwargs)


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _load_platt() -> Any | None:
    """Load the fitted Platt calibrator (ADR-026), gated on CALIBRATOR_VERSION.

    The gate is the safety property: ``data/calibrator.json`` may exist from a
    fit against an OLDER corpus, and score distributions shift when the corpus
    changes. The operator opts in by setting ``CALIBRATOR_VERSION`` (e.g. ``v1``)
    in ``.env`` after refitting — a stale calibrator is never silently reused.

    Uses the GLOBAL fit only: per-category params exist in the file, but the
    query's eval category is unknown at rerank time.
    """
    from pharmarag.config import DATA, settings

    if settings.calibrator_version == "uncalibrated":
        return None
    path = DATA / "calibrator.json"
    if not path.is_file():
        return None
    import json

    g = json.loads(path.read_text(encoding="utf-8"))["global"]
    a, b = float(g["a"]), float(g["b"])

    def predict(raw: float) -> float:
        return _sigmoid(a * raw + b)

    return predict


class Reranker:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or settings.resolve_device()
        self.model_name = RERANKER_GPU if self.device == "cuda" else RERANKER_CPU
        self._model: Any | None = None
        self._calibrator: Any | None = _load_platt()  # None while CALIBRATOR_VERSION=uncalibrated

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = _load(self.device)
        return self._model

    def score(self, query: str, candidates: list[Any]) -> list[ScoredCandidate]:
        """Score EVERY candidate. No skip path exists by design (ADR-025)."""
        if not candidates:
            return []
        pairs = [(query, str(c.payload.get("display_text", ""))) for c in candidates]
        raw = self.model.predict(pairs, show_progress_bar=False)

        out: list[ScoredCandidate] = []
        for cand, r in zip(candidates, raw, strict=True):
            rv = float(r)
            cal = self._calibrate(rv)
            out.append(ScoredCandidate(cand.chunk_id, rv, cal, cand.payload))
        return sorted(out, key=lambda s: -s.calibrated_score)

    def _calibrate(self, raw: float) -> float:
        """Platt-calibrated when CALIBRATOR_VERSION is set; sigmoid squash otherwise.

        The fallback sigmoid makes thresholds *usable*, not *calibrated* — the
        audit log carries `calibrator_version=uncalibrated` so the distinction is
        visible in every record until a fit is activated (ADR-026).
        """
        if self._calibrator is not None:
            return float(self._calibrator(raw))
        return raw if 0.0 <= raw <= 1.0 else _sigmoid(raw)
