"""Milestone C5 — Platt calibration (ADR-026).

Cross-encoder scores are NOT probabilities. A raw 0.7 does not mean 70% likely
relevant; the scale shifts across query types. Thresholding raw scores is an
arbitrary rule dressed up as a principled one, which is why every audit record
currently reads calibrator_version=uncalibrated.

Three corrections that ADR-026 locked, all implemented here:

  1. PLATT, not isotonic. At ~35 items per category isotonic overfits and
     produces a step function.
  2. Fit on the POST-RRF candidate distribution, not random corpus chunks.
     Random negatives are easy negatives; a calibrator trained on them is
     systematically overconfident on the hard, already-filtered candidates it
     meets at inference.
  3. 5-FOLD CROSS-FITTING. Fitting on all 800 judgments then reporting metrics
     on the same judgments means your abstention thresholds are tuned to the
     test set.

Pure-numpy logistic fit — no sklearn dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class PlattParams:
    a: float = 1.0
    b: float = 0.0
    n: int = 0
    category: str = "global"

    def predict(self, raw: float) -> float:
        z = max(-30.0, min(30.0, self.a * raw + self.b))
        return 1.0 / (1.0 + math.exp(-z))


@dataclass(slots=True)
class Calibrator:
    per_category: dict[str, PlattParams] = field(default_factory=dict)
    global_params: PlattParams = field(default_factory=PlattParams)
    version: str = "v1"
    reliability: dict[str, Any] = field(default_factory=dict)

    def predict(self, raw: float, category: str = "global") -> float:
        # Categories with too few judgments fall back to the global fit rather
        # than trusting an overfitted one.
        p = self.per_category.get(category)
        return (p or self.global_params).predict(raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "global": asdict(self.global_params),
                    "per_category": {k: asdict(v) for k, v in self.per_category.items()},
                    "reliability": self.reliability,
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            per_category={k: PlattParams(**v) for k, v in d["per_category"].items()},
            global_params=PlattParams(**d["global"]),
            version=d.get("version", "v1"),
            reliability=d.get("reliability", {}),
        )


MIN_PER_CATEGORY = 40  # below this, a per-category fit is not trustworthy


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, iters: int = 200) -> tuple[float, float]:
    """Newton-Raphson on a 1-D logistic, with Platt's label smoothing.

    Smoothing matters: with a small sample and clean separation the MLE runs off
    to infinity and every score collapses to 0 or 1.

    Each Newton step is backtracked until it decreases the loss. Without this,
    one overshooting step saturates the sigmoid, the Hessian collapses to the
    1e-9 floor while the gradient stays finite, and the next step divides by
    that floor — coefficients explode and every prediction lands on 0 or 1.
    """
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    hi = (n_pos + 1.0) / (n_pos + 2.0) if n_pos else 0.5
    lo = 1.0 / (n_neg + 2.0) if n_neg else 0.5
    t = np.where(y > 0, hi, lo)

    def loss(a_: float, b_: float) -> float:
        z_ = np.clip(a_ * x + b_, -30, 30)
        p_ = np.clip(1.0 / (1.0 + np.exp(-z_)), 1e-12, 1.0 - 1e-12)
        return float(-np.sum(t * np.log(p_) + (1 - t) * np.log(1 - p_)))

    a, b = 1.0, 0.0
    cur = loss(a, b)
    for _ in range(iters):
        z = np.clip(a * x + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1 - p), 1e-9, None)
        g_a = float(np.sum((p - t) * x))
        g_b = float(np.sum(p - t))
        h_aa = float(np.sum(w * x * x)) + 1e-9
        h_ab = float(np.sum(w * x))
        h_bb = float(np.sum(w)) + 1e-9
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        step = 1.0
        while step > 1e-6 and loss(a - step * da, b - step * db) > cur:
            step /= 2.0
        if step <= 1e-6:
            break
        a -= step * da
        b -= step * db
        cur = loss(a, b)
        if abs(step * da) < 1e-8 and abs(step * db) < 1e-8:
            break
    return a, b


def _reliability(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Reliability diagram data + ECE. The artifact that turns 'calibrated'
    from a bullet point into a measurement."""
    edges = np.linspace(0, 1, bins + 1)
    out: list[dict[str, float]] = []
    ece = 0.0
    for i in range(bins):
        m = (y_prob >= edges[i]) & (y_prob < edges[i + 1] if i < bins - 1 else y_prob <= 1.0)
        if not m.any():
            continue
        conf = float(y_prob[m].mean())
        acc = float(y_true[m].mean())
        frac = float(m.sum()) / len(y_true)
        ece += frac * abs(acc - conf)
        out.append(
            {
                "bin_lo": float(edges[i]),
                "bin_hi": float(edges[i + 1]),
                "confidence": round(conf, 4),
                "observed": round(acc, 4),
                "n": int(m.sum()),
            }
        )
    return {"bins": out, "ece": round(ece, 4), "n": len(y_true)}


def fit(judgments: list[dict[str, Any]], *, folds: int = 5) -> Calibrator:
    """Fit per category with 5-fold cross-fitting for honest reliability."""
    if not judgments:
        raise ValueError("no calibration judgments — run the labeling app in calibration mode")

    x_all = np.array([float(j["raw_score"]) for j in judgments], dtype=float)
    y_all = np.array([1.0 if j["is_relevant"] else 0.0 for j in judgments], dtype=float)
    cats = [str(j.get("category", "global")) for j in judgments]

    cal = Calibrator()
    a, b = _fit_logistic(x_all, y_all)
    cal.global_params = PlattParams(a, b, len(x_all), "global")

    for cat in sorted(set(cats)):
        m = np.array([c == cat for c in cats])
        if int(m.sum()) < MIN_PER_CATEGORY:
            continue
        ca, cb = _fit_logistic(x_all[m], y_all[m])
        cal.per_category[cat] = PlattParams(ca, cb, int(m.sum()), cat)

    # Out-of-fold predictions -> honest reliability, no leakage.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x_all))
    oof = np.zeros_like(y_all)
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        if len(train) < 10 or len(test) == 0:
            continue
        fa, fb = _fit_logistic(x_all[train], y_all[train])
        p = PlattParams(fa, fb)
        oof[test] = [p.predict(v) for v in x_all[test]]

    cal.reliability = _reliability(y_all, oof)
    cal.reliability["folds"] = folds
    cal.reliability["note"] = "out-of-fold predictions — no leakage"
    return cal


def abstention_curve(
    judgments: list[dict[str, Any]],
    cal: Calibrator,
    steps: int = 21,
) -> list[dict[str, float]]:
    """Accuracy vs coverage across thresholds — the headline artifact (ADR-043).

    'At 78% coverage the system is 99% accurate; the 22% it declines are the
    queries it should decline.' That single chart is what makes calibrated
    abstention a measurement rather than a claim.
    """
    rows: list[dict[str, float]] = []
    scored = [
        (
            cal.predict(float(j["raw_score"]), str(j.get("category", "global"))),
            bool(j["is_relevant"]),
        )
        for j in judgments
    ]
    for i in range(steps):
        t = i / (steps - 1)
        kept = [(p, r) for p, r in scored if p >= t]
        coverage = len(kept) / len(scored) if scored else 0.0
        precision = sum(1 for _, r in kept if r) / len(kept) if kept else 1.0
        rows.append(
            {
                "threshold": round(t, 3),
                "coverage": round(coverage, 4),
                "precision": round(precision, 4),
                "n_kept": len(kept),
            }
        )
    return rows
