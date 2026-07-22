"""Central configuration. Every constant here traces to a locked ADR.

Changing a value in this file changes a design decision — update the ADR too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- paths
ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = ROOT / "data"
SNAPSHOTS: Final[Path] = DATA / "snapshots"
ARCHIVE: Final[Path] = DATA / "archive"
QDRANT_PATH: Final[Path] = DATA / "qdrant"

DB_MAIN: Final[Path] = DATA / "pharmarag.db"  # corpus + chunks + parents + audit
DB_CHECKPOINTS: Final[Path] = DATA / "checkpoints.db"  # LangGraph (Track B)
DB_MLFLOW: Final[Path] = DATA / "mlflow.db"


# --------------------------------------------------------------------------- ADR-005
# LOINC-coded SPL sections. codeSystem is always 2.16.840.1.113883.6.1
LOINC_CODESYSTEM: Final[str] = "2.16.840.1.113883.6.1"

SECTION_NAMES: Final[dict[str, str]] = {
    "34066-1": "Boxed Warning",
    "34070-3": "Contraindications",
    "43685-7": "Warnings and Precautions",
    "34071-1": "Warnings",
    "34073-7": "Drug Interactions",
    "34068-7": "Dosage and Administration",
    "43684-0": "Use in Specific Populations",
    "34090-1": "Clinical Pharmacology",
    "34088-5": "Overdosage",  # ingested + archived, NEVER indexed (ADR-005)
}

# Sections written to the vector store.
RETRIEVABLE_SECTIONS: Final[frozenset[str]] = frozenset(
    {"34066-1", "34070-3", "43685-7", "34071-1", "34073-7", "34068-7", "43684-0", "34090-1"}
)

# ADR-005: ingested and archived for completeness, never written to Qdrant.
NON_RETRIEVABLE_SECTIONS: Final[frozenset[str]] = frozenset({"34088-5"})

assert not (RETRIEVABLE_SECTIONS & NON_RETRIEVABLE_SECTIONS), "ADR-005 violated"

# ADR-027 safety tiers — context ordering is by tier first, relevance within tier.
SAFETY_TIER: Final[dict[str, int]] = {
    "34066-1": 1,  # Boxed Warning
    "34070-3": 1,  # Contraindications
    "34073-7": 2,  # Drug Interactions
    "43685-7": 2,  # Warnings and Precautions
    "34071-1": 2,
    "34068-7": 3,  # Dosage
    "43684-0": 3,  # Specific Populations
    "34090-1": 4,  # Clinical Pharmacology
}

# ADR-027 asymmetric relevance floors, by consequence.
FLOOR_TIER_1: Final[float] = 0.25
FLOOR_TIER_OTHER: Final[float] = 0.40


# --------------------------------------------------------------------------- ADR-012
class ChunkPolicy:
    """Per-section chunking. A single global size would merge unrelated
    interaction statements into one vector (ADR-012)."""

    __slots__ = ("atomic", "max_tokens", "name", "overlap_sentences", "split")

    def __init__(
        self,
        name: str,
        max_tokens: int,
        *,
        atomic: bool = False,
        split: bool = True,
        overlap_sentences: int = 0,
    ) -> None:
        self.name = name
        self.max_tokens = max_tokens
        self.atomic = atomic  # one item == one chunk, never merged
        self.split = split  # False == never split this section
        self.overlap_sentences = overlap_sentences


CHUNK_POLICIES: Final[dict[str, ChunkPolicy]] = {
    "34073-7": ChunkPolicy("interaction-atomic", 250, atomic=True),
    "34070-3": ChunkPolicy("contraindication-atomic", 250, atomic=True),
    "34066-1": ChunkPolicy("boxed-whole", 2000, split=False),
    "43684-0": ChunkPolicy("population-subsection", 400, atomic=True),
    "34068-7": ChunkPolicy("dosing-prose", 300, overlap_sentences=1),
    "43685-7": ChunkPolicy("warnings-subsection", 400, overlap_sentences=1),
    "34071-1": ChunkPolicy("warnings-subsection", 400, overlap_sentences=1),
    "34090-1": ChunkPolicy("pharmacology-prose", 400, overlap_sentences=1),
}
DEFAULT_POLICY: Final[ChunkPolicy] = ChunkPolicy("default-prose", 350, overlap_sentences=1)

# ADR-014: below this a chunk is mostly prefix, so merge upward with its sibling.
MIN_CHUNK_TOKENS: Final[int] = 60
MAX_PARENT_TOKENS: Final[int] = 1500

# ADR-013 population tags
POPULATION_TAGS: Final[tuple[str, ...]] = (
    "renal",
    "hepatic",
    "pediatric",
    "geriatric",
    "pregnancy",
    "lactation",
)


# --------------------------------------------------------------------------- retrieval
COLLECTION: Final[str] = "pharmarag"
DENSE_VECTOR: Final[str] = "dense"
SPARSE_VECTOR: Final[str] = "sparse"
EMBED_DIM: Final[int] = 1536  # ADR-015 — full, no Matryoshka truncation

# ADR-018 payload indexes. effective_time is deliberately absent: staleness is
# surfaced, never filtered, or a currency problem becomes a false negative.
PAYLOAD_INDEXES: Final[tuple[str, ...]] = (
    "rxcui",
    "content_type",
    "loinc_section_code",
    "population_tags",
    "pharm_class_epc",
    "is_canonical",
    "ingredient_name",
)

# ADR-023 prefetch branch limits. RRF is a candidate generator, not a final
# ranker, so these are tuned for recall.
PREFETCH_DENSE_PROSE: Final[int] = 20
PREFETCH_SPARSE_PROSE: Final[int] = 20
PREFETCH_DENSE_TABLE: Final[int] = 15
PREFETCH_SPARSE_TABLE: Final[int] = 15
RRF_K: Final[int] = 60
TOP_PARENTS: Final[int] = 8
MAX_CONTEXT_TOKENS: Final[int] = 8000  # ADR-027

MAX_EXPANSION_RXCUIS: Final[int] = 25  # ADR-037


# --------------------------------------------------------------------------- models
EMBED_MODEL: Final[str] = "text-embedding-3-small"
MODEL_GUARD: Final[str] = "gpt-5.4"  # ADR-038 highest-consequence classification
MODEL_ROUTE: Final[str] = "gpt-5.4-nano"  # ADR-038 trivial consequence
MODEL_SYNTHESIS: Final[str] = "gpt-5.4-nano"  # ADR-030 — only call seeing full context
MODEL_EVALUATOR: Final[str] = "gpt-5.4-mini"  # ADR-030 — sees ~2k, so affordable
RERANKER_GPU: Final[str] = "BAAI/bge-reranker-v2-m3"  # ADR-024
RERANKER_CPU: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # CPU fallback

PROMPT_TEMPLATE_VERSION: Final[str] = "v1.0.0"  # ADR-047 reconstruction key

# ADR-026 thresholds, expressed on the RAW (sigmoid-squashed) reranker scale.
# A Platt calibrator is fit and validated (eval/calibrate.py, ECE 0.064) but
# NOT active: its shallow slope compresses scores so far that even the best
# candidate maps below 0.60 — activating CALIBRATOR_VERSION=v1 without first
# migrating these two constants to the calibrated scale mass-abstains.
THRESHOLD_INCLUDE: Final[float] = 0.60
THRESHOLD_FLAG: Final[float] = 0.40


# --------------------------------------------------------------------------- settings
def _drop_hostile_sslkeylogfile() -> None:
    """Avast's HTTPS scanner injects SSLKEYLOGFILE=\\\\.\\aswMonFltProxy\\<handle>.

    Python opens that path as a FILE* when building an SSLContext, and because it is a
    device handle rather than a regular file OpenSSL aborts the process with
    "OPENSSL_Uplink: no OPENSSL_Applink". That kills even fully offline code paths, since
    merely constructing an HTTPS-capable session creates a context. Drop the variable only
    when it names a non-regular path, so a genuine keylog file used for TLS debugging is
    left alone.
    """
    value = os.environ.get("SSLKEYLOGFILE")
    if not value:
        return
    try:
        usable = Path(value).parent.is_dir()
    except OSError:
        usable = False  # device paths raise PermissionError rather than returning False
    if not usable:
        del os.environ["SSLKEYLOGFILE"]


_drop_hostile_sslkeylogfile()


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — avoids a dependency in the module everything imports."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # Qdrant server URL (e.g. http://localhost:6333). Empty means embedded local
    # mode, which loads the ENTIRE collection into Python memory at client
    # construction — fine for tests, unusable at the 1000-drug corpus size
    # (3.7 GB on disk -> 6+ GB RAM and minutes of load). See scripts/start_qdrant.ps1.
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    corpus_version: str = field(default_factory=lambda: os.getenv("CORPUS_VERSION", "dev"))
    graph_version: str = field(default_factory=lambda: os.getenv("GRAPH_VERSION", "none"))
    calibrator_version: str = field(
        default_factory=lambda: os.getenv("CALIBRATOR_VERSION", "uncalibrated")
    )
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    dailymed_base: str = "https://dailymed.nlm.nih.gov"
    rxnav_base: str = "https://rxnav.nlm.nih.gov"
    openfda_base: str = "https://api.fda.gov"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


settings = Settings()


def ensure_dirs() -> None:
    for p in (DATA, SNAPSHOTS, ARCHIVE, QDRANT_PATH):
        p.mkdir(parents=True, exist_ok=True)
