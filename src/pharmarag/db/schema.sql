-- PharmaRAG main store: corpus + chunks + parents + audit (ADR-019, ADR-046).
-- checkpoints.db and mlflow.db are SEPARATE FILES and must never be merged here:
-- the append-only triggers below would break LangGraph resume, and MLflow
-- migrations must never touch the audit table.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------------ documents
CREATE TABLE IF NOT EXISTS documents (
    set_id            TEXT PRIMARY KEY,
    doc_version       INTEGER NOT NULL,
    effective_time    TEXT,                    -- surfaced, NEVER filtered (ADR-018)
    title             TEXT,
    ingredient_name   TEXT,
    rxcui             TEXT,
    brand_names       TEXT,                    -- JSON array
    application_type  TEXT,                    -- NDA | ANDA | BLA
    is_canonical      INTEGER NOT NULL DEFAULT 0,
    conflict_of       TEXT,                    -- set_id of the canonical label
    sha256            TEXT NOT NULL,           -- content-addressed archive key
    corpus_version    TEXT NOT NULL,
    ingested_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_documents_rxcui ON documents(rxcui);
CREATE INDEX IF NOT EXISTS ix_documents_canonical ON documents(is_canonical);

-- ------------------------------------------------------------------ parents
-- Parents live OUTSIDE the vector store (ADR-019). They are fetched, never searched.
CREATE TABLE IF NOT EXISTS parents (
    parent_chunk_id   TEXT PRIMARY KEY,
    set_id            TEXT NOT NULL REFERENCES documents(set_id),
    loinc_section_code TEXT NOT NULL,
    section_name      TEXT NOT NULL,
    section_path      TEXT,
    parent_part       TEXT,                    -- "1/3" when a subsection exceeds the cap
    text              TEXT NOT NULL,
    token_count       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_parents_set ON parents(set_id);

-- ------------------------------------------------------------------ chunks
-- Full manifest INCLUDING non-retrievable chunks. `retrievable=0` rows document
-- the ADR-005 Overdusage exclusion; they are never written to Qdrant.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          TEXT PRIMARY KEY,
    set_id            TEXT NOT NULL REFERENCES documents(set_id),
    parent_chunk_id   TEXT REFERENCES parents(parent_chunk_id),
    doc_version       INTEGER,
    effective_time    TEXT,
    corpus_version    TEXT NOT NULL,
    content_sha256    TEXT NOT NULL,
    chunk_policy      TEXT NOT NULL,
    token_count       INTEGER NOT NULL,

    section_path      TEXT,
    source_url        TEXT,

    embed_text        TEXT NOT NULL,           -- PREFIXED   -> dense  (ADR-014)
    display_text      TEXT NOT NULL,
    raw_text          TEXT NOT NULL,           -- UNPREFIXED -> BM25   (ADR-021)

    rxcui             TEXT,
    rxcui_all         TEXT,
    ingredient_name   TEXT,
    brand_names       TEXT,
    pharm_class_epc   TEXT,

    loinc_section_code TEXT NOT NULL,
    section_name      TEXT NOT NULL,
    content_type      TEXT NOT NULL,           -- prose | table_row | table_json | list
    table_id          TEXT,
    footnote_text     TEXT,

    retrievable       INTEGER NOT NULL DEFAULT 1,
    units_present     TEXT,
    population_tags   TEXT,
    dose_values       TEXT,                    -- JSON array of DoseValue

    application_type  TEXT,
    is_canonical      INTEGER NOT NULL DEFAULT 1,
    is_variant        INTEGER NOT NULL DEFAULT 0,
    conflict_of       TEXT
);
CREATE INDEX IF NOT EXISTS ix_chunks_rxcui ON chunks(rxcui);
CREATE INDEX IF NOT EXISTS ix_chunks_parent ON chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS ix_chunks_retrievable ON chunks(retrievable);
CREATE INDEX IF NOT EXISTS ix_chunks_sha ON chunks(content_sha256);

-- ------------------------------------------------------------------ embedding cache
-- ADR-016: keyed by content hash, so re-running the pipeline on unchanged
-- chunks costs $0.
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_sha256    TEXT NOT NULL,
    model             TEXT NOT NULL,
    dim               INTEGER NOT NULL,
    vector            BLOB NOT NULL,
    created_at        TEXT NOT NULL,
    PRIMARY KEY (content_sha256, model, dim)
);

-- ------------------------------------------------------------------ audit log
-- ADR-047. Design standard: a reader must be able to reconstruct WHY the system
-- said what it said, from this record alone.
CREATE TABLE IF NOT EXISTS audit_log (
    query_id                    TEXT PRIMARY KEY,
    parent_query_id             TEXT,          -- multi-turn disambiguation linkage
    session_id                  TEXT,
    timestamp_utc               TEXT NOT NULL,

    raw_query                   TEXT,          -- NULL when K1 flags first-person clinical
    normalized_query            TEXT NOT NULL,
    redacted                    INTEGER NOT NULL DEFAULT 0,
    guard_verdict               TEXT,
    guard_model_version         TEXT,

    resolved_rxcuis             TEXT,
    resolution_tier             TEXT,
    resolution_confidence       REAL,
    substitutions_surfaced      TEXT,
    expansion_applied           TEXT,
    expansion_overflow          INTEGER DEFAULT 0,

    retrieved_chunk_ids         TEXT,
    chunk_sha256                TEXT,          -- content-addressed, not chunk text
    reranker_scores             TEXT,
    calibrated_scores           TEXT,
    context_assembled_chunk_ids TEXT,          -- DISTINCT from retrieved (8k cap drops)

    prompt_template_version     TEXT NOT NULL, -- enables reconstruction
    prompt_hash                 TEXT,          -- verifies the reconstruction
    synthesis_model_version     TEXT,
    evaluator_model_version     TEXT,
    guardrail_results           TEXT,
    evaluator_verdict           TEXT,
    retry_count                 INTEGER DEFAULT 0,
    rejection_reasons           TEXT,

    structured_output           TEXT,
    final_action                TEXT,
    reason_code                 TEXT,
    disclaimer_shown            INTEGER NOT NULL DEFAULT 1,

    corpus_version              TEXT,
    graph_version               TEXT,
    calibrator_version          TEXT,
    latency_ms_by_stage         TEXT,
    cost_usd                    REAL
);
CREATE INDEX IF NOT EXISTS ix_audit_ts ON audit_log(timestamp_utc);
CREATE INDEX IF NOT EXISTS ix_audit_parent ON audit_log(parent_query_id);

-- Tamper-evidence for the cost of two DDL statements (ADR-047).
-- A privileged retention job is the only sanctioned deletion path, and it must
-- drop these triggers explicitly and write its own audit entry.
CREATE TRIGGER IF NOT EXISTS audit_no_update
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only (ADR-047)');
END;

CREATE TRIGGER IF NOT EXISTS audit_no_delete
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only (ADR-047)');
END;
