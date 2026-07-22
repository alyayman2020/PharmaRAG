"""spl-parser — HL7 SPL XML extraction.

No mature open-source SPL parser exists (pySPL on GitHub is a stub), so this is
written from scratch. Kept free of pharmarag imports so it can be extracted to
its own public repo in Phase 5 (ADR-010).
"""

from spl_parser.models import Section, SPLDocument, Table
from spl_parser.sections import parse_spl
from spl_parser.tables import LinearizedRow, linearize, linearize_all

__all__ = [
    "LinearizedRow",
    "SPLDocument",
    "Section",
    "Table",
    "linearize",
    "linearize_all",
    "parse_spl",
]
__version__ = "0.1.0"
