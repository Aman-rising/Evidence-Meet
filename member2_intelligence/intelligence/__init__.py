"""
member2_intelligence.intelligence

Evidence-based meeting intelligence package.
Provides schemas, extractor, evidence retriever, and evidence verifier.
"""

from member2_intelligence.intelligence.schemas import (
    Evidence,
    Claim,
    Decision,
    ActionItem,
    Topic,
    Contradiction,
    MeetingIntelligence,
)
from member2_intelligence.intelligence.extractor import (
    extract_intelligence,
    ExtractionError,
)
from member2_intelligence.intelligence.evidence_retriever import (
    retrieve_evidence,
)
from member2_intelligence.intelligence.evidence_verifier import (
    verify_evidence,
    verify_claim,
    verify_item,
    VerificationResult,
    THRESHOLD_SUPPORTED,
    THRESHOLD_PARTIAL,
)

__all__ = [
    "Evidence",
    "Claim",
    "Decision",
    "ActionItem",
    "Topic",
    "Contradiction",
    "MeetingIntelligence",
    "extract_intelligence",
    "ExtractionError",
    "retrieve_evidence",
    "verify_evidence",
    "verify_claim",
    "verify_item",
    "VerificationResult",
    "THRESHOLD_SUPPORTED",
    "THRESHOLD_PARTIAL",
]
