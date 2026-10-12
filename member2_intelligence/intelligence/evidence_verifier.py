"""
Deterministic Evidence Verifier

Verifies that Evidence objects genuinely correspond to the structured
transcript and that claims have textual support from their cited evidence.

This is a deterministic baseline verifier using lexical overlap only.
It does NOT understand semantics, negation, uncertainty, temporal changes,
or contradiction. Those limitations will be addressed in a future
LLM-assisted verification stage.

Uses Python standard-library only — no external dependencies.
"""
import string
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from member2_intelligence.intelligence.schemas import Evidence, Claim


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------

# Lexical overlap thresholds for claim support classification.
# These are deliberately conservative for a lexical-only baseline.
#
#   SUPPORTED:           >= 50% of claim tokens found in evidence text
#   PARTIALLY_SUPPORTED: >= 20% of claim tokens found in evidence text
#   UNSUPPORTED:         <  20% of claim tokens found in evidence text
#
# These thresholds operate on the BEST single evidence object's score,
# after aggregating across all evidence (taking the max).

THRESHOLD_SUPPORTED = 0.50
THRESHOLD_PARTIAL = 0.20


@dataclass
class VerificationResult:
    """
    The result of verifying a Claim against a transcript.

    Attributes:
        status:            One of "SUPPORTED", "PARTIALLY_SUPPORTED",
                           or "UNSUPPORTED".
        reason:            Human-readable explanation of the verdict.
        verified_evidence: List of Evidence objects that passed structural
                           verification (exist in transcript, metadata matches).
        score:             The best lexical overlap score across all
                           verified evidence (0.0–1.0). Useful for debugging
                           and threshold tuning.
    """
    status: str
    reason: str
    verified_evidence: List[Evidence] = field(default_factory=list)
    score: float = 0.0


# ---------------------------------------------------------------------------
# Stop words — same minimal set as the evidence retriever for consistency
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "between", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "no", "so", "if", "then", "than",
    "too", "very", "just",
})


# ---------------------------------------------------------------------------
# Text normalization (mirrors evidence_retriever for consistency)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return text


def _tokenize(text: str) -> List[str]:
    """Normalize, split, remove stop words. Returns deduplicated tokens
    in encounter order."""
    normalized = _normalize(text)
    words = normalized.split()
    seen = set()
    tokens = []
    for w in words:
        if w and w not in _STOP_WORDS and w not in seen:
            seen.add(w)
            tokens.append(w)
    return tokens


# ---------------------------------------------------------------------------
# Segment lookup helper
# ---------------------------------------------------------------------------

def _find_segment(transcript: Dict[str, Any], segment_id: int) -> Optional[Dict[str, Any]]:
    """Find a segment by id in the transcript. Returns None if not found."""
    for seg in transcript.get("segments", []):
        if seg.get("id") == segment_id:
            return seg
    return None


# ---------------------------------------------------------------------------
# Evidence verification (structural)
# ---------------------------------------------------------------------------

def verify_evidence(transcript: Dict[str, Any], evidence: Evidence) -> bool:
    """
    Verifies that an Evidence object structurally corresponds to the
    transcript.

    Checks:
      1. segment_id exists in the transcript.
      2. speaker matches exactly.
      3. start timestamp matches exactly.
      4. end timestamp matches exactly.
      5. quote matches segment["raw_text"] exactly (no paraphrasing).

    Args:
        transcript: The structured transcript dictionary.
        evidence:   The Evidence object to verify.

    Returns:
        True if all checks pass, False otherwise.
    """
    segment = _find_segment(transcript, evidence.segment_id)
    if segment is None:
        return False

    if evidence.speaker != segment.get("speaker"):
        return False

    if evidence.start != segment.get("start"):
        return False

    if evidence.end != segment.get("end"):
        return False

    if evidence.quote != segment.get("raw_text"):
        return False

    return True


# ---------------------------------------------------------------------------
# Claim support scoring
# ---------------------------------------------------------------------------

def _claim_evidence_overlap(claim_text: str, quote: str) -> float:
    """
    Computes lexical overlap between a claim and an evidence quote.

    Score = (number of claim tokens found in the quote) / (total claim tokens)

    Returns a float in [0.0, 1.0].
    """
    claim_tokens = _tokenize(claim_text)
    if not claim_tokens:
        return 0.0

    quote_words = set(_normalize(quote).split())
    matches = sum(1 for ct in claim_tokens if ct in quote_words)
    return matches / len(claim_tokens)


# ---------------------------------------------------------------------------
# Claim verification (structural + lexical support)
# ---------------------------------------------------------------------------

def verify_claim(claim: Claim, transcript: Dict[str, Any]) -> VerificationResult:
    """
    Verifies a Claim against the transcript.

    Process:
      1. For each Evidence in the claim, run structural verification
         (verify_evidence). Only structurally valid evidence proceeds.
      2. For each verified evidence, compute lexical overlap between
         the claim text and the evidence quote.
      3. Take the best (maximum) overlap score across all verified evidence.
      4. Classify using thresholds:
           score >= THRESHOLD_SUPPORTED  (0.50) -> "SUPPORTED"
           score >= THRESHOLD_PARTIAL    (0.20) -> "PARTIALLY_SUPPORTED"
           score <  THRESHOLD_PARTIAL    (0.20) -> "UNSUPPORTED"

    Known limitations (deterministic lexical baseline):
      - Cannot understand negation ("budget approved" vs "budget NOT approved")
      - Cannot detect temporal changes ("was approved" vs "is no longer approved")
      - Cannot assess semantic equivalence or paraphrasing
      - Cannot detect contradiction between evidence segments
      These will be addressed in a future LLM-assisted verification stage.

    Args:
        claim:      The Claim object to verify.
        transcript: The structured transcript dictionary.

    Returns:
        A VerificationResult with status, reason, verified_evidence, and score.
    """
    # Handle empty/missing claim text
    if not claim.text or not claim.text.strip():
        return VerificationResult(
            status="UNSUPPORTED",
            reason="Claim text is empty.",
            verified_evidence=[],
            score=0.0,
        )

    # Handle empty evidence list
    if not claim.evidence:
        return VerificationResult(
            status="UNSUPPORTED",
            reason="Claim has no evidence to verify.",
            verified_evidence=[],
            score=0.0,
        )

    # Step 1: Structural verification of each evidence object
    verified: List[Evidence] = []
    for ev in claim.evidence:
        if verify_evidence(transcript, ev):
            verified.append(ev)

    if not verified:
        return VerificationResult(
            status="UNSUPPORTED",
            reason="None of the cited evidence could be verified against the transcript.",
            verified_evidence=[],
            score=0.0,
        )

    # Step 2: Compute lexical overlap for each verified evidence
    best_score = 0.0
    for ev in verified:
        score = _claim_evidence_overlap(claim.text, ev.quote)
        if score > best_score:
            best_score = score

    # Step 3: Classify
    if best_score >= THRESHOLD_SUPPORTED:
        status = "SUPPORTED"
        reason = (
            f"Claim is lexically supported by evidence "
            f"(best overlap score: {best_score:.2f} >= {THRESHOLD_SUPPORTED})."
        )
    elif best_score >= THRESHOLD_PARTIAL:
        status = "PARTIALLY_SUPPORTED"
        reason = (
            f"Claim has partial lexical support from evidence "
            f"(best overlap score: {best_score:.2f}, "
            f"between {THRESHOLD_PARTIAL} and {THRESHOLD_SUPPORTED})."
        )
    else:
        status = "UNSUPPORTED"
        reason = (
            f"Claim has insufficient lexical support from evidence "
            f"(best overlap score: {best_score:.2f} < {THRESHOLD_PARTIAL})."
        )

    return VerificationResult(
        status=status,
        reason=reason,
        verified_evidence=verified,
        score=best_score,
    )


# ---------------------------------------------------------------------------
# General item verification
# ---------------------------------------------------------------------------

def verify_item(item: Any, transcript: Dict[str, Any]) -> VerificationResult:
    """
    Verifies any intelligence item (Claim, Decision, ActionItem, Topic,
    or Contradiction) against the transcript using deterministic structural
    and lexical evidence verification.

    Score semantics:
      - score is a float in [0.0, 1.0], measuring direct textual grounding
        (lexical overlap ratio between the item's query tokens and the evidence).
      - score is NOT a statistical probability of factual truth, and MUST NOT
        be confused with model confidence.
      - Status classification:
          score >= THRESHOLD_SUPPORTED (0.50) -> "SUPPORTED"
          score >= THRESHOLD_PARTIAL   (0.20) -> "PARTIALLY_SUPPORTED"
          score <  THRESHOLD_PARTIAL   (0.20) -> "UNSUPPORTED"

    Args:
        item:       An intelligence item (Claim, Decision, ActionItem, Topic, Contradiction).
        transcript: The structured transcript dictionary.

    Returns:
        A VerificationResult with status, reason, verified_evidence, and score.
    """
    if isinstance(item, Claim):
        text = item.text
        evidence = item.evidence
    elif hasattr(item, "text"):
        text = getattr(item, "text")
        evidence = getattr(item, "evidence", [])
    elif hasattr(item, "task"):
        text = getattr(item, "task")
        evidence = getattr(item, "evidence", [])
    elif hasattr(item, "name"):
        text = getattr(item, "name")
        evidence = getattr(item, "evidence", [])
    elif hasattr(item, "description"):
        text = getattr(item, "description")
        evidence = getattr(item, "evidence", [])
    else:
        text = str(item)
        evidence = getattr(item, "evidence", [])

    temp_claim = Claim(
        id=str(getattr(item, "id", "item")),
        text=text,
        evidence=evidence,
    )
    return verify_claim(temp_claim, transcript)

