"""
Targeted tests for confidence and evidence-support score semantics,
thresholds, and verification architecture.

Tests:
1. Clear distinction between evidence-support score (lexical grounding) and model confidence.
2. High model confidence with no evidence remains honestly UNSUPPORTED.
3. Threshold semantics:
     score >= 0.50 -> SUPPORTED
     0.20 <= score < 0.50 -> PARTIALLY_SUPPORTED
     score < 0.20 -> UNSUPPORTED
4. Partial evidence handling: weak items do not have unverified evidence claimed as verified.
5. verify_item helper verifies all intelligence item types (Claim, Decision, ActionItem, Topic, Contradiction).
6. Schema validation for support_score and verification_status values.
"""

import json
import pytest
from member2_intelligence.intelligence import (
    Evidence,
    Claim,
    Decision,
    ActionItem,
    Topic,
    Contradiction,
    MeetingIntelligence,
    verify_evidence,
    verify_claim,
    verify_item,
    VerificationResult,
    THRESHOLD_SUPPORTED,
    THRESHOLD_PARTIAL,
)
from member2_intelligence.pipeline import attach_evidence


def _make_sample_transcript():
    return {
        "meeting_id": "score_semantics_test",
        "language": "en",
        "segments": [
            {
                "id": 1,
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 6.0,
                "raw_text": "The marketing budget of seventy thousand dollars was approved by executive management.",
                "cleaned_text": "The marketing budget of seventy thousand dollars was approved by executive management.",
                "text": "The marketing budget of seventy thousand dollars was approved by executive management.",
                "overlap": False,
            },
            {
                "id": 2,
                "speaker": "SPEAKER_01",
                "start": 6.5,
                "end": 12.0,
                "raw_text": "We noticed some marketing expenses last month.",
                "cleaned_text": "We noticed some marketing expenses last month.",
                "text": "We noticed some marketing expenses last month.",
                "overlap": False,
            },
        ],
    }


def test_threshold_constants_defined_and_documented():
    """Threshold constants must be 0.50 for SUPPORTED and 0.20 for PARTIAL."""
    assert THRESHOLD_SUPPORTED == 0.50
    assert THRESHOLD_PARTIAL == 0.20


def test_model_confidence_distinct_from_evidence_support():
    """High model confidence without transcript evidence must NOT produce verified status."""
    transcript = _make_sample_transcript()

    # Claim with 99% model confidence, but text has zero transcript grounding
    intel = MeetingIntelligence(
        claims=[
            Claim(
                id="c_hallucinated",
                text="The company acquired a quantum computing startup in Tokyo.",
                confidence=0.99,
                evidence=[],
            )
        ]
    )

    attach_evidence(intel, transcript, top_k=3)

    c = intel.claims[0]
    # Model confidence remains separate
    assert c.confidence == 0.99

    # Evidence support is 0.0 and status is honestly UNSUPPORTED
    assert c.support_score == 0.0
    assert c.verification_status == "UNSUPPORTED"
    assert c.evidence == []
    assert c.partial_evidence == []


def test_strong_support_classification():
    """Item with >= 50% lexical overlap receives SUPPORTED status and attached evidence."""
    transcript = _make_sample_transcript()

    intel = MeetingIntelligence(
        claims=[
            Claim(
                id="c_strong",
                text="The marketing budget of seventy thousand dollars was approved by executive management.",
                confidence=0.85,
                evidence=[],
            )
        ]
    )

    attach_evidence(intel, transcript, top_k=3)

    c = intel.claims[0]
    assert c.verification_status == "SUPPORTED"
    assert c.support_score >= THRESHOLD_SUPPORTED
    assert len(c.evidence) >= 1
    assert c.evidence[0].segment_id == 1
    assert c.partial_evidence == []


def test_partial_support_classification():
    """Item with partial overlap (>= 0.20 and < 0.50) receives PARTIALLY_SUPPORTED without false verified evidence."""
    transcript = _make_sample_transcript()

    # Query shares terms like "marketing" and "expenses", but fails >= 0.50 threshold
    intel = MeetingIntelligence(
        claims=[
            Claim(
                id="c_partial",
                text="Marketing expenses exceeded quarterly operational financial forecasts significantly.",
                confidence=0.70,
                evidence=[],
            )
        ]
    )

    attach_evidence(intel, transcript, top_k=3)

    c = intel.claims[0]
    assert c.verification_status == "PARTIALLY_SUPPORTED"
    assert THRESHOLD_PARTIAL <= c.support_score < THRESHOLD_SUPPORTED

    # Verified evidence list remains empty to prevent false grounding
    assert c.evidence == []
    # Partial candidates stored in partial_evidence
    assert len(c.partial_evidence) >= 1
    assert c.partial_evidence[0].segment_id == 2


def test_unsupported_classification():
    """Item with < 0.20 overlap receives UNSUPPORTED status and empty evidence."""
    transcript = _make_sample_transcript()

    intel = MeetingIntelligence(
        claims=[
            Claim(
                id="c_unsupported",
                text="Unrelated statement about weather conditions in Antarctica.",
                confidence=0.60,
                evidence=[],
            )
        ]
    )

    attach_evidence(intel, transcript, top_k=3)

    c = intel.claims[0]
    assert c.verification_status == "UNSUPPORTED"
    assert c.support_score < THRESHOLD_PARTIAL
    assert c.evidence == []
    assert c.partial_evidence == []


def test_verify_item_works_across_all_item_types():
    """verify_item works on Claim, Decision, ActionItem, Topic, and Contradiction."""
    transcript = _make_sample_transcript()

    ev1 = Evidence(
        segment_id=1,
        speaker="SPEAKER_00",
        start=0.0,
        end=6.0,
        quote="The marketing budget of seventy thousand dollars was approved by executive management.",
    )

    # Claim
    claim = Claim(id="c1", text="Marketing budget approved by executive management", evidence=[ev1])
    res_claim = verify_item(claim, transcript)
    assert res_claim.status == "SUPPORTED"
    assert res_claim.score >= THRESHOLD_SUPPORTED

    # Decision
    dec = Decision(id="d1", text="Executive management approved seventy thousand dollars marketing budget", evidence=[ev1])
    res_dec = verify_item(dec, transcript)
    assert res_dec.status == "SUPPORTED"

    # ActionItem
    act = ActionItem(id="a1", task="Executive management allocate seventy thousand dollars marketing budget", evidence=[ev1])
    res_act = verify_item(act, transcript)
    assert res_act.status == "SUPPORTED"

    # Topic
    top = Topic(id="t1", name="Marketing budget approved", evidence=[ev1])
    res_top = verify_item(top, transcript)
    assert res_top.status == "SUPPORTED"

    # Contradiction
    contra = Contradiction(id="x1", description="Marketing budget seventy thousand dollars approved", evidence=[ev1])
    res_contra = verify_item(contra, transcript)
    assert res_contra.status == "SUPPORTED"


def test_schema_validates_support_score_and_verification_status():
    """Schemas reject out-of-range support_score or invalid verification_status."""
    ev = Evidence(
        segment_id=1,
        speaker="SPEAKER_00",
        start=0.0,
        end=6.0,
        quote="The marketing budget of seventy thousand dollars was approved by executive management.",
    )

    # Valid values validate fine
    c_valid = Claim(id="c1", text="Valid claim", evidence=[ev], support_score=0.75, verification_status="SUPPORTED")
    c_valid.validate()

    # Invalid support_score > 1.0 raises ValueError
    c_bad_score = Claim(id="c2", text="Bad score", evidence=[ev], support_score=1.5, verification_status="SUPPORTED")
    with pytest.raises(ValueError, match="support_score"):
        c_bad_score.validate()

    # Invalid status raises ValueError
    c_bad_status = Claim(id="c3", text="Bad status", evidence=[ev], support_score=0.5, verification_status="SUPER_VERIFIED")
    with pytest.raises(ValueError, match="verification_status"):
        c_bad_status.validate()
