"""
Targeted tests for extraction accuracy, classification edge cases,
and completeness in Member 2 intelligence extraction.

Tests:
1. Extraction completeness across all categories (multiple items per category).
2. Distinguishing requests vs accepted commitments (requests must not become accepted commitments).
3. Distinguishing tentative proposals vs confirmed decisions.
4. Preserving unknown owners and deadlines as None without fabrication.
5. Preserving contradictions without resolving them.
6. Evidence integrity: extraction never hallucinates evidence.
7. Traceability to original transcript evidence via attach_evidence.
8. Baseline BART summarizer remains untouched and functional.
"""

import json
import pytest
from member2_intelligence.intelligence import (
    Claim,
    Decision,
    ActionItem,
    Topic,
    Contradiction,
    MeetingIntelligence,
    extract_intelligence,
)
from member2_intelligence.intelligence.extractor import (
    _parse_commitment_status,
    _parse_decision_status,
    _parse_confidence,
    _convert_to_meeting_intelligence,
    _build_prompt,
)
from member2_intelligence.pipeline import (
    attach_evidence,
    extract_verified_intelligence,
)
from member2_intelligence.llm.summarizer import generate_summary


def _make_transcript():
    return {
        "meeting_id": "classification_test",
        "language": "en",
        "segments": [
            {
                "id": 1,
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 5.0,
                "raw_text": "The budget was finalized at fifty thousand dollars.",
                "cleaned_text": "The budget was finalized at fifty thousand dollars.",
                "text": "The budget was finalized at fifty thousand dollars.",
                "overlap": False,
            },
            {
                "id": 2,
                "speaker": "SPEAKER_01",
                "start": 5.5,
                "end": 10.0,
                "raw_text": "Can someone please check the server logs for memory spikes?",
                "cleaned_text": "Can someone please check the server logs for memory spikes?",
                "text": "Can someone please check the server logs for memory spikes?",
                "overlap": False,
            },
            {
                "id": 3,
                "speaker": "SPEAKER_02",
                "start": 10.5,
                "end": 15.0,
                "raw_text": "I will deploy the backend patch by tomorrow afternoon.",
                "cleaned_text": "I will deploy the backend patch by tomorrow afternoon.",
                "text": "I will deploy the backend patch by tomorrow afternoon.",
                "overlap": False,
            },
            {
                "id": 4,
                "speaker": "SPEAKER_00",
                "start": 15.5,
                "end": 20.0,
                "raw_text": "Maybe we should consider migrating to PostgreSQL next quarter.",
                "cleaned_text": "Maybe we should consider migrating to PostgreSQL next quarter.",
                "text": "Maybe we should consider migrating to PostgreSQL next quarter.",
                "overlap": False,
            },
            {
                "id": 5,
                "speaker": "SPEAKER_01",
                "start": 20.5,
                "end": 25.0,
                "raw_text": "We all agreed and decided to release version 2.0 next Wednesday.",
                "cleaned_text": "We all agreed and decided to release version 2.0 next Wednesday.",
                "text": "We all agreed and decided to release version 2.0 next Wednesday.",
                "overlap": False,
            },
            {
                "id": 6,
                "speaker": "SPEAKER_02",
                "start": 25.5,
                "end": 30.0,
                "raw_text": "Wait, I said version 2.0 cannot be released next Wednesday because testing is incomplete.",
                "cleaned_text": "Wait, I said version 2.0 cannot be released next Wednesday because testing is incomplete.",
                "text": "Wait, I said version 2.0 cannot be released next Wednesday because testing is incomplete.",
                "overlap": False,
            },
        ],
    }


def test_distinguish_request_from_accepted_commitment():
    """A request without an owner or acceptance must NOT become an accepted commitment."""
    # Test helper directly
    status_request = _parse_commitment_status(None, task_text="Can someone check the server logs?", owner=None)
    assert status_request == "requested"

    status_ask = _parse_commitment_status("request", task_text="Check server logs", owner=None)
    assert status_ask == "requested"

    # Accepted commitment
    status_committed = _parse_commitment_status("committed", task_text="I will deploy the backend patch", owner="SPEAKER_02")
    assert status_committed == "committed"

    status_normal = _parse_commitment_status(None, task_text="Deploy the backend patch", owner="SPEAKER_02")
    assert status_normal == "committed"


def test_distinguish_proposal_from_confirmed_decision():
    """Tentative suggestions/proposals must have status 'proposed', not 'confirmed'."""
    status_proposed = _parse_decision_status("proposed", text="Consider migrating to PostgreSQL")
    assert status_proposed == "proposed"

    status_tentative_text = _parse_decision_status(None, text="Proposal to migrate to PostgreSQL next quarter")
    assert status_tentative_text == "proposed"

    status_confirmed = _parse_decision_status("confirmed", text="Decided to release version 2.0 next Wednesday")
    assert status_confirmed == "confirmed"


def test_preserve_unknown_owner_and_deadline_as_none():
    """Unknown or unassigned owner and deadline must be preserved as None without fabrication."""
    raw_data = {
        "claims": [],
        "decisions": [],
        "action_items": [
            {
                "id": "a1",
                "task": "Can someone check the server logs for memory spikes?",
                "owner": None,
                "deadline": None,
                "commitment_status": "requested",
            },
            {
                "id": "a2",
                "task": "Investigate database caching options",
                "owner": "unknown",
                "deadline": "null",
            },
            {
                "id": "a3",
                "task": "Deploy backend patch",
                "owner": "SPEAKER_02",
                "deadline": "tomorrow afternoon",
                "commitment_status": "committed",
            },
        ],
        "topics": [],
        "contradictions": [],
    }

    intel = _convert_to_meeting_intelligence(raw_data)
    assert len(intel.action_items) == 3

    # a1 has None owner & deadline, commitment_status == 'requested'
    assert intel.action_items[0].owner is None
    assert intel.action_items[0].deadline is None
    assert intel.action_items[0].commitment_status == "requested"

    # a2 string 'unknown' / 'null' converted to None
    assert intel.action_items[1].owner is None
    assert intel.action_items[1].deadline is None

    # a3 has explicit owner & deadline & commitment
    assert intel.action_items[2].owner == "SPEAKER_02"
    assert intel.action_items[2].deadline == "tomorrow afternoon"
    assert intel.action_items[2].commitment_status == "committed"


def test_contradictions_preserve_both_perspectives():
    """Contradictions must describe opposing statements without resolving or picking a side."""
    raw_data = {
        "claims": [],
        "decisions": [
            {"id": "d1", "text": "Release version 2.0 next Wednesday", "status": "confirmed"}
        ],
        "action_items": [],
        "topics": [],
        "contradictions": [
            {
                "id": "x1",
                "description": "SPEAKER_01 states version 2.0 is scheduled for release next Wednesday, while SPEAKER_02 states it cannot be released then due to incomplete testing.",
            }
        ],
    }

    intel = _convert_to_meeting_intelligence(raw_data)
    assert len(intel.contradictions) == 1
    desc = intel.contradictions[0].description
    assert "SPEAKER_01" in desc
    assert "SPEAKER_02" in desc
    assert "cannot be released" in desc


def test_extraction_completeness_and_traceability():
    """All categories with multiple items are extracted and remain traceable to transcript evidence."""
    transcript = _make_transcript()

    mock_llm_json = json.dumps({
        "claims": [
            {"id": "c1", "text": "The budget was finalized at fifty thousand dollars.", "confidence": 0.95},
            {"id": "c2", "text": "Testing is incomplete for version 2.0.", "confidence": 0.88},
        ],
        "decisions": [
            {"id": "d1", "text": "We all agreed and decided to release version 2.0 next Wednesday.", "status": "confirmed", "confidence": 0.92},
            {"id": "d2", "text": "Maybe we should consider migrating to PostgreSQL next quarter.", "status": "proposed", "confidence": 0.75},
        ],
        "action_items": [
            {"id": "a1", "task": "Can someone please check the server logs for memory spikes?", "owner": None, "deadline": None, "commitment_status": "requested", "confidence": 0.85},
            {"id": "a2", "task": "Deploy the backend patch by tomorrow afternoon", "owner": "SPEAKER_02", "deadline": "tomorrow afternoon", "commitment_status": "committed", "confidence": 0.94},
        ],
        "topics": [
            {"id": "t1", "name": "Budget and Financials", "confidence": 0.90},
            {"id": "t2", "name": "Release Schedule", "confidence": 0.91},
        ],
        "contradictions": [
            {"id": "x1", "description": "SPEAKER_01 agreed to release version 2.0 next Wednesday, but SPEAKER_02 stated it cannot be released then.", "confidence": 0.89},
        ],
    })

    def mock_generate(**kwargs):
        return mock_llm_json

    # Step 1: Extraction produces items with empty evidence
    intel = extract_intelligence(transcript, generate_fn=mock_generate)
    assert len(intel.claims) == 2
    assert len(intel.decisions) == 2
    assert len(intel.action_items) == 2
    assert len(intel.topics) == 2
    assert len(intel.contradictions) == 1

    # Check request vs commitment
    assert intel.action_items[0].commitment_status == "requested"
    assert intel.action_items[1].commitment_status == "committed"

    # Check confirmed decision vs proposal
    assert intel.decisions[0].status == "confirmed"
    assert intel.decisions[1].status == "proposed"

    # Evidence lists initially empty
    for c in intel.claims:
        assert c.evidence == []

    # Step 2: Attach evidence traces to original transcript segments
    attach_evidence(intel, transcript, top_k=3)

    # Claim 1 matches Segment 1 (fifty thousand dollars)
    assert len(intel.claims[0].evidence) >= 1
    assert intel.claims[0].evidence[0].segment_id == 1
    assert intel.claims[0].verification_status == "SUPPORTED"
    assert intel.claims[0].support_score >= 0.50

    # Action item 2 matches Segment 3 (deploy the backend patch)
    assert len(intel.action_items[1].evidence) >= 1
    assert intel.action_items[1].evidence[0].segment_id == 3
    assert intel.action_items[1].verification_status == "SUPPORTED"


def test_bart_baseline_summarizer_preserved():
    """BART summarizer function generate_summary exists, is callable, and preserves baseline behavior."""
    assert callable(generate_summary)
    # Fast test with short input
    short_text = "The team met to review the quarterly budget and agreed to deploy the update."
    result = generate_summary(short_text)
    assert isinstance(result, str)
    assert len(result) > 0
