"""
Unit and integration tests for the Member 2 Verified Intelligence Pipeline.

Tests the end-to-end verified intelligence backend stage:
    processed structured transcript
              ↓
        Qwen extractor
              ↓
    Claims / Decisions / ActionItems / Topics / Contradictions
              ↓
    evidence_retriever.retrieve_evidence()
              ↓
    evidence_verifier.verify_claim()
              ↓
    verified evidence attached to each intelligence item

Verifies all milestone requirements using deterministic mocked Qwen output:
1. A Qwen claim gets relevant transcript evidence.
2. Verified evidence is attached (structurally and lexically valid).
3. Unsupported evidence is not attached.
4. Decisions and action items are processed.
5. Multiple evidence segments work.
6. No-match items remain present but have empty evidence.
7. Pipeline is deterministic and preserves transcript integrity.

Requires NO Ollama, external network, or GPU/audio dependencies.
"""

import sys
import os
import copy
import json
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
)
from member2_intelligence.pipeline import (
    extract_verified_intelligence,
    attach_evidence,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_sample_transcript():
    """
    Builds a realistic processed structured transcript with 4 distinct segments.
    """
    return {
        "meeting_id": "eng_sync_001",
        "language": "en",
        "segments": [
            {
                "id": 1,
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 6.5,
                "raw_text": "The board approved fifty thousand dollars for marketing yesterday.",
                "cleaned_text": "The board approved fifty thousand dollars for marketing yesterday.",
                "text": "The board approved fifty thousand dollars for marketing yesterday.",
                "overlap": False,
            },
            {
                "id": 2,
                "speaker": "SPEAKER_01",
                "start": 7.0,
                "end": 14.0,
                "raw_text": "Alice will complete the database schema migration by Friday.",
                "cleaned_text": "Alice will complete the database schema migration by Friday.",
                "text": "Alice will complete the database schema migration by Friday.",
                "overlap": False,
            },
            {
                "id": 3,
                "speaker": "SPEAKER_02",
                "start": 14.5,
                "end": 21.0,
                "raw_text": "We decided to deploy the new microservices architecture next week.",
                "cleaned_text": "We decided to deploy the new microservices architecture next week.",
                "text": "We decided to deploy the new microservices architecture next week.",
                "overlap": False,
            },
            {
                "id": 4,
                "speaker": "SPEAKER_00",
                "start": 21.5,
                "end": 28.0,
                "raw_text": "Fifty thousand dollars for marketing will be allocated directly to online campaigns.",
                "cleaned_text": "Fifty thousand dollars for marketing will be allocated directly to online campaigns.",
                "text": "Fifty thousand dollars for marketing will be allocated directly to online campaigns.",
                "overlap": False,
            },
        ],
    }


# ===========================================================================
# Test 1: A Qwen claim gets relevant transcript evidence
# ===========================================================================

def test_qwen_claim_gets_relevant_transcript_evidence():
    """
    Requirement 1: A Qwen claim receives relevant transcript evidence from the
    actual transcript segment.
    """
    transcript = _make_sample_transcript()

    mock_qwen_response = json.dumps({
        "claims": [
            {"id": "c1", "text": "The board approved fifty thousand dollars for marketing."}
        ],
        "decisions": [],
        "action_items": [],
        "topics": [],
        "contradictions": [],
    })

    def mock_generate(**kwargs):
        return mock_qwen_response

    intelligence = extract_verified_intelligence(transcript, generate_fn=mock_generate)

    assert len(intelligence.claims) == 1
    claim = intelligence.claims[0]
    assert len(claim.evidence) >= 1

    # Segment 1 is the primary source
    evidence = claim.evidence[0]
    assert evidence.segment_id == 1
    assert evidence.speaker == "SPEAKER_00"
    assert evidence.start == 0.0
    assert evidence.end == 6.5
    assert evidence.quote == "The board approved fifty thousand dollars for marketing yesterday."


# ===========================================================================
# Test 2: Verified evidence is attached
# ===========================================================================

def test_verified_evidence_is_attached():
    """
    Requirement 2: Only evidence that passes structural verification against
    the transcript is attached, and original raw_text is preserved as quote.
    """
    transcript = _make_sample_transcript()

    mock_qwen_response = json.dumps({
        "claims": [
            {"id": "c1", "text": "Deploy the new microservices architecture next week."}
        ],
        "decisions": [
            {"id": "d1", "text": "We decided to deploy microservices architecture next week."}
        ],
        "action_items": [
            {"id": "a1", "task": "Alice complete database schema migration Friday", "owner": "SPEAKER_01", "deadline": "Friday"}
        ],
        "topics": [],
        "contradictions": [],
    })

    def mock_generate(**kwargs):
        return mock_qwen_response

    intelligence = extract_verified_intelligence(transcript, generate_fn=mock_generate)

    all_items = intelligence.claims + intelligence.decisions + intelligence.action_items
    for item in all_items:
        assert len(item.evidence) > 0, f"Expected verified evidence for item {item.id}"
        for ev in item.evidence:
            # 1. Structural verification passes
            assert verify_evidence(transcript, ev) is True

            # 2. Schema validation passes
            ev.validate()

            # 3. quote preserves segment raw_text exactly
            matching_seg = next(s for s in transcript["segments"] if s["id"] == ev.segment_id)
            assert ev.quote == matching_seg["raw_text"]
            assert ev.speaker == matching_seg["speaker"]
            assert ev.start == matching_seg["start"]
            assert ev.end == matching_seg["end"]


# ===========================================================================
# Test 3: Unsupported evidence is not attached
# ===========================================================================

def test_unsupported_evidence_not_attached():
    """
    Requirement 3: Unsupported claims (unrelated to the transcript) or weak
    candidates (< 0.50 threshold) do not receive evidence.
    """
    transcript = _make_sample_transcript()

    mock_qwen_response = json.dumps({
        "claims": [
            # Totally unrelated to meeting
            {"id": "c_unsupported", "text": "Quantum computing breaks RSA encryption completely."},
            # Weak partial overlap (< 50% match)
            {"id": "c_weak", "text": "Marketing team went to Paris for conference yesterday."},
        ],
        "decisions": [],
        "action_items": [],
        "topics": [],
        "contradictions": [],
    })

    def mock_generate(**kwargs):
        return mock_qwen_response

    intelligence = extract_verified_intelligence(transcript, generate_fn=mock_generate)

    assert len(intelligence.claims) == 2

    c_unsupported = next(c for c in intelligence.claims if c.id == "c_unsupported")
    assert c_unsupported.evidence == [], "Unsupported claim must have empty evidence"

    c_weak = next(c for c in intelligence.claims if c.id == "c_weak")
    assert c_weak.evidence == [], "Weak/insufficient claim must not have unsupported evidence attached"


def test_unsupported_candidate_filtered_out_when_one_supported():
    """
    When the retriever returns multiple candidates, only candidates that individually
    meet the SUPPORTED threshold are attached; unsupported candidates are excluded.
    """
    transcript = _make_sample_transcript()

    # Pre-build an intelligence item
    intelligence = MeetingIntelligence(
        claims=[
            Claim(
                id="c1",
                text="The board approved fifty thousand dollars for marketing yesterday.",
                evidence=[],
            )
        ]
    )

    # Attach evidence with top_k=4. Candidate 1 is exact match (SUPPORTED).
    # Candidate 2 (Alice complete database migration) has 0 overlap with marketing budget.
    attach_evidence(intelligence, transcript, top_k=4)

    claim = intelligence.claims[0]
    assert len(claim.evidence) >= 1
    segment_ids = [ev.segment_id for ev in claim.evidence]

    # Segment 1 and Segment 4 both talk about fifty thousand dollars for marketing
    assert 1 in segment_ids
    # Segment 2 (Alice database migration) MUST NOT be attached
    assert 2 not in segment_ids
    # Segment 3 (Microservices deployment) MUST NOT be attached
    assert 3 not in segment_ids


# ===========================================================================
# Test 4: Decisions and action items are processed
# ===========================================================================

def test_decisions_and_action_items_processed():
    """
    Requirement 4: Decisions and Action Items are processed by the pipeline,
    receiving verified evidence, while maintaining metadata (owner, deadline).
    """
    transcript = _make_sample_transcript()

    mock_qwen_response = json.dumps({
        "claims": [],
        "decisions": [
            {"id": "d1", "text": "We decided to deploy the new microservices architecture next week."}
        ],
        "action_items": [
            {
                "id": "a1",
                "task": "Alice will complete the database schema migration by Friday",
                "owner": "SPEAKER_01",
                "deadline": "Friday",
            }
        ],
        "topics": [
            {"id": "t1", "name": "Marketing budget"}
        ],
        "contradictions": [],
    })

    def mock_generate(**kwargs):
        return mock_qwen_response

    intelligence = extract_verified_intelligence(transcript, generate_fn=mock_generate)

    # Verify Decision
    assert len(intelligence.decisions) == 1
    d1 = intelligence.decisions[0]
    assert d1.text == "We decided to deploy the new microservices architecture next week."
    assert len(d1.evidence) == 1
    assert d1.evidence[0].segment_id == 3
    assert d1.evidence[0].speaker == "SPEAKER_02"

    # Verify ActionItem
    assert len(intelligence.action_items) == 1
    a1 = intelligence.action_items[0]
    assert a1.task == "Alice will complete the database schema migration by Friday"
    assert a1.owner == "SPEAKER_01"
    assert a1.deadline == "Friday"
    assert len(a1.evidence) == 1
    assert a1.evidence[0].segment_id == 2
    assert a1.evidence[0].speaker == "SPEAKER_01"

    # Verify Topic
    assert len(intelligence.topics) == 1
    t1 = intelligence.topics[0]
    assert len(t1.evidence) >= 1
    assert t1.evidence[0].segment_id in (1, 4)


# ===========================================================================
# Test 5: Multiple evidence segments work
# ===========================================================================

def test_multiple_evidence_segments_work():
    """
    Requirement 5: An intelligence item supported across multiple distinct
    transcript segments receives multiple verified Evidence objects.
    """
    transcript = _make_sample_transcript()
    # Segments 1 and 4 both discuss "fifty thousand dollars for marketing"
    # Seg 1: "The board approved fifty thousand dollars for marketing yesterday."
    # Seg 4: "Fifty thousand dollars for marketing will be allocated directly to online campaigns."

    mock_qwen_response = json.dumps({
        "claims": [
            {"id": "c_multi", "text": "Fifty thousand dollars allocated for marketing."}
        ],
        "decisions": [],
        "action_items": [],
        "topics": [],
        "contradictions": [],
    })

    def mock_generate(**kwargs):
        return mock_qwen_response

    intelligence = extract_verified_intelligence(transcript, generate_fn=mock_generate, top_k=3)

    claim = intelligence.claims[0]
    assert len(claim.evidence) >= 2, f"Expected multiple evidence segments, got {len(claim.evidence)}"

    attached_segment_ids = {ev.segment_id for ev in claim.evidence}
    assert 1 in attached_segment_ids, "Segment 1 must be attached as evidence"
    assert 4 in attached_segment_ids, "Segment 4 must be attached as evidence"

    for ev in claim.evidence:
        assert verify_evidence(transcript, ev) is True
        ev.validate()


# ===========================================================================
# Test 6: No-match items remain present but have empty evidence
# ===========================================================================

def test_no_match_items_remain_present_with_empty_evidence():
    """
    Requirement 6: Items with no matching evidence in the transcript are NOT
    silently discarded or deleted; they remain present in the MeetingIntelligence
    container with an empty evidence list (evidence == []).
    """
    transcript = _make_sample_transcript()

    mock_qwen_response = json.dumps({
        "claims": [
            {"id": "c_valid", "text": "The board approved fifty thousand dollars for marketing."},
            {"id": "c_nomatch", "text": "The moon rover discovered liquid water in crater."},
        ],
        "decisions": [
            {"id": "d_nomatch", "text": "The company will acquire a robotics startup."},
        ],
        "action_items": [
            {"id": "a_nomatch", "task": "Design quantum processor architecture", "owner": None, "deadline": None},
        ],
        "topics": [
            {"id": "t_nomatch", "name": "Interstellar navigation"},
        ],
        "contradictions": [
            {"id": "x_nomatch", "description": "Contradiction about rocket fuel composition."},
        ],
    })

    def mock_generate(**kwargs):
        return mock_qwen_response

    intelligence = extract_verified_intelligence(transcript, generate_fn=mock_generate)

    # 1. Matching claim has evidence
    c_valid = next(c for c in intelligence.claims if c.id == "c_valid")
    assert len(c_valid.evidence) >= 1

    # 2. All no-match items remain present
    assert len(intelligence.claims) == 2
    assert len(intelligence.decisions) == 1
    assert len(intelligence.action_items) == 1
    assert len(intelligence.topics) == 1
    assert len(intelligence.contradictions) == 1

    c_nomatch = next(c for c in intelligence.claims if c.id == "c_nomatch")
    assert c_nomatch.evidence == []

    d_nomatch = intelligence.decisions[0]
    assert d_nomatch.id == "d_nomatch"
    assert d_nomatch.evidence == []

    a_nomatch = intelligence.action_items[0]
    assert a_nomatch.id == "a_nomatch"
    assert a_nomatch.evidence == []

    t_nomatch = intelligence.topics[0]
    assert t_nomatch.id == "t_nomatch"
    assert t_nomatch.evidence == []

    x_nomatch = intelligence.contradictions[0]
    assert x_nomatch.id == "x_nomatch"
    assert x_nomatch.evidence == []


# ===========================================================================
# Edge Cases & Pipeline Contract
# ===========================================================================

def test_empty_transcript_handling():
    """Empty transcript returns empty MeetingIntelligence without error."""
    intelligence = extract_verified_intelligence({})
    assert isinstance(intelligence, MeetingIntelligence)
    assert len(intelligence.claims) == 0

    intelligence_empty_segs = extract_verified_intelligence({"segments": []})
    assert isinstance(intelligence_empty_segs, MeetingIntelligence)
    assert len(intelligence_empty_segs.claims) == 0


def test_transcript_not_mutated_by_verified_intelligence():
    """The input transcript dict must never be modified by the intelligence stage."""
    transcript = _make_sample_transcript()
    original = copy.deepcopy(transcript)

    mock_qwen_response = json.dumps({
        "claims": [{"id": "c1", "text": "Board approved marketing budget."}],
        "decisions": [],
        "action_items": [],
        "topics": [],
        "contradictions": [],
    })

    extract_verified_intelligence(transcript, generate_fn=lambda **kw: mock_qwen_response)

    assert transcript == original


def test_pipeline_determinism():
    """Given identical transcript and LLM response, output is 100% deterministic."""
    transcript = _make_sample_transcript()

    mock_qwen_response = json.dumps({
        "claims": [
            {"id": "c1", "text": "Fifty thousand dollars allocated for marketing."}
        ],
        "decisions": [
            {"id": "d1", "text": "We decided to deploy the new microservices architecture next week."}
        ],
        "action_items": [
            {"id": "a1", "task": "Alice will complete the database schema migration by Friday", "owner": "SPEAKER_01", "deadline": "Friday"}
        ],
        "topics": [],
        "contradictions": [],
    })

    res1 = extract_verified_intelligence(transcript, generate_fn=lambda **kw: mock_qwen_response)
    res2 = extract_verified_intelligence(transcript, generate_fn=lambda **kw: mock_qwen_response)

    assert len(res1.claims[0].evidence) == len(res2.claims[0].evidence)
    for ev1, ev2 in zip(res1.claims[0].evidence, res2.claims[0].evidence):
        assert ev1.segment_id == ev2.segment_id
        assert ev1.quote == ev2.quote
        assert ev1.start == ev2.start
        assert ev1.end == ev2.end


def test_extract_verified_intelligence_stage_callbacks():
    """extract_verified_intelligence fires stage_callback for qwen_extraction and evidence."""
    transcript = _make_sample_transcript()
    recorded = []

    mock_qwen_response = json.dumps({
        "claims": [{"id": "c1", "text": "Board approved fifty thousand dollars for marketing."}],
        "decisions": [],
        "action_items": [],
        "topics": [],
        "contradictions": [],
    })

    def callback(stage, dur):
        recorded.append((stage, dur))

    extract_verified_intelligence(
        transcript,
        generate_fn=lambda **kw: mock_qwen_response,
        stage_callback=callback,
    )

    stage_names = [s[0] for s in recorded]
    assert "qwen_extraction" in stage_names
    assert "evidence" in stage_names
    for name, dur in recorded:
        assert isinstance(dur, (int, float))
        assert dur >= 0.0


# ===========================================================================
# Standalone Runner
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Running Member 2 Verified Intelligence Tests")
    print("=" * 60)

    print("\n--- Milestone Requirements ---")
    test_qwen_claim_gets_relevant_transcript_evidence()
    print("  PASS: 1. Qwen claim gets relevant transcript evidence")

    test_verified_evidence_is_attached()
    print("  PASS: 2. Verified evidence is attached")

    test_unsupported_evidence_not_attached()
    test_unsupported_candidate_filtered_out_when_one_supported()
    print("  PASS: 3. Unsupported evidence is not attached")

    test_decisions_and_action_items_processed()
    print("  PASS: 4. Decisions and action items are processed")

    test_multiple_evidence_segments_work()
    print("  PASS: 5. Multiple evidence segments work")

    test_no_match_items_remain_present_with_empty_evidence()
    print("  PASS: 6. No-match items remain present with empty evidence")

    print("\n--- Pipeline Integrity & Determinism ---")
    test_empty_transcript_handling()
    print("  PASS: Empty transcript handled safely")

    test_transcript_not_mutated_by_verified_intelligence()
    print("  PASS: Transcript not mutated")

    test_pipeline_determinism()
    print("  PASS: Deterministic execution verified")

    print("\n" + "=" * 60)
    print("[OK] ALL VERIFIED INTELLIGENCE TESTS PASSED")
    print("=" * 60)
