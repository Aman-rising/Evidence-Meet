"""
Unit tests for member2_intelligence.intelligence.extractor

Tests the Qwen/Ollama intelligence extraction pipeline using mock LLM
responses. Fully deterministic — does NOT require Ollama to be running.

All tests inject a mock generate_fn that returns predetermined JSON,
simulating Qwen3 responses without network access.
"""
import sys
import os
import copy
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from member2_intelligence.intelligence.extractor import (
    extract_intelligence,
    ExtractionError,
    _clean_json_response,
    _parse_llm_response,
    _convert_to_meeting_intelligence,
    _build_prompt,
)
from member2_intelligence.intelligence.schemas import (
    MeetingIntelligence,
    Claim,
    Decision,
    ActionItem,
    Topic,
    Contradiction,
)


# -- Fixtures ---------------------------------------------------------------

def _make_transcript():
    """A fixed multi-segment transcript for testing."""
    return {
        "meeting_id": "test_extract",
        "language": "en",
        "segments": [
            {
                "id": 1,
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 5.5,
                "raw_text": "The budget for Q3 has been approved by the board.",
                "cleaned_text": "The budget for q3 has been approved by the board.",
                "text": "The budget for Q3 has been approved by the board.",
                "overlap": False,
            },
            {
                "id": 2,
                "speaker": "SPEAKER_01",
                "start": 6.0,
                "end": 12.0,
                "raw_text": "We need to hire two engineers before Friday.",
                "cleaned_text": "We need to hire two engineers before friday.",
                "text": "We need to hire two engineers before Friday.",
                "overlap": False,
            },
            {
                "id": 3,
                "speaker": "SPEAKER_02",
                "start": 13.0,
                "end": 18.5,
                "raw_text": "I disagree. The budget should not have been approved.",
                "cleaned_text": "I disagree. The budget should not have been approved.",
                "text": "I disagree. The budget should not have been approved.",
                "overlap": False,
            },
        ],
    }


# A well-formed Qwen JSON response covering all categories
_VALID_QWEN_RESPONSE = json.dumps({
    "claims": [
        {"id": "c1", "text": "The Q3 budget has been approved by the board."},
        {"id": "c2", "text": "Two engineers need to be hired."},
    ],
    "decisions": [
        {"id": "d1", "text": "The board approved the Q3 budget."},
    ],
    "action_items": [
        {"id": "a1", "task": "Hire two engineers", "owner": "SPEAKER_01", "deadline": "Friday"},
        {"id": "a2", "task": "Review budget allocation", "owner": None, "deadline": None},
    ],
    "topics": [
        {"id": "t1", "name": "Q3 Budget"},
        {"id": "t2", "name": "Hiring"},
    ],
    "contradictions": [
        {"id": "x1", "description": "SPEAKER_00 says budget was approved; SPEAKER_02 disagrees."},
    ],
})


def _mock_generate_valid(**kwargs):
    """Mock generate function returning valid JSON."""
    return _VALID_QWEN_RESPONSE


def _mock_generate_with_markdown(**kwargs):
    """Mock returning JSON wrapped in markdown fences."""
    return "```json\n" + _VALID_QWEN_RESPONSE + "\n```"


def _mock_generate_with_think_tags(**kwargs):
    """Mock returning JSON with Qwen3 thinking tags."""
    return (
        "<think>Let me analyze this transcript carefully...</think>\n"
        + _VALID_QWEN_RESPONSE
    )


def _mock_generate_malformed(**kwargs):
    """Mock returning invalid JSON."""
    return "This is not valid JSON at all {broken"


def _mock_generate_empty_lists(**kwargs):
    """Mock returning valid JSON with all empty lists."""
    return json.dumps({
        "claims": [],
        "decisions": [],
        "action_items": [],
        "topics": [],
        "contradictions": [],
    })


def _mock_generate_null_owner_deadline(**kwargs):
    """Mock returning action items with explicit null owner/deadline."""
    return json.dumps({
        "claims": [],
        "decisions": [],
        "action_items": [
            {"id": "a1", "task": "Investigate options", "owner": None, "deadline": None},
            {"id": "a2", "task": "Send report", "owner": "null", "deadline": "unknown"},
        ],
        "topics": [],
        "contradictions": [],
    })


# Multi-item response fixture with multiple items across EVERY category
_MULTI_ITEM_PER_CATEGORY_RESPONSE = json.dumps({
    "claims": [
        {"id": "c1", "text": "The Q3 budget has been approved by the board."},
        {"id": "c2", "text": "Two engineers need to be hired before Friday."},
        {"id": "c3", "text": "The release candidate is scheduled for next Wednesday."},
    ],
    "decisions": [
        {"id": "d1", "text": "The board approved the Q3 budget allocation."},
        {"id": "d2", "text": "Deploy the new release candidate next Wednesday."},
    ],
    "action_items": [
        {"id": "a1", "task": "Hire two engineers", "owner": "SPEAKER_01", "deadline": "Friday"},
        {"id": "a2", "task": "Review budget allocation", "owner": None, "deadline": None},
        {"id": "a3", "task": "Prepare deployment checklist", "owner": "SPEAKER_02", "deadline": "Wednesday"},
    ],
    "topics": [
        {"id": "t1", "name": "Q3 Budget"},
        {"id": "t2", "name": "Hiring"},
        {"id": "t3", "name": "Deployment Schedule"},
    ],
    "contradictions": [
        {"id": "x1", "description": "SPEAKER_00 says budget was approved; SPEAKER_02 disagrees."},
        {"id": "x2", "description": "Conflicting deadlines mentioned for hiring engineers."},
    ],
})


def _mock_generate_multiple_items(**kwargs):
    """Mock generate function returning multiple items in all categories."""
    return _MULTI_ITEM_PER_CATEGORY_RESPONSE


# -- Tests: Valid Extraction ------------------------------------------------

def test_valid_qwen_json_extraction():
    """Full extraction from valid Qwen JSON response."""
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_valid)

    assert isinstance(result, MeetingIntelligence)
    assert len(result.claims) == 2
    assert len(result.decisions) == 1
    assert len(result.action_items) == 2
    assert len(result.topics) == 2
    assert len(result.contradictions) == 1
    print("  PASS: valid Qwen JSON extracted into MeetingIntelligence")


def test_claims_conversion():
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_valid)

    assert result.claims[0].id == "c1"
    assert "Q3 budget" in result.claims[0].text
    assert result.claims[1].id == "c2"
    print("  PASS: claims converted correctly")


def test_decisions_conversion():
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_valid)

    assert result.decisions[0].id == "d1"
    assert "approved" in result.decisions[0].text.lower()
    print("  PASS: decisions converted correctly")


def test_action_items_conversion():
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_valid)

    a1 = result.action_items[0]
    assert a1.id == "a1"
    assert a1.task == "Hire two engineers"
    assert a1.owner == "SPEAKER_01"
    assert a1.deadline == "Friday"

    a2 = result.action_items[1]
    assert a2.owner is None
    assert a2.deadline is None
    print("  PASS: action items converted with owner/deadline")


def test_topics_conversion():
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_valid)

    assert result.topics[0].name == "Q3 Budget"
    assert result.topics[1].name == "Hiring"
    print("  PASS: topics converted correctly")


def test_contradictions_conversion():
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_valid)

    assert result.contradictions[0].id == "x1"
    assert "disagrees" in result.contradictions[0].description
    print("  PASS: contradictions converted correctly")


def test_multiple_items_conversion_all_categories():
    """All categories with multiple items must be converted without truncation."""
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_multiple_items)

    assert isinstance(result, MeetingIntelligence)
    # Check counts
    assert len(result.claims) == 3, f"Expected 3 claims, got {len(result.claims)}"
    assert len(result.decisions) == 2, f"Expected 2 decisions, got {len(result.decisions)}"
    assert len(result.action_items) == 3, f"Expected 3 action items, got {len(result.action_items)}"
    assert len(result.topics) == 3, f"Expected 3 topics, got {len(result.topics)}"
    assert len(result.contradictions) == 2, f"Expected 2 contradictions, got {len(result.contradictions)}"

    # Check distinct IDs and content
    assert [c.id for c in result.claims] == ["c1", "c2", "c3"]
    assert [d.id for d in result.decisions] == ["d1", "d2"]
    assert [a.id for a in result.action_items] == ["a1", "a2", "a3"]
    assert [t.id for t in result.topics] == ["t1", "t2", "t3"]
    assert [x.id for x in result.contradictions] == ["x1", "x2"]

    # Check action item owners and deadlines
    assert result.action_items[0].owner == "SPEAKER_01"
    assert result.action_items[0].deadline == "Friday"
    assert result.action_items[1].owner is None
    assert result.action_items[1].deadline is None
    assert result.action_items[2].owner == "SPEAKER_02"
    assert result.action_items[2].deadline == "Wednesday"

    # Evidence lists must be empty for all extracted items
    for c in result.claims:
        assert c.evidence == []
    for d in result.decisions:
        assert d.evidence == []
    for a in result.action_items:
        assert a.evidence == []
    for t in result.topics:
        assert t.evidence == []
    for x in result.contradictions:
        assert x.evidence == []

    print("  PASS: multiple items in all categories converted into MeetingIntelligence")


def test_extraction_prompt_explicitly_instructs_multiple_items():
    """The extraction prompt must explicitly instruct extracting all items and use multi-entry arrays."""
    transcript = _make_transcript()
    prompt = _build_prompt(transcript)

    # Must contain instruction for exhaustive extraction
    assert "Extract ALL distinct claims, decisions, action items, topics, and contradictions" in prompt
    assert "Do NOT limit any category to a single item" in prompt
    assert "Never limit any category to one item" in prompt
    assert "Use multiple array entries whenever multiple items of that category exist" in prompt

    # Prompt schema template must illustrate multi-item arrays
    assert '"c1"' in prompt and '"c2"' in prompt
    assert '"d1"' in prompt and '"d2"' in prompt
    assert '"a1"' in prompt and '"a2"' in prompt
    assert '"t1"' in prompt and '"t2"' in prompt
    assert '"x1"' in prompt and '"x2"' in prompt
    print("  PASS: extraction prompt explicitly instructs multiple items")


# -- Tests: Evidence Must Be Empty -----------------------------------------

def test_no_fabricated_evidence():
    """LLM extraction must NOT produce any evidence — all lists empty."""
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_valid)

    for claim in result.claims:
        assert claim.evidence == [], f"Claim {claim.id} has fabricated evidence!"
    for decision in result.decisions:
        assert decision.evidence == [], f"Decision {decision.id} has fabricated evidence!"
    for action in result.action_items:
        assert action.evidence == [], f"ActionItem {action.id} has fabricated evidence!"
    for topic in result.topics:
        assert topic.evidence == [], f"Topic {topic.id} has fabricated evidence!"
    for contradiction in result.contradictions:
        assert contradiction.evidence == [], f"Contradiction {contradiction.id} has fabricated evidence!"
    print("  PASS: no fabricated evidence in any extracted item")


# -- Tests: Nullable Fields ------------------------------------------------

def test_nullable_owner_and_deadline():
    """Action items with null/unknown owner/deadline should become None."""
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_null_owner_deadline)

    a1 = result.action_items[0]
    assert a1.owner is None, f"Expected None, got {a1.owner!r}"
    assert a1.deadline is None, f"Expected None, got {a1.deadline!r}"

    a2 = result.action_items[1]
    assert a2.owner is None, f"Expected None for 'null' string, got {a2.owner!r}"
    assert a2.deadline is None, f"Expected None for 'unknown' string, got {a2.deadline!r}"
    print("  PASS: null/unknown owner and deadline become None")


# -- Tests: LLM Response Cleaning ------------------------------------------

def test_markdown_fences_stripped():
    """JSON wrapped in markdown fences should be handled."""
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_with_markdown)
    assert len(result.claims) == 2
    print("  PASS: markdown fences stripped from response")


def test_think_tags_stripped():
    """Qwen3 <think> tags should be stripped before parsing."""
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_with_think_tags)
    assert len(result.claims) == 2
    print("  PASS: Qwen3 <think> tags stripped from response")


# -- Tests: Error Handling --------------------------------------------------

def test_malformed_json_raises_extraction_error():
    """Invalid JSON from LLM should raise ExtractionError, not crash."""
    transcript = _make_transcript()
    try:
        extract_intelligence(transcript, generate_fn=_mock_generate_malformed)
        print("  FAIL: should have raised ExtractionError")
        return
    except ExtractionError:
        pass
    print("  PASS: malformed JSON raises ExtractionError")


def test_empty_transcript_returns_empty_intelligence():
    """Empty transcript should return empty MeetingIntelligence without calling LLM."""
    empty = {"meeting_id": "empty", "language": "en", "segments": []}

    def _should_not_be_called(**kwargs):
        raise AssertionError("LLM should not be called for empty transcript")

    result = extract_intelligence(empty, generate_fn=_should_not_be_called)
    assert isinstance(result, MeetingIntelligence)
    assert len(result.claims) == 0
    assert len(result.decisions) == 0
    assert len(result.action_items) == 0
    assert len(result.topics) == 0
    assert len(result.contradictions) == 0
    print("  PASS: empty transcript returns empty MeetingIntelligence")


def test_empty_lists_from_llm():
    """LLM returning all empty lists should produce empty MeetingIntelligence."""
    transcript = _make_transcript()
    result = extract_intelligence(transcript, generate_fn=_mock_generate_empty_lists)
    assert len(result.claims) == 0
    assert len(result.decisions) == 0
    print("  PASS: empty LLM lists produce empty MeetingIntelligence")


# -- Tests: Input Immutability ----------------------------------------------

def test_transcript_not_mutated():
    transcript = _make_transcript()
    original = copy.deepcopy(transcript)
    extract_intelligence(transcript, generate_fn=_mock_generate_valid)
    assert transcript == original
    print("  PASS: transcript not mutated by extraction")


# -- Tests: JSON Cleaning Utility -------------------------------------------

def test_clean_json_strips_fences():
    raw = "```json\n{\"a\": 1}\n```"
    assert _clean_json_response(raw) == '{"a": 1}'
    print("  PASS: _clean_json_response strips markdown fences")


def test_clean_json_strips_think_tags():
    raw = "<think>thinking...</think>\n{\"a\": 1}"
    assert _clean_json_response(raw) == '{"a": 1}'
    print("  PASS: _clean_json_response strips think tags")


def test_clean_json_handles_plain():
    raw = '{"a": 1}'
    assert _clean_json_response(raw) == '{"a": 1}'
    print("  PASS: _clean_json_response handles plain JSON")


# -- Runner ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Running Intelligence Extractor Tests")
    print("=" * 60)

    print("\n--- Valid Extraction ---")
    test_valid_qwen_json_extraction()
    test_claims_conversion()
    test_decisions_conversion()
    test_action_items_conversion()
    test_topics_conversion()
    test_contradictions_conversion()
    test_multiple_items_conversion_all_categories()
    test_extraction_prompt_explicitly_instructs_multiple_items()

    print("\n--- Evidence Safety ---")
    test_no_fabricated_evidence()

    print("\n--- Nullable Fields ---")
    test_nullable_owner_and_deadline()

    print("\n--- LLM Response Cleaning ---")
    test_markdown_fences_stripped()
    test_think_tags_stripped()

    print("\n--- Error Handling ---")
    test_malformed_json_raises_extraction_error()
    test_empty_transcript_returns_empty_intelligence()
    test_empty_lists_from_llm()

    print("\n--- Immutability ---")
    test_transcript_not_mutated()

    print("\n--- JSON Cleaning Utility ---")
    test_clean_json_strips_fences()
    test_clean_json_strips_think_tags()
    test_clean_json_handles_plain()

    print("\n" + "=" * 60)
    print("[OK] ALL EXTRACTOR TESTS PASSED")
    print("=" * 60)
