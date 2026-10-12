"""
Integration tests for the Member 2 intelligence pipeline.

Tests the full flow: Qwen extraction → evidence retrieval → verification
→ attachment, all using mocked LLM responses. Does NOT require Ollama.

Also verifies that existing pipeline outputs (transcript, cleaned_text,
summary) are preserved unchanged.
"""
import sys
import os
import copy
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from member2_intelligence.pipeline import (
    run_intelligence_pipeline,
    attach_evidence,
    build_structured_transcript,
)
from member2_intelligence.intelligence.schemas import (
    MeetingIntelligence,
    Claim,
    Decision,
    ActionItem,
    Topic,
    Contradiction,
    Evidence,
)
from member2_intelligence.intelligence.extractor import extract_intelligence


# -- Fixtures ---------------------------------------------------------------

def _make_aligned_transcript():
    """Simulates Member 1's align_and_merge() output."""
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.5,
         "text": "The budget for Q3 has been approved by the board."},
        {"speaker": "SPEAKER_01", "start": 6.0, "end": 12.0,
         "text": "We need to hire two engineers before Friday."},
        {"speaker": "SPEAKER_02", "start": 13.0, "end": 18.5,
         "text": "I disagree. The budget should not have been approved."},
    ]


# Qwen response with items that should match transcript segments
_VALID_QWEN_RESPONSE = json.dumps({
    "claims": [
        {"id": "c1", "text": "The Q3 budget has been approved by the board."},
        {"id": "c2", "text": "Two engineers need to be hired before Friday."},
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
        {"id": "t2", "name": "Hiring engineers"},
    ],
    "contradictions": [
        {"id": "x1", "description": "SPEAKER_00 says budget was approved; SPEAKER_02 disagrees."},
    ],
})


def _mock_generate_valid(**kwargs):
    """Mock returning valid Qwen JSON."""
    return _VALID_QWEN_RESPONSE


def _mock_generate_malformed(**kwargs):
    """Mock returning broken JSON."""
    return "NOT JSON AT ALL {{"


# Response with unrelated content — should not match transcript segments
_UNRELATED_QWEN_RESPONSE = json.dumps({
    "claims": [
        {"id": "c1", "text": "Quantum computing will revolutionize cryptography."},
    ],
    "decisions": [],
    "action_items": [],
    "topics": [],
    "contradictions": [],
})


def _mock_generate_unrelated(**kwargs):
    return _UNRELATED_QWEN_RESPONSE


# -- Tests: Full Pipeline Integration --------------------------------------

def test_pipeline_produces_intelligence():
    """run_intelligence_pipeline should produce a MeetingIntelligence object."""
    aligned = _make_aligned_transcript()
    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_valid,
    )
    assert "intelligence" in result
    assert "verified_intelligence" in result
    assert isinstance(result["intelligence"], MeetingIntelligence)
    assert isinstance(result["verified_intelligence"], MeetingIntelligence)
    assert result["verified_intelligence"] is result["intelligence"]
    assert result["intelligence_error"] is None
    print("  PASS: pipeline produces MeetingIntelligence")


def test_pipeline_returns_both_summary_and_verified_intelligence():
    """Pipeline must return both the baseline BART summary and verified intelligence."""
    aligned = _make_aligned_transcript()
    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_valid,
    )
    # Check baseline summary
    assert "summary" in result
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0

    # Check verified intelligence
    assert "verified_intelligence" in result
    intel = result["verified_intelligence"]
    assert isinstance(intel, MeetingIntelligence)
    assert len(intel.claims) > 0
    assert len(intel.claims[0].evidence) > 0

    # raw_text must remain unchanged
    for seg in result["transcript"]["segments"]:
        assert "raw_text" in seg
        assert len(seg["raw_text"]) > 0
    print("  PASS: pipeline returns both BART baseline summary and verified intelligence")


def test_backend_e2e_run_intelligence_pipeline():
    """
    Backend end-to-end integration test for run_intelligence_pipeline().

    Verifies the full Member 2 execution path:
        aligned transcript
                ↓
        build_structured_transcript()
                ↓
        preprocess_transcript()
                ↓
        BART baseline summary (mocked for determinism/speed)
                ↓
        extract_verified_intelligence() (mocked Qwen)
                ↓
        return {transcript, cleaned_text, summary, verified_intelligence}

    Validates:
    - Result contains: transcript, cleaned_text, summary, verified_intelligence
    - Verified intelligence has claims, decisions, action items with transcript evidence
    - raw_text is preserved as evidence quote and matches source segments exactly
    - Preprocessed transcript preserves segment raw_text and structure
    """
    # 1. Small realistic structured transcript fixture
    aligned = [
        {
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 5.2,
            "text": "The budget for the AI project has been finalized at one hundred thousand dollars.",
        },
        {
            "speaker": "SPEAKER_01",
            "start": 5.8,
            "end": 11.4,
            "text": "I will prepare the sprint plan and assign development tasks by Monday.",
        },
        {
            "speaker": "SPEAKER_02",
            "start": 12.0,
            "end": 17.5,
            "text": "We agreed to deploy the new release candidate next Wednesday.",
        },
    ]

    # 2. Mock Qwen output with claim, decision, and action item
    mock_qwen_json = json.dumps({
        "claims": [
            {"id": "c1", "text": "The budget for the AI project has been finalized at one hundred thousand dollars."}
        ],
        "decisions": [
            {"id": "d1", "text": "We agreed to deploy the new release candidate next Wednesday."}
        ],
        "action_items": [
            {"id": "a1", "task": "Prepare the sprint plan and assign development tasks by Monday", "owner": "SPEAKER_01", "deadline": "Monday"}
        ],
        "topics": [
            {"id": "t1", "name": "AI project budget"}
        ],
        "contradictions": [],
    })

    def mock_generate(**kwargs):
        return mock_qwen_json

    # 3. Mock BART summarizer for fast, deterministic execution
    expected_summary = "The AI project budget was set at $100k, sprint tasks are due Monday, and release is Wednesday."

    def mock_summarize(text: str) -> str:
        return expected_summary

    # Execute the full pipeline
    result = run_intelligence_pipeline(
        aligned_transcript=aligned,
        language="en",
        audio_filename="meeting_sync_2026.wav",
        generate_fn=mock_generate,
        summarize_fn=mock_summarize,
    )

    # 4. Verify returned keys
    assert "transcript" in result, "Result must contain 'transcript'"
    assert "cleaned_text" in result, "Result must contain 'cleaned_text'"
    assert "summary" in result, "Result must contain 'summary'"
    assert "verified_intelligence" in result, "Result must contain 'verified_intelligence'"
    assert "intelligence" in result, "Result must contain 'intelligence' for compatibility"
    assert result["verified_intelligence"] is result["intelligence"]

    # Verify summary is returned
    assert result["summary"] == expected_summary

    # Verify cleaned_text is non-empty string
    assert isinstance(result["cleaned_text"], str)
    assert len(result["cleaned_text"]) > 0

    # Verify transcript structure and raw_text integrity
    transcript = result["transcript"]
    assert transcript["meeting_id"] == "meeting_sync_2026"
    assert transcript["language"] == "en"
    assert len(transcript["segments"]) == 3
    for idx, seg in enumerate(transcript["segments"]):
        assert seg["id"] == idx + 1
        assert seg["speaker"] == aligned[idx]["speaker"]
        assert seg["start"] == aligned[idx]["start"]
        assert seg["end"] == aligned[idx]["end"]
        # raw_text MUST match the original alignment text
        assert seg["raw_text"] == aligned[idx]["text"]

    # 5. Verify verified_intelligence contains verified items with transcript evidence
    intel = result["verified_intelligence"]
    assert isinstance(intel, MeetingIntelligence)

    # Check Claim
    assert len(intel.claims) == 1
    claim = intel.claims[0]
    assert claim.id == "c1"
    assert len(claim.evidence) >= 1
    ev_claim = claim.evidence[0]
    assert ev_claim.segment_id == 1
    assert ev_claim.speaker == "SPEAKER_00"
    # 6. Verify raw_text is preserved as the evidence quote
    assert ev_claim.quote == aligned[0]["text"]
    assert ev_claim.quote == transcript["segments"][0]["raw_text"]

    # Check Decision
    assert len(intel.decisions) == 1
    decision = intel.decisions[0]
    assert decision.id == "d1"
    assert len(decision.evidence) >= 1
    ev_dec = decision.evidence[0]
    assert ev_dec.segment_id == 3
    assert ev_dec.speaker == "SPEAKER_02"
    assert ev_dec.quote == aligned[2]["text"]

    # Check ActionItem
    assert len(intel.action_items) == 1
    action = intel.action_items[0]
    assert action.id == "a1"
    assert action.owner == "SPEAKER_01"
    assert action.deadline == "Monday"
    assert len(action.evidence) >= 1
    ev_act = action.evidence[0]
    assert ev_act.segment_id == 2
    assert ev_act.speaker == "SPEAKER_01"
    assert ev_act.quote == aligned[1]["text"]

    print("  PASS: backend end-to-end run_intelligence_pipeline succeeded with verified intelligence")


def test_pipeline_preserves_existing_outputs():
    """transcript, cleaned_text, and summary must still be present."""
    aligned = _make_aligned_transcript()
    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_valid,
    )
    assert "transcript" in result
    assert "cleaned_text" in result
    assert "summary" in result
    assert isinstance(result["transcript"], dict)
    assert isinstance(result["cleaned_text"], str)
    assert isinstance(result["summary"], str)
    print("  PASS: pipeline preserves transcript, cleaned_text, summary")


def test_qwen_extraction_occurs():
    """Extracted intelligence should contain items from the Qwen response."""
    aligned = _make_aligned_transcript()
    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_valid,
    )
    intel = result["intelligence"]
    assert len(intel.claims) == 2, f"Expected 2 claims, got {len(intel.claims)}"
    assert len(intel.decisions) == 1
    assert len(intel.action_items) == 2
    assert len(intel.topics) == 2
    assert len(intel.contradictions) == 1
    print("  PASS: Qwen extraction produced expected items")


def test_evidence_retrieval_occurs():
    """Extracted items should have evidence attached from the transcript."""
    aligned = _make_aligned_transcript()
    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_valid,
    )
    intel = result["intelligence"]
    # Claim "The Q3 budget has been approved by the board" should match segment 1
    c1 = intel.claims[0]
    assert len(c1.evidence) > 0, f"Claim c1 should have evidence, got {len(c1.evidence)}"
    print("  PASS: evidence retrieval attached evidence to items")


def test_only_verified_evidence_attached():
    """All attached evidence must pass structural verification."""
    aligned = _make_aligned_transcript()
    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_valid,
    )
    transcript = result["transcript"]
    intel = result["intelligence"]

    all_items = (
        intel.claims + intel.decisions + intel.action_items
        + intel.topics + intel.contradictions
    )
    for item in all_items:
        for ev in item.evidence:
            # Evidence.quote must match the transcript segment's raw_text
            seg = None
            for s in transcript["segments"]:
                if s["id"] == ev.segment_id:
                    seg = s
                    break
            assert seg is not None, f"Evidence points to missing segment {ev.segment_id}"
            assert ev.speaker == seg["speaker"], \
                f"Evidence speaker {ev.speaker} != segment speaker {seg['speaker']}"
            assert ev.start == seg["start"]
            assert ev.end == seg["end"]
            assert ev.quote == seg["raw_text"], \
                f"Evidence quote doesn't match raw_text"
    print("  PASS: all attached evidence is structurally verified")


def test_unrelated_query_no_evidence():
    """Items with no transcript overlap should have empty evidence."""
    aligned = _make_aligned_transcript()
    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_unrelated,
    )
    intel = result["intelligence"]
    c1 = intel.claims[0]
    assert len(c1.evidence) == 0, \
        f"Unrelated claim should have no evidence, got {len(c1.evidence)}"
    print("  PASS: unrelated items get no evidence attached")


def test_multiple_categories_work():
    """All intelligence categories should be populated and have evidence where appropriate."""
    aligned = _make_aligned_transcript()
    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_valid,
    )
    intel = result["intelligence"]

    # Topics should have evidence (topic names overlap with transcript)
    has_any_evidence = False
    for topic in intel.topics:
        if len(topic.evidence) > 0:
            has_any_evidence = True
    assert has_any_evidence, "At least one topic should have evidence"

    # Contradictions about "budget approved/disagrees" should get evidence
    assert len(intel.contradictions) == 1
    print("  PASS: multiple intelligence categories populated with evidence")


# -- Tests: Error Handling --------------------------------------------------

def test_malformed_qwen_returns_empty_intelligence():
    """Pipeline should not crash on malformed Qwen JSON."""
    aligned = _make_aligned_transcript()
    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_malformed,
    )
    intel = result["intelligence"]
    assert isinstance(intel, MeetingIntelligence)
    assert len(intel.claims) == 0
    assert result["intelligence_error"] is not None
    assert "failed" in result["intelligence_error"].lower()
    # BART baseline should still work
    assert isinstance(result["summary"], str)
    print("  PASS: malformed Qwen response handled gracefully")


def test_empty_transcript_no_extraction():
    """Empty aligned transcript should skip extraction entirely."""
    result = run_intelligence_pipeline(
        [], "en", "empty.wav",
        generate_fn=_mock_generate_valid,
    )
    intel = result["intelligence"]
    assert isinstance(intel, MeetingIntelligence)
    assert len(intel.claims) == 0
    assert result["intelligence_error"] is None
    print("  PASS: empty transcript produces empty intelligence")


# -- Tests: attach_evidence standalone --------------------------------------

def test_attach_evidence_standalone():
    """attach_evidence should work on a pre-built MeetingIntelligence."""
    transcript = build_structured_transcript(
        _make_aligned_transcript(), "en", "test.wav"
    )
    # Manually populate cleaned_text so retrieval works on raw_text
    for seg in transcript["segments"]:
        seg["cleaned_text"] = seg["raw_text"]

    intelligence = MeetingIntelligence(
        claims=[Claim(id="c1", text="The Q3 budget approved by the board.", evidence=[])],
    )
    attach_evidence(intelligence, transcript)
    # Should have found evidence from segment 1
    assert len(intelligence.claims[0].evidence) > 0
    print("  PASS: attach_evidence works standalone")


def test_attach_evidence_does_not_mutate_transcript():
    """attach_evidence must not modify the transcript."""
    transcript = build_structured_transcript(
        _make_aligned_transcript(), "en", "test.wav"
    )
    for seg in transcript["segments"]:
        seg["cleaned_text"] = seg["raw_text"]

    original = copy.deepcopy(transcript)
    intelligence = MeetingIntelligence(
        claims=[Claim(id="c1", text="budget approved board", evidence=[])],
    )
    attach_evidence(intelligence, transcript)
    assert transcript == original
    print("  PASS: attach_evidence does not mutate transcript")


# -- Tests: Pipeline Stage Timing -------------------------------------------

def test_pipeline_returns_timing():
    """run_intelligence_pipeline must return timing dict with Member 2 stages."""
    aligned = _make_aligned_transcript()
    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_valid,
    )
    assert "timing" in result
    timing = result["timing"]
    assert isinstance(timing, dict)
    for key in ("preprocessing", "bart_summary", "qwen_extraction", "evidence", "total"):
        assert key in timing, f"Missing key: {key}"
        assert isinstance(timing[key], (int, float))
        assert timing[key] >= 0.0
    print("  PASS: pipeline returns valid timing dictionary")


def test_pipeline_integrates_member1_timing():
    """Member 1 timings passed in should be incorporated into the timing dict and total."""
    aligned = _make_aligned_transcript()
    m1_timing = {
        "transcription": 18.42,
        "diarization": 43.71,
        "alignment": 1.18,
    }
    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_valid,
        member1_timing=m1_timing,
    )
    timing = result["timing"]
    for key in ("transcription", "diarization", "alignment", "preprocessing",
                "bart_summary", "qwen_extraction", "evidence", "total"):
        assert key in timing, f"Missing key: {key}"
    assert timing["transcription"] == 18.42
    assert timing["diarization"] == 43.71
    assert timing["alignment"] == 1.18
    # Total must include Member 1 + Member 2
    expected_sum = round(sum(v for k, v in timing.items() if k != "total"), 2)
    assert abs(timing["total"] - expected_sum) <= 0.05
    print("  PASS: pipeline integrates member1 timing into total")


def test_pipeline_stage_callback():
    """stage_callback should be invoked with stage name and positive duration."""
    aligned = _make_aligned_transcript()
    recorded_stages = []

    def _callback(stage_name, duration):
        recorded_stages.append((stage_name, duration))

    result = run_intelligence_pipeline(
        aligned, "en", "test_meeting.wav",
        generate_fn=_mock_generate_valid,
        stage_callback=_callback,
    )
    stage_names = [s[0] for s in recorded_stages]
    assert "preprocessing" in stage_names
    assert "bart_summary" in stage_names
    assert "qwen_extraction" in stage_names
    assert "evidence" in stage_names
    for name, dur in recorded_stages:
        assert isinstance(dur, (int, float))
        assert dur >= 0.0
    print("  PASS: stage_callback receives progress events")


# -- Runner ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Running Pipeline Integration Tests")
    print("=" * 60)

    print("\n--- Full Pipeline ---")
    test_pipeline_produces_intelligence()
    test_pipeline_returns_both_summary_and_verified_intelligence()
    test_backend_e2e_run_intelligence_pipeline()
    test_pipeline_preserves_existing_outputs()
    test_qwen_extraction_occurs()
    test_evidence_retrieval_occurs()
    test_only_verified_evidence_attached()
    test_unrelated_query_no_evidence()
    test_multiple_categories_work()

    print("\n--- Timing Tests ---")
    test_pipeline_returns_timing()
    test_pipeline_integrates_member1_timing()
    test_pipeline_stage_callback()

    print("\n--- Error Handling ---")
    test_malformed_qwen_returns_empty_intelligence()
    test_empty_transcript_no_extraction()

    print("\n--- attach_evidence Standalone ---")
    test_attach_evidence_standalone()
    test_attach_evidence_does_not_mutate_transcript()

    print("\n" + "=" * 60)
    print("[OK] ALL PIPELINE INTEGRATION TESTS PASSED")
    print("=" * 60)

