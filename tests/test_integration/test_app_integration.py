"""Integration tests for Streamlit app/app.py.

Tests:
1. Initial app render state (file uploader prompt).
2. App render with processed meeting data, verified intelligence, and timing breakdown.
3. Distinguishing verified vs unverified intelligence items.
4. Display of evidence citation (speaker, timestamp, segment ID, quote).
5. All 8 pipeline timing values in Processing Performance section.
6. Safe handling of empty/missing intelligence and timing.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest
from member2_intelligence.intelligence import (
    MeetingIntelligence,
    Claim,
    Decision,
    ActionItem,
    Topic,
    Contradiction,
    Evidence,
)

APP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "app", "app.py")
)


def _make_dummy_intelligence() -> MeetingIntelligence:
    return MeetingIntelligence(
        claims=[
            Claim(
                id="c1",
                text="The budget is finalized at $100k.",
                evidence=[
                    Evidence(
                        segment_id=1,
                        speaker="SPEAKER_00",
                        start=1.5,
                        end=4.8,
                        quote="The budget is finalized at $100k.",
                    )
                ],
            ),
            Claim(
                id="c2",
                text="We will exceed the projected deadline.",
                evidence=[],  # Unverified / unsupported
            ),
        ],
        decisions=[
            Decision(
                id="d1",
                text="Release date fixed for next Wednesday.",
                evidence=[
                    Evidence(
                        segment_id=3,
                        speaker="SPEAKER_02",
                        start=12.0,
                        end=15.5,
                        quote="Release date fixed for next Wednesday.",
                    )
                ],
            ),
            Decision(
                id="d2",
                text="Postpone marketing campaign indefinitely.",
                evidence=[],  # Unverified
            ),
        ],
        action_items=[
            ActionItem(
                id="a1",
                task="Prepare sprint tasks by Monday",
                owner="SPEAKER_01",
                deadline="Monday",
                evidence=[
                    Evidence(
                        segment_id=2,
                        speaker="SPEAKER_01",
                        start=6.0,
                        end=10.2,
                        quote="I will prepare sprint tasks by Monday.",
                    )
                ],
            ),
        ],
        topics=[
            Topic(
                id="t1",
                name="Budget and Sprint Planning",
                evidence=[
                    Evidence(
                        segment_id=1,
                        speaker="SPEAKER_00",
                        start=1.5,
                        end=4.8,
                        quote="The budget is finalized at $100k.",
                    )
                ],
            ),
        ],
        contradictions=[
            Contradiction(
                id="x1",
                description="Speaker 0 and Speaker 2 disagree on timeline.",
                evidence=[
                    Evidence(
                        segment_id=3,
                        speaker="SPEAKER_02",
                        start=12.0,
                        end=15.5,
                        quote="Release date fixed for next Wednesday.",
                    )
                ],
            ),
        ],
    )


def test_app_initial_state():
    """App should run without exceptions in initial state and show upload prompt."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception, f"App threw exception on load: {at.exception}"
    # Verify main title and upload info exist
    assert any("EvidenceMeet: Meeting Intelligence" in str(t.value) for t in at.title)
    assert any("Upload an audio file" in str(i.value) for i in at.info)


def test_app_uploaded_file_pre_processing_state():
    """App should handle uploaded audio before processing without UnboundLocalError on tabs."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception
    at.file_uploader[0].upload("meeting_raw.wav", b"dummy audio bytes").run(timeout=30)
    assert not at.exception, f"App threw exception on file upload before processing: {at.exception}"
    # Process audio button should be visible
    assert any("Process Meeting Audio" in str(b.label) for b in at.button)


def test_app_with_processed_data_and_verified_intelligence():
    """App should render all 5 tabs, display verified intelligence, and timing metrics."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception, f"Initial run threw exception: {at.exception}"

    intel = _make_dummy_intelligence()
    timing = {
        "transcription": 12.34,
        "diarization": 24.56,
        "alignment": 0.85,
        "preprocessing": 0.05,
        "bart_summary": 7.12,
        "qwen_extraction": 4.50,
        "evidence": 0.08,
        "total": 49.50,
    }

    aligned_transcript = [
        {"speaker": "SPEAKER_00", "start": 1.5, "end": 4.8, "text": "The budget is finalized at $100k."},
        {"speaker": "SPEAKER_01", "start": 6.0, "end": 10.2, "text": "I will prepare sprint tasks by Monday."},
        {"speaker": "SPEAKER_02", "start": 12.0, "end": 15.5, "text": "Release date fixed for next Wednesday."},
    ]

    at.session_state["processed_data"] = {
        "aligned_transcript": aligned_transcript,
        "unique_speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        "cleaned_text": "Cleaned meeting transcript text.",
        "summary_text": "BART meeting summary baseline.",
        "verified_intelligence": intel,
        "intelligence_error": None,
        "timing": timing,
        "total_str": "0 min 49.50 sec",
    }
    at.session_state["current_file"] = "meeting.wav"

    # Simulate uploaded file
    at.file_uploader[0].upload("meeting.wav", b"dummy audio content").run(timeout=30)
    assert not at.exception, f"App threw exception with processed data: {at.exception}"

    # 1. Verify markdown content contains intelligence items and badges
    all_markdown = [m.value for m in at.markdown]
    joined_markdown = " ".join(all_markdown)

    # Claims
    assert "Factual Claims (2)" in joined_markdown
    assert "The budget is finalized at $100k." in joined_markdown
    assert "We will exceed the projected deadline." in joined_markdown

    # Decisions
    assert "Decisions (2)" in joined_markdown
    assert "Release date fixed for next Wednesday." in joined_markdown

    # Action Items
    assert "Action Items (1)" in joined_markdown
    assert "Prepare sprint tasks by Monday" in joined_markdown

    # Topics
    assert "Topics Discussed (1)" in joined_markdown
    assert "Budget and Sprint Planning" in joined_markdown

    # Contradictions
    assert "Contradictions & Conflicts (1)" in joined_markdown
    assert "Speaker 0 and Speaker 2 disagree on timeline." in joined_markdown

    # Verified vs Unverified badges
    assert "🟢 Verified" in joined_markdown
    assert "🔴 Unverified" in joined_markdown

    # Evidence details (speaker, timestamp, quote, segment)
    all_captions = [c.value for c in at.caption]
    joined_captions = " ".join(all_captions)
    assert "SPEAKER_00" in joined_captions
    assert "01.50s - 04.80s" in joined_captions
    assert "Segment #1" in joined_captions
    assert '"The budget is finalized at $100k."' in joined_markdown

    # Unsupported notice
    assert "No verified transcript evidence found" in joined_markdown

    # 2. Verify Summary and Cleaned text
    all_success = [s.value for s in at.success]
    assert any("BART meeting summary baseline." in s for s in all_success)
    assert any("Cleaned meeting transcript text." in str(t.value) for t in at.text_area)

    # 3. Verify Performance Metrics
    all_metrics = [(m.label, m.value) for m in at.metric]
    metric_labels = [m[0] for m in all_metrics]
    assert "Total Duration" in metric_labels
    assert "Speech Pipeline" in metric_labels
    assert "BART Summary" in metric_labels
    assert "Intelligence Stage" in metric_labels

    # 4. Verify Performance Table has all 8 timing stages
    assert len(at.table) >= 1
    table_data = at.table[0].value
    table_str = str(table_data)
    assert "Transcription" in table_str
    assert "Speaker Diarization" in table_str
    assert "Speaker Alignment" in table_str
    assert "Text Preprocessing" in table_str
    assert "Baseline Summary (BART)" in table_str
    assert "Qwen Extraction" in table_str
    assert "Evidence Retrieval & Verification" in table_str
    assert "Total Pipeline Execution" in table_str


def test_app_handles_empty_and_missing_intelligence_safely():
    """App should handle None/empty intelligence and missing timing safely."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception, f"Initial run threw exception: {at.exception}"

    at.session_state["processed_data"] = {
        "aligned_transcript": [],
        "unique_speakers": [],
        "cleaned_text": "",
        "summary_text": "",
        "verified_intelligence": None,
        "intelligence_error": "Qwen model endpoint connection timed out.",
        "timing": {},
        "total_str": "0 min 0.00 sec",
    }
    at.session_state["current_file"] = "test.wav"

    # Simulate uploaded file
    at.file_uploader[0].upload("test.wav", b"dummy audio content").run(timeout=30)
    assert not at.exception, f"App threw exception on empty data: {at.exception}"

    all_warnings = [w.value for w in at.warning]
    assert any("Qwen model endpoint connection timed out." in w for w in all_warnings)

    all_infos = [i.value for i in at.info]
    assert any("No verified intelligence items detected" in i for i in all_infos)

    # Performance table should still render with 0.00 default values
    assert len(at.table) >= 1
    table_str = str(at.table[0].value)
    assert "0.00" in table_str


def test_speaker_renaming_synchronization_across_tabs():
    """Renaming speakers from Verified Intelligence or Smart Transcript synchronizes across tabs
    while preserving original speaker IDs in the underlying data."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception, f"Initial run threw exception: {at.exception}"

    intel = _make_dummy_intelligence()
    timing = {
        "transcription": 12.34,
        "diarization": 24.56,
        "alignment": 0.85,
        "preprocessing": 0.05,
        "bart_summary": 7.12,
        "qwen_extraction": 4.50,
        "evidence": 0.08,
        "total": 49.50,
    }
    diar_timings = {
        "model_load": 2.15,
        "conversion": 0.35,
        "inference": 22.06,
        "total": 24.56,
    }

    aligned_transcript = [
        {"speaker": "SPEAKER_00", "start": 1.5, "end": 4.8, "text": "The budget is finalized at $100k."},
        {"speaker": "SPEAKER_01", "start": 6.0, "end": 10.2, "text": "I will prepare sprint tasks by Monday."},
        {"speaker": "SPEAKER_02", "start": 12.0, "end": 15.5, "text": "Release date fixed for next Wednesday."},
    ]

    at.session_state["processed_data"] = {
        "aligned_transcript": aligned_transcript,
        "unique_speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        "cleaned_text": "Cleaned meeting transcript text.",
        "summary_text": "BART meeting summary baseline.",
        "verified_intelligence": intel,
        "intelligence_error": None,
        "timing": timing,
        "total_str": "0 min 49.50 sec",
        "diar_timings": diar_timings,
    }
    at.session_state["current_file"] = "meeting.wav"

    at.file_uploader[0].upload("meeting.wav", b"dummy audio content").run(timeout=30)
    assert not at.exception, f"App threw exception with processed data: {at.exception}"

    # Step 1: Rename SPEAKER_00 from Verified Intelligence tab
    intel_input = at.text_input(key="intel_speaker_input_SPEAKER_00")
    intel_input.input("Alice").run(timeout=30)
    assert not at.exception

    # Verify session state mapping is synchronized
    assert at.session_state["speaker_name_map"]["SPEAKER_00"] == "Alice"
    assert at.session_state["speaker_input_SPEAKER_00"] == "Alice"

    # Verify Tab 1 evidence citation displays the renamed speaker alongside raw ID
    joined_captions = " ".join([c.value for c in at.caption])
    assert "**Alice** (SPEAKER_00)" in joined_captions

    # Step 2: Rename SPEAKER_01 from Smart Transcript tab
    transcript_input = at.text_input(key="speaker_input_SPEAKER_01")
    transcript_input.input("Bob").run(timeout=30)
    assert not at.exception

    # Verify session state mapping is synchronized to Tab 1
    assert at.session_state["speaker_name_map"]["SPEAKER_01"] == "Bob"
    assert at.session_state["intel_speaker_input_SPEAKER_01"] == "Bob"

    # Verify Tab 1 Action Item owner reflects Bob (SPEAKER_01)
    joined_captions = " ".join([c.value for c in at.caption])
    assert "Bob (SPEAKER_01)" in joined_captions

    # Verify underlying data structures preserved original IDs
    for block in aligned_transcript:
        assert block["speaker"] in ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    for claim in intel.claims:
        for ev in claim.evidence:
            assert ev.speaker in ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]


def test_diarization_timing_breakdown_display():
    """App should display granular diarization breakdown timings when available."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception

    diar_timings = {
        "model_load": 1.23,
        "conversion": 0.45,
        "inference": 15.67,
        "total": 17.35,
    }

    at.session_state["processed_data"] = {
        "aligned_transcript": [],
        "unique_speakers": ["SPEAKER_00"],
        "cleaned_text": "",
        "summary_text": "",
        "verified_intelligence": None,
        "intelligence_error": None,
        "timing": {"diarization": 17.35, "total": 17.35},
        "total_str": "0 min 17.35 sec",
        "diar_timings": diar_timings,
    }
    at.session_state["current_file"] = "meeting.wav"

    at.file_uploader[0].upload("meeting.wav", b"dummy audio content").run(timeout=30)
    assert not at.exception

    joined_captions = " ".join([c.value for c in at.caption])
    assert "Model Initialization:" in joined_captions or "Model Init:" in joined_captions
    assert "Audio Prep" in joined_captions
    assert "Inference:" in joined_captions

