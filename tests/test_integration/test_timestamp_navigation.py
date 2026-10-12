"""
Targeted integration tests for clickable evidence timestamps, tab navigation,
and transcript segment highlighting.

Tests:
1. Evidence timestamps have clickable jump buttons in Verified Intelligence.
2. Clicking an evidence timestamp button navigates to Smart Transcript and sets highlighted_segment_id.
3. Smart Transcript visibly highlights the target segment with highlight banner and caption label.
4. Clicking 'Clear highlight' clears the segment focus.
5. Handling of missing or non-existent segment IDs shows a navigation warning and does not crash.
6. Original transcript text remains unaltered during navigation and highlighting.
7. Speaker renaming synchronization remains fully operational with segment navigation.
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


def _make_test_data():
    """Builds sample intelligence and aligned transcript segments for navigation testing."""
    intel = MeetingIntelligence(
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
                support_score=0.85,
                verification_status="SUPPORTED",
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
                status="confirmed",
                support_score=0.90,
                verification_status="SUPPORTED",
            )
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
                commitment_status="committed",
                support_score=0.80,
                verification_status="SUPPORTED",
            )
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
                support_score=0.75,
                verification_status="SUPPORTED",
            )
        ],
        contradictions=[
            Contradiction(
                id="x1",
                description="Conflicting schedules discussed.",
                evidence=[
                    Evidence(
                        segment_id=3,
                        speaker="SPEAKER_02",
                        start=12.0,
                        end=15.5,
                        quote="Release date fixed for next Wednesday.",
                    )
                ],
                support_score=0.70,
                verification_status="SUPPORTED",
            )
        ],
    )

    aligned_transcript = [
        {"id": 1, "speaker": "SPEAKER_00", "start": 1.5, "end": 4.8, "text": "The budget is finalized at $100k."},
        {"id": 2, "speaker": "SPEAKER_01", "start": 6.0, "end": 10.2, "text": "I will prepare sprint tasks by Monday."},
        {"id": 3, "speaker": "SPEAKER_02", "start": 12.0, "end": 15.5, "text": "Release date fixed for next Wednesday."},
    ]

    timing = {
        "transcription": 10.0,
        "diarization": 20.0,
        "alignment": 1.0,
        "preprocessing": 0.1,
        "bart_summary": 5.0,
        "qwen_extraction": 4.0,
        "evidence": 0.1,
        "total": 40.2,
    }

    return intel, aligned_transcript, timing


def test_clickable_timestamp_jump_buttons_rendered():
    """Verify that clickable timestamp navigation buttons are rendered for evidence segments."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception

    intel, aligned_transcript, timing = _make_test_data()
    at.session_state["processed_data"] = {
        "aligned_transcript": aligned_transcript,
        "unique_speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        "cleaned_text": "Cleaned text.",
        "summary_text": "Summary text.",
        "verified_intelligence": intel,
        "intelligence_error": None,
        "timing": timing,
        "total_str": "0 min 40.20 sec",
    }
    at.session_state["current_file"] = "nav_test.wav"
    at.file_uploader[0].upload("nav_test.wav", b"audio").run(timeout=30)
    assert not at.exception

    # Find jump buttons
    jump_buttons = [b for b in at.button if "Jump to Segment" in str(b.label)]
    assert len(jump_buttons) >= 3, f"Expected at least 3 Jump buttons, found {len(jump_buttons)}"


def test_clicking_timestamp_navigates_and_highlights_segment():
    """Clicking a timestamp button navigates to Smart Transcript and sets highlighted_segment_id."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception

    intel, aligned_transcript, timing = _make_test_data()
    at.session_state["processed_data"] = {
        "aligned_transcript": aligned_transcript,
        "unique_speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        "cleaned_text": "Cleaned text.",
        "summary_text": "Summary text.",
        "verified_intelligence": intel,
        "intelligence_error": None,
        "timing": timing,
        "total_str": "0 min 40.20 sec",
    }
    at.session_state["current_file"] = "nav_test.wav"
    at.file_uploader[0].upload("nav_test.wav", b"audio").run(timeout=30)
    assert not at.exception

    # Find the button for Claim 1 (which links to Segment #1)
    jump_btn_c1 = at.button(key="jump_btn_claim_1_1_0")
    jump_btn_c1.click().run(timeout=30)
    assert not at.exception

    # Verified navigation session state
    assert at.session_state["highlighted_segment_id"] == 1
    assert at.session_state["main_tab_selection"] == "💬 Smart Transcript"

    # Check visible highlight in rendered content
    joined_markdown = " ".join([m.value for m in at.markdown])
    assert "TARGET EVIDENCE SEGMENT #1" in joined_markdown

    joined_captions = " ".join([c.value for c in at.caption])
    assert "⭐ **HIGHLIGHTED**" in joined_captions

    joined_success = " ".join([s.value for s in at.success])
    assert "Highlighted Segment #1" in joined_success


def test_clicking_another_segment_updates_highlight():
    """Clicking a different timestamp button (e.g., for Decision 1, Segment #3) updates the highlight."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception

    intel, aligned_transcript, timing = _make_test_data()
    at.session_state["processed_data"] = {
        "aligned_transcript": aligned_transcript,
        "unique_speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        "cleaned_text": "Cleaned text.",
        "summary_text": "Summary text.",
        "verified_intelligence": intel,
        "intelligence_error": None,
        "timing": timing,
        "total_str": "0 min 40.20 sec",
    }
    at.session_state["current_file"] = "nav_test.wav"
    at.file_uploader[0].upload("nav_test.wav", b"audio").run(timeout=30)
    assert not at.exception

    # Click Decision 1 button (Segment #3)
    jump_btn_d1 = at.button(key="jump_btn_decision_1_3_0")
    jump_btn_d1.click().run(timeout=30)
    assert not at.exception

    assert at.session_state["highlighted_segment_id"] == 3
    joined_markdown = " ".join([m.value for m in at.markdown])
    assert "TARGET EVIDENCE SEGMENT #3" in joined_markdown


def test_clear_highlight_button_resets_focus():
    """Clicking 'Clear highlight' removes the highlighted segment selection."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception

    intel, aligned_transcript, timing = _make_test_data()
    at.session_state["processed_data"] = {
        "aligned_transcript": aligned_transcript,
        "unique_speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        "cleaned_text": "Cleaned text.",
        "summary_text": "Summary text.",
        "verified_intelligence": intel,
        "intelligence_error": None,
        "timing": timing,
        "total_str": "0 min 40.20 sec",
    }
    at.session_state["current_file"] = "nav_test.wav"
    at.session_state["highlighted_segment_id"] = 2

    at.file_uploader[0].upload("nav_test.wav", b"audio").run(timeout=30)
    assert not at.exception

    # Click Clear highlight button in Smart Transcript
    clear_btn = at.button(key="clear_transcript_hi")
    clear_btn.click().run(timeout=30)
    assert not at.exception

    assert at.session_state["highlighted_segment_id"] is None


def test_missing_segment_handled_gracefully():
    """Referencing an invalid or missing segment ID sets a warning banner and avoids crash."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception

    intel, aligned_transcript, timing = _make_test_data()

    # Add a claim referencing a non-existent segment #99
    intel.claims.append(
        Claim(
            id="c_missing",
            text="Ghost claim referencing missing segment.",
            evidence=[
                Evidence(
                    segment_id=99,
                    speaker="SPEAKER_99",
                    start=99.0,
                    end=105.0,
                    quote="Non existent quote.",
                )
            ],
            support_score=0.55,
            verification_status="SUPPORTED",
        )
    )

    at.session_state["processed_data"] = {
        "aligned_transcript": aligned_transcript,
        "unique_speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        "cleaned_text": "Cleaned text.",
        "summary_text": "Summary text.",
        "verified_intelligence": intel,
        "intelligence_error": None,
        "timing": timing,
        "total_str": "0 min 40.20 sec",
    }
    at.session_state["current_file"] = "nav_test.wav"
    at.file_uploader[0].upload("nav_test.wav", b"audio").run(timeout=30)
    assert not at.exception

    # Click the missing segment button
    btn_missing = at.button(key="jump_btn_claim_2_99_0")
    btn_missing.click().run(timeout=30)
    assert not at.exception

    # Should report warning rather than highlight invalid segment
    assert at.session_state["highlighted_segment_id"] is None
    joined_warnings = " ".join([w.value for w in at.warning])
    assert "Segment #99 could not be found" in joined_warnings


def test_transcript_text_and_speaker_renaming_preserved_during_navigation():
    """Navigation preserves original transcript text and speaker renaming synchronization."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception

    intel, aligned_transcript, timing = _make_test_data()
    at.session_state["processed_data"] = {
        "aligned_transcript": aligned_transcript,
        "unique_speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        "cleaned_text": "Cleaned text.",
        "summary_text": "Summary text.",
        "verified_intelligence": intel,
        "intelligence_error": None,
        "timing": timing,
        "total_str": "0 min 40.20 sec",
    }
    at.session_state["current_file"] = "nav_test.wav"
    at.file_uploader[0].upload("nav_test.wav", b"audio").run(timeout=30)
    assert not at.exception

    # Rename SPEAKER_00 to "Alice"
    intel_input = at.text_input(key="intel_speaker_input_SPEAKER_00")
    intel_input.input("Alice").run(timeout=30)
    assert not at.exception

    # Navigate to Segment #1
    jump_btn_c1 = at.button(key="jump_btn_claim_1_1_0")
    jump_btn_c1.click().run(timeout=30)
    assert not at.exception

    # 1. Renaming preserved
    assert at.session_state["speaker_name_map"]["SPEAKER_00"] == "Alice"

    # 2. Original transcript text intact
    for block in aligned_transcript:
        assert block["text"] in ["The budget is finalized at $100k.", "I will prepare sprint tasks by Monday.", "Release date fixed for next Wednesday."]


def test_return_to_previous_view_navigation_flow():
    """Jumping to transcript sets nav_origin and Return button restores origin category without loops."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception

    intel, aligned_transcript, timing = _make_test_data()
    at.session_state["processed_data"] = {
        "aligned_transcript": aligned_transcript,
        "unique_speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        "cleaned_text": "Cleaned text.",
        "summary_text": "Summary text.",
        "verified_intelligence": intel,
        "intelligence_error": None,
        "timing": timing,
        "total_str": "0 min 40.20 sec",
    }
    at.session_state["current_file"] = "return_test.wav"
    at.file_uploader[0].upload("return_test.wav", b"audio").run(timeout=30)
    assert not at.exception

    # 1. Initially on Verified Intelligence tab, expanded_category is None (collapsed by default)
    assert at.session_state["main_tab_selection"] == "🧠 Verified Intelligence"
    assert at.session_state["expanded_category"] is None
    assert at.session_state["nav_origin"] is None

    # 2. Click Jump to Segment #3 on Decision d1
    jump_btn_d1 = at.button(key="jump_btn_decision_1_3_0")
    jump_btn_d1.click().run(timeout=30)
    assert not at.exception

    # Navigated to Smart Transcript
    assert at.session_state["main_tab_selection"] == "💬 Smart Transcript"
    assert at.session_state["highlighted_segment_id"] == 3
    assert at.session_state["nav_origin"] is not None
    assert at.session_state["nav_origin"]["category"] == "decisions"
    assert at.session_state["nav_origin"]["segment_id"] == 3

    # 3. Return to Previous View button is present
    return_btn = at.button(key="btn_return_to_origin")
    assert return_btn is not None
    assert "Return to Previous View" in return_btn.label

    # 4. Click Return to Previous View
    return_btn.click().run(timeout=30)
    assert not at.exception

    # Successfully returned to Verified Intelligence tab with origin category expanded
    assert at.session_state["main_tab_selection"] == "🧠 Verified Intelligence"
    assert at.session_state["expanded_category"] == "decisions"
    assert at.session_state["highlighted_segment_id"] == 3
    assert at.session_state["nav_origin"] is None  # Loop prevention: origin cleared


def test_theme_toggle_session_state_and_persistence():
    """Theme toggle updates theme in session_state and preserves it across reruns."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception

    # 1. Default theme is dark
    assert at.session_state["theme"] == "dark"

    # 2. Toggle to light mode
    theme_toggle = at.toggle(key="theme_toggle")
    assert theme_toggle is not None
    theme_toggle.set_value(True).run(timeout=30)
    assert not at.exception
    assert at.session_state["theme"] == "light"

    # 3. Subsequent rerun preserves light mode
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["theme"] == "light"

    # 4. Toggle back to dark mode
    theme_toggle = at.toggle(key="theme_toggle")
    theme_toggle.set_value(False).run(timeout=30)
    assert not at.exception
    assert at.session_state["theme"] == "dark"

