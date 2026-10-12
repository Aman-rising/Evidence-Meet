"""
Unit tests for timing precision and duration formatting in EvidenceMeet.

Verifies:
1. format_duration() formats durations under 1 second in milliseconds (e.g., '0.44 ms').
2. format_duration() formats durations of 1 second or more in seconds (e.g., '21.72 s').
3. Sub-millisecond, millisecond, edge cases (zero, negative, None, boundary at 1.0s),
   and multi-second values are formatted accurately.
4. run_intelligence_pipeline preserves full float precision in its timing dictionary
   without premature rounding.
5. stage_callback receives full precision durations.
"""

import os
import sys
import json
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from member2_intelligence.pipeline import (
    format_duration,
    run_intelligence_pipeline,
)


# ===========================================================================
# 1. format_duration Unit Tests
# ===========================================================================

def test_format_duration_sub_millisecond():
    """Durations under 1 millisecond (< 0.001s) should display in milliseconds with two decimals."""
    # Typical in-memory alignment duration: ~0.44 ms
    assert format_duration(0.00044) == "0.44 ms"
    assert format_duration(0.000435) == "0.43 ms"
    assert format_duration(0.000100) == "0.10 ms"
    assert format_duration(0.000042) == "0.04 ms"
    assert format_duration(0.000001) == "0.00 ms"


def test_format_duration_millisecond():
    """Durations from 1 ms to 999 ms should display in milliseconds."""
    # Typical regex preprocessing duration: ~1.19 ms
    assert format_duration(0.001194) == "1.19 ms"
    # Typical evidence retrieval duration: ~15 to 80 ms
    assert format_duration(0.0150) == "15.00 ms"
    assert format_duration(0.0800) == "80.00 ms"
    assert format_duration(0.5000) == "500.00 ms"
    assert format_duration(0.9990) == "999.00 ms"
    assert format_duration(0.9999) == "999.90 ms"


def test_format_duration_edge_cases():
    """Zero, negative, None, and boundary at 1.0s."""
    assert format_duration(0.0) == "0.00 ms"
    assert format_duration(None) == "0.00 ms"
    assert format_duration(-0.5) == "0.00 ms"
    # Exactly 1.0 second boundary
    assert format_duration(1.0) == "1.00 s"


def test_format_duration_multi_second():
    """Durations >= 1.0s should display in seconds with two decimals."""
    assert format_duration(1.05) == "1.05 s"
    assert format_duration(7.12) == "7.12 s"
    assert format_duration(21.72) == "21.72 s"
    assert format_duration(77.05) == "77.05 s"
    assert format_duration(334.43) == "334.43 s"
    assert format_duration(454.98) == "454.98 s"


# ===========================================================================
# 2. Pipeline Full-Precision Timing Tests
# ===========================================================================

def test_pipeline_preserves_full_precision_timing():
    """
    Pipeline's timing dictionary must keep raw full float precision values
    and must not round them to 2 decimal places before returning.
    """
    aligned = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "Deploy the backend service by Friday."},
        {"speaker": "SPEAKER_01", "start": 2.8, "end": 5.0, "text": "I agree with the Friday deadline."},
    ]

    mock_qwen_json = json.dumps({
        "claims": [{"id": "c1", "text": "Deploy the backend service by Friday."}],
        "decisions": [],
        "action_items": [],
        "topics": [],
        "contradictions": [],
    })

    def mock_generate(**kwargs):
        return mock_qwen_json

    def mock_summarize(text: str) -> str:
        return "Backend deploy Friday."

    # Provide high-precision Member 1 timing values
    m1_timing = {
        "transcription": 12.345678,
        "diarization": 45.678912,
        "alignment": 0.00044,  # Sub-millisecond alignment duration!
    }

    result = run_intelligence_pipeline(
        aligned_transcript=aligned,
        language="en",
        audio_filename="test_timing.wav",
        generate_fn=mock_generate,
        summarize_fn=mock_summarize,
        member1_timing=m1_timing,
    )

    timing = result["timing"]

    # 1. Member 1 timings must NOT be prematurely rounded
    assert timing["alignment"] == 0.00044, (
        f"Alignment timing was truncated/rounded: {timing['alignment']} (expected 0.00044)"
    )
    assert timing["transcription"] == 12.345678
    assert timing["diarization"] == 45.678912

    # 2. Member 2 preprocessing duration must be preserved as unrounded float
    assert "preprocessing" in timing
    assert isinstance(timing["preprocessing"], float)
    assert timing["preprocessing"] > 0.0
    # A rounded value would be 0.0; full precision float must be non-zero
    assert timing["preprocessing"] != 0.0

    # 3. Evidence retrieval duration must be float
    assert "evidence" in timing
    assert isinstance(timing["evidence"], float)

    # 4. Total must be the unrounded sum of all stage durations
    all_stages_sum = sum(v for k, v in timing.items() if k != "total")
    assert abs(timing["total"] - all_stages_sum) < 1e-9

    # 5. Format durations from the timing dict
    assert format_duration(timing["alignment"]) == "0.44 ms"
    assert "ms" in format_duration(timing["preprocessing"])


def test_pipeline_stage_callback_receives_unrounded_durations():
    """stage_callback must receive raw unrounded float durations."""
    aligned = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "Quick sync."},
    ]

    mock_qwen_json = json.dumps({
        "claims": [],
        "decisions": [],
        "action_items": [],
        "topics": [],
        "contradictions": [],
    })

    recorded_stages = {}

    def _callback(stage_name, duration):
        recorded_stages[stage_name] = duration

    run_intelligence_pipeline(
        aligned_transcript=aligned,
        language="en",
        audio_filename="callback_test.wav",
        generate_fn=lambda **kw: mock_qwen_json,
        summarize_fn=lambda txt: "Quick summary.",
        stage_callback=_callback,
    )

    assert "preprocessing" in recorded_stages
    prep_dur = recorded_stages["preprocessing"]
    assert isinstance(prep_dur, float)
    assert prep_dur > 0.0
    # Must be unrounded float
    assert prep_dur != 0.0
