"""
Unit tests for PyAnnote diarization profiling, timing breakdown, and cached initialization.

All tests use mocks to avoid downloading model weights, network requests,
or running GPU/CPU intensive inference.
"""

import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from member1_speech.diarization.diarizer import (
    diarize_audio,
    load_diarization_pipeline,
)


class MockTrack:
    """Mock for turn in pyannote itertracks."""
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


def _make_mock_pipeline():
    """Builds a mock PyAnnote pipeline that returns deterministic speaker turns."""
    mock_pipeline = MagicMock()
    mock_output = MagicMock()
    # Mock itertracks yielding (turn, track, speaker)
    mock_output.speaker_diarization.itertracks.return_value = [
        (MockTrack(0.0, 3.5), None, "SPEAKER_00"),
        (MockTrack(3.8, 7.2), None, "SPEAKER_01"),
    ]
    mock_pipeline.return_value = mock_output
    return mock_pipeline


# ===========================================================================
# 1. Timing Breakdown & Profiling Tests
# ===========================================================================

def test_diarize_audio_backward_compatibility():
    """
    Calling diarize_audio(audio_path, hf_token) without return_timings
    must return only the list of speaker intervals (List[Dict]).
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_audio = f.name

    try:
        mock_pipe = _make_mock_pipeline()
        with patch("subprocess.run") as mock_subproc, \
             patch("member1_speech.diarization.diarizer.load_diarization_pipeline", return_value=mock_pipe):

            result = diarize_audio(temp_audio, "hf_mock_token")

            # Must return list, not tuple
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0] == {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5}
            assert result[1] == {"speaker": "SPEAKER_01", "start": 3.8, "end": 7.2}
            mock_subproc.assert_called_once()
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        cand = temp_audio.rsplit('.', 1)[0] + ".wav"
        if os.path.exists(cand):
            os.remove(cand)


def test_diarize_audio_timing_breakdown():
    """
    Calling with return_timings=True returns (speaker_intervals, timing_dict).
    The timing dict must contain conversion, model_load, inference, and total.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_audio = f.name

    try:
        mock_pipe = _make_mock_pipeline()
        with patch("subprocess.run"), \
             patch("member1_speech.diarization.diarizer.load_diarization_pipeline", return_value=mock_pipe):

            intervals, timings = diarize_audio(
                temp_audio,
                "hf_mock_token",
                return_timings=True,
            )

            assert isinstance(intervals, list)
            assert len(intervals) == 2

            assert isinstance(timings, dict)
            for k in ("conversion", "model_load", "inference", "total"):
                assert k in timings, f"Missing key: {k}"
                assert isinstance(timings[k], float)
                assert timings[k] >= 0.0

            # Total must equal the sum of the components
            expected_total = timings["conversion"] + timings["model_load"] + timings["inference"]
            assert abs(timings["total"] - expected_total) < 1e-5
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        cand = temp_audio.rsplit('.', 1)[0] + ".wav"
        if os.path.exists(cand):
            os.remove(cand)


def test_diarize_audio_with_preloaded_pipeline():
    """
    When a pre-loaded/cached pipeline is passed in, load_diarization_pipeline
    must NOT be called, and model_load_time must be accurately preserved.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_audio = f.name

    try:
        mock_pipe = _make_mock_pipeline()
        with patch("subprocess.run"), \
             patch("member1_speech.diarization.diarizer.load_diarization_pipeline") as mock_load:

            intervals, timings = diarize_audio(
                temp_audio,
                pipeline=mock_pipe,
                model_load_time=0.052,
                return_timings=True,
            )

            # load_diarization_pipeline should NOT have been invoked
            mock_load.assert_not_called()

            assert len(intervals) == 2
            assert timings["model_load"] == 0.052
            assert timings["inference"] >= 0.0
            assert timings["conversion"] >= 0.0
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        cand = temp_audio.rsplit('.', 1)[0] + ".wav"
        if os.path.exists(cand):
            os.remove(cand)


# ===========================================================================
# 2. Token Validation & Safe Initialization Tests
# ===========================================================================

def test_load_diarization_pipeline_validates_token():
    """Missing or empty Hugging Face token must raise ValueError immediately."""
    with pytest.raises(ValueError, match="Hugging Face token is required"):
        load_diarization_pipeline("")

    with pytest.raises(ValueError, match="Hugging Face token is required"):
        load_diarization_pipeline("   ")

    with pytest.raises(ValueError, match="Hugging Face token is required"):
        load_diarization_pipeline(None)


def test_load_diarization_pipeline_calls_pyannote():
    """Valid token initializes Pipeline.from_pretrained and transfers to device."""
    with patch("member1_speech.diarization.diarizer.Pipeline.from_pretrained") as mock_from_pretrained:
        mock_instance = MagicMock()
        mock_from_pretrained.return_value = mock_instance

        pipeline = load_diarization_pipeline("hf_dummy_valid_token")

        assert pipeline is mock_instance
        mock_from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-3.1",
            token="hf_dummy_valid_token",
        )
        mock_instance.to.assert_called_once()


def test_diarize_audio_missing_file_raises():
    """Non-existent audio file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        diarize_audio("non_existent_audio_file.wav", "hf_token")


# ===========================================================================
# 3. Streamlit Cached Resource Initialization Behavior
# ===========================================================================

def test_cached_pipeline_reuse_behavior():
    """
    Simulates the Streamlit caching mechanism: repeated calls with the same
    token return the cached model instance without re-executing model loading.
    """
    from app.app import get_cached_diarization_pipeline

    with patch("app.app.load_diarization_pipeline") as mock_loader:
        mock_pipe = MagicMock()
        mock_loader.return_value = mock_pipe

        # Clear any prior Streamlit cached resource state for this function
        get_cached_diarization_pipeline.clear()

        # Call 1: should invoke loader
        pipe1 = get_cached_diarization_pipeline("hf_test_token_123")
        assert pipe1 is mock_pipe
        assert mock_loader.call_count == 1

        # Call 2 with identical token: should hit cache and NOT invoke loader again
        pipe2 = get_cached_diarization_pipeline("hf_test_token_123")
        assert pipe2 is mock_pipe
        assert mock_loader.call_count == 1  # Unchanged!


def test_cached_pipeline_invalidation_on_token_change():
    """
    When the Hugging Face token changes, the cached function must invalidate
    the previous cached value and invoke loader with the new token.
    """
    from app.app import get_cached_diarization_pipeline

    with patch("app.app.load_diarization_pipeline") as mock_loader:
        pipe_a = MagicMock(name="pipe_a")
        pipe_b = MagicMock(name="pipe_b")
        mock_loader.side_effect = [pipe_a, pipe_b]

        get_cached_diarization_pipeline.clear()

        # Call with Token A
        res_a = get_cached_diarization_pipeline("hf_token_AAA")
        assert res_a is pipe_a
        assert mock_loader.call_count == 1
        mock_loader.assert_called_with("hf_token_AAA")

        # Call with different Token B: must reload with Token B
        res_b = get_cached_diarization_pipeline("hf_token_BBB")
        assert res_b is pipe_b
        assert mock_loader.call_count == 2
        mock_loader.assert_called_with("hf_token_BBB")

