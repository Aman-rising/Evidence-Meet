"""
Thread-isolation tests for EvidenceMeet CPU optimizations.

Verifies:
1. transcribe_audio() restores PyTorch thread count after completion so
   downstream PyAnnote diarization is not capped at the ASR-optimal 4 threads.
2. load_diarization_pipeline() stores num_inference_threads on the pipeline.
3. diarize_audio() applies the pipeline's thread count before inference and
   restores the caller's count after.
4. The default num_inference_threads=10 reflects the benchmark winner on
   i5-1235U (10 threads: RTF=0.896 vs. 4 threads: RTF=1.274).
"""

import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from member1_speech.diarization.diarizer import (
    load_diarization_pipeline,
    diarize_audio,
)


class MockTrack:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


def _make_mock_pipeline(num_threads: int = 10) -> MagicMock:
    mock_pipeline = MagicMock()
    mock_pipeline._evidencemeet_num_threads = num_threads
    mock_output = MagicMock()
    mock_output.speaker_diarization.itertracks.return_value = [
        (MockTrack(0.0, 2.5), None, "SPEAKER_00"),
        (MockTrack(3.0, 5.0), None, "SPEAKER_01"),
    ]
    mock_pipeline.return_value = mock_output
    return mock_pipeline


def test_transcribe_audio_restores_thread_count():
    from member1_speech.asr.transcriber import transcribe_audio
    torch.set_num_threads(10)
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.text = " Hello world."
    mock_segment.words = []
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model.transcribe.return_value = ([mock_segment], mock_info)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_audio = f.name
    try:
        with patch("member1_speech.asr.transcriber.WhisperModel", return_value=mock_model):
            transcribe_audio(temp_audio)
        restored = torch.get_num_threads()
        assert restored == 10, (
            f"transcribe_audio() leaked its thread count: expected 10, got {restored}"
        )
    finally:
        os.remove(temp_audio)


def test_transcribe_audio_applies_4_threads_internally():
    from member1_speech.asr.transcriber import transcribe_audio
    threads_during_transcription = []
    def capturing_transcribe(*args, **kwargs):
        threads_during_transcription.append(torch.get_num_threads())
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.text = " Test."
        mock_segment.words = []
        mock_info = MagicMock()
        mock_info.language = "en"
        return [mock_segment], mock_info
    mock_model = MagicMock()
    mock_model.transcribe.side_effect = capturing_transcribe
    torch.set_num_threads(10)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_audio = f.name
    try:
        with patch("member1_speech.asr.transcriber.WhisperModel", return_value=mock_model):
            transcribe_audio(temp_audio)
        assert len(threads_during_transcription) == 1
        assert threads_during_transcription[0] == 4, (
            f"Expected Whisper at 4 threads, got {threads_during_transcription[0]}"
        )
        assert torch.get_num_threads() == 10
    finally:
        os.remove(temp_audio)


def test_load_diarization_pipeline_stores_default_threads():
    with patch("member1_speech.diarization.diarizer.Pipeline.from_pretrained") as mock_fp:
        mock_instance = MagicMock()
        mock_fp.return_value = mock_instance
        pipeline = load_diarization_pipeline("hf_dummy_token")
        assert hasattr(pipeline, "_evidencemeet_num_threads")
        assert pipeline._evidencemeet_num_threads == 10, (
            f"Expected default 10 threads, got {pipeline._evidencemeet_num_threads}"
        )


def test_load_diarization_pipeline_stores_custom_threads():
    with patch("member1_speech.diarization.diarizer.Pipeline.from_pretrained") as mock_fp:
        mock_instance = MagicMock()
        mock_fp.return_value = mock_instance
        pipeline = load_diarization_pipeline("hf_dummy_token", num_inference_threads=8)
        assert pipeline._evidencemeet_num_threads == 8


def test_diarize_audio_applies_pipeline_thread_count():
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_audio = f.name
    try:
        # Use a simple object (not MagicMock) so _evidencemeet_num_threads stays as int.
        class MockPipeline:
            _evidencemeet_num_threads = 10

            def __init__(self):
                self._output = None

            def __call__(self, wav_path):
                threads_during_inference.append(torch.get_num_threads())
                return _make_mock_output()

        def _make_mock_output():
            mock_output = MagicMock()
            mock_output.speaker_diarization.itertracks.return_value = [
                (MockTrack(0.0, 2.5), None, "SPEAKER_00"),
            ]
            return mock_output

        threads_during_inference = []
        mock_pipe = MockPipeline()

        torch.set_num_threads(4)

        with patch("subprocess.run"), \
             patch("member1_speech.diarization.diarizer.load_diarization_pipeline", return_value=mock_pipe):
            diarize_audio(temp_audio, pipeline=mock_pipe, model_load_time=0.5, return_timings=True)

        assert torch.get_num_threads() == 4, (
            f"Thread count not restored: expected 4, got {torch.get_num_threads()}"
        )
        assert len(threads_during_inference) == 1
        assert threads_during_inference[0] == 10, (
            f"Expected inference at 10 threads, got {threads_during_inference[0]}"
        )
    finally:
        os.remove(temp_audio)
        cand = temp_audio.rsplit(".", 1)[0] + ".wav"
        if os.path.exists(cand):
            os.remove(cand)


def test_diarize_audio_restores_threads_even_without_evidencemeet_attr():
    """
    If a pipeline does not have a valid integer _evidencemeet_num_threads,
    diarize_audio() falls back to 10 and still restores the caller's thread count.
    With MagicMock, _evidencemeet_num_threads auto-creates as a MagicMock object;
    int(MagicMock()) raises TypeError, triggering the fallback.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_audio = f.name
    try:
        # MagicMock auto-creates _evidencemeet_num_threads as a MagicMock object,
        # which will fail int() conversion and trigger the fallback to 10.
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.speaker_diarization.itertracks.return_value = [
            (MockTrack(0.0, 1.0), None, "SPEAKER_00"),
        ]
        mock_pipe.return_value = mock_output
        torch.set_num_threads(7)
        with patch("subprocess.run"), \
             patch("member1_speech.diarization.diarizer.load_diarization_pipeline", return_value=mock_pipe):
            diarize_audio(temp_audio, pipeline=mock_pipe, model_load_time=0.0)
        assert torch.get_num_threads() == 7, (
            f"Thread count not restored: expected 7, got {torch.get_num_threads()}"
        )
    finally:
        os.remove(temp_audio)
        cand = temp_audio.rsplit(".", 1)[0] + ".wav"
        if os.path.exists(cand):
            os.remove(cand)


def test_benchmark_evidence_10_threads_is_faster_than_4():
    BENCHMARK_10_THREADS_RTF = 0.896
    BENCHMARK_4_THREADS_RTF = 1.274
    BENCHMARK_COMMUNITY1_RTF = 1.448
    BASELINE_SPEAKERS_DETECTED = 2
    COMMUNITY1_SPEAKERS_DETECTED = 1
    assert BENCHMARK_10_THREADS_RTF < BENCHMARK_4_THREADS_RTF
    assert BENCHMARK_COMMUNITY1_RTF > BENCHMARK_10_THREADS_RTF
    assert COMMUNITY1_SPEAKERS_DETECTED < BASELINE_SPEAKERS_DETECTED
    import inspect
    sig = inspect.signature(load_diarization_pipeline)
    default_threads = sig.parameters["num_inference_threads"].default
    assert default_threads == 10, (
        f"Default num_inference_threads={default_threads}, not 10. "
        "Change only after re-running benchmark with new measurements."
    )
