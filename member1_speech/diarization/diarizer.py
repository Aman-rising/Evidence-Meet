import os
import shutil
import subprocess
import torch
import time

# --- WINDOWS PYTHON 3.8+ DLL FIX FOR TORCHCODEC ---
# Python 3.8+ ignores the system PATH for DLLs. We must explicitly inject it so it finds FFmpeg.
ffmpeg_exe_path = shutil.which("ffmpeg")
if not ffmpeg_exe_path:
    fallback_bin = r"C:\Users\Hp\Downloads\ffmpeg-9.0.2-full_build-shared\ffmpeg-9.0.2-full_build-shared\bin"
    if os.path.exists(os.path.join(fallback_bin, "ffmpeg.exe")):
        ffmpeg_exe_path = os.path.join(fallback_bin, "ffmpeg.exe")
        os.environ["PATH"] = fallback_bin + os.pathsep + os.environ.get("PATH", "")

if ffmpeg_exe_path:
    ffmpeg_bin_dir = os.path.dirname(ffmpeg_exe_path)
    try:
        os.add_dll_directory(ffmpeg_bin_dir)
    except AttributeError:
        pass # Ignored if not on Windows
else:
    print("[WARNING] FFmpeg not found in PATH! Please ensure FFmpeg is installed.")

from typing import Optional, List, Dict, Any, Tuple, Union

# Now we can safely import the audio libraries
from pyannote.audio import Pipeline


def load_diarization_pipeline(
    hf_token: str,
    model_name: str = "pyannote/speaker-diarization-3.1",
    device: Optional[torch.device] = None,
    num_inference_threads: int = 10,
) -> Any:
    """
    Loads and initializes the PyAnnote speaker diarization pipeline on the specified device.
    Validates token presence before loading.

    Args:
        hf_token: Hugging Face authentication token.
        model_name: PyAnnote pipeline model identifier.
        device: Target device (defaults to CPU).
        num_inference_threads: PyTorch thread count for diarization inference.
            Benchmark on i5-1235U shows 10 threads (RTF=0.896) is 42% faster than
            4 threads (RTF=1.274).  Default: 10.  Override only after new measurements.
    """
    if not hf_token or not str(hf_token).strip():
        raise ValueError("A valid Hugging Face token is required for PyAnnote speaker diarization.")

    print(f"Loading PyAnnote Diarization pipeline ({model_name})...")
    try:
        pipeline = Pipeline.from_pretrained(
            model_name,
            token=str(hf_token).strip(),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load PyAnnote. Error: {e}")

    if device is None:
        device = torch.device("cpu")
    pipeline.to(device)

    # Store the desired inference thread count on the pipeline object so
    # diarize_audio() can apply it immediately before inference regardless
    # of which modules were imported before it.
    pipeline._evidencemeet_num_threads = num_inference_threads

    return pipeline


def diarize_audio(
    audio_path: str,
    hf_token: Optional[str] = None,
    pipeline: Optional[Any] = None,
    return_timings: bool = False,
    model_load_time: Optional[float] = None,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, float]]]:
    """
    Runs PyAnnote speaker diarization on the provided audio file.
    Optimized to fallback to CPU gracefully and bypass compressed audio bugs.

    Args:
        audio_path: Path to the audio file.
        hf_token: Hugging Face authentication token (optional if pipeline is provided).
        pipeline: Optional pre-loaded / cached PyAnnote pipeline instance.
        return_timings: If True, returns (speaker_intervals, timings_dict).
                        If False (default), returns speaker_intervals for backward compatibility.
        model_load_time: Optional duration (in seconds) of pre-loaded model initialization.

    Returns:
        speaker_intervals: List of dicts with 'speaker', 'start', 'end'.
        If return_timings is True:
            Tuple of (speaker_intervals, {"conversion": float, "model_load": float, "inference": float, "total": float})
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Stage 1: Audio conversion to standard 16kHz Mono WAV file
    t0_conv = time.perf_counter()
    wav_path = audio_path.rsplit('.', 1)[0] + ("_converted.wav" if audio_path.lower().endswith(".wav") else ".wav")
    print(f"Converting {os.path.basename(audio_path)} to uncompressed WAV format...")
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path, 
        "-ac", "1", "-ar", "16000", 
        "-loglevel", "error", wav_path
    ], check=True)
    t_conversion = time.perf_counter() - t0_conv

    # Stage 2: Pipeline initialization (reusing pre-loaded/cached pipeline if provided)
    if pipeline is not None:
        t_model_load = float(model_load_time) if model_load_time is not None else 0.0
    else:
        if not hf_token or not str(hf_token).strip():
            raise ValueError("Either a valid hf_token or a pre-loaded pipeline must be provided.")
        t0_load = time.perf_counter()
        pipeline = load_diarization_pipeline(hf_token)
        t_model_load = time.perf_counter() - t0_load

    # Stage 3: Diarization inference
    print(f"Running diarization on {os.path.basename(wav_path)}...")
    print("NOTE: PyAnnote on CPU is intensive. This may take 1.5x - 2.5x the audio length.")

    # Apply the optimal thread count for diarization inference.
    # Benchmark on i5-1235U (2026-10-11): 10 threads → RTF=0.896; 4 threads → RTF=1.274.
    # We read from the pipeline attribute (set by load_diarization_pipeline) so the
    # count is config-driven.  We restore the previous count after inference.
    # Use int() conversion with a fallback: MagicMock (used in tests) would otherwise
    # auto-create the attribute and crash torch.set_num_threads with a non-int value.
    try:
        _infer_threads = int(pipeline._evidencemeet_num_threads)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        _infer_threads = 10  # Benchmark-proven default for i5-1235U
    _prev_threads = torch.get_num_threads()
    torch.set_num_threads(_infer_threads)

    t0_infer = time.perf_counter()
    # Run the model on the clean WAV file
    output = pipeline(wav_path)
    t_inference = time.perf_counter() - t0_infer

    # Restore caller's thread count after inference.
    torch.set_num_threads(_prev_threads)
    
    # PyAnnote 4.x returns a wrapper; we need to extract the actual speaker diarization object
    if hasattr(output, "speaker_diarization"):
        diarization = output.speaker_diarization
    else:
        diarization = output

    # Extract results into a clean, structured list of dictionaries
    speaker_intervals = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_intervals.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })

    unique_speakers_count = len(set([s['speaker'] for s in speaker_intervals]))
    t_total = t_conversion + t_model_load + t_inference
    print(f"Diarization complete! Found {unique_speakers_count} unique speakers in {t_total:.2f}s "
          f"(conv: {t_conversion:.2f}s, load: {t_model_load:.2f}s, infer: {t_inference:.2f}s).")

    timings = {
        "conversion": t_conversion,
        "model_load": t_model_load,
        "inference": t_inference,
        "total": t_total,
    }

    if return_timings:
        return speaker_intervals, timings
    return speaker_intervals


# --- Integration Test ---
if __name__ == "__main__":
    # 1. PASTE YOUR ACTUAL HUGGING FACE TOKEN HERE
    HF_TOKEN = "your_token_here" 
    
    # 2. Dynamic path resolution to data/samples/sample2.mp3
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    test_audio = os.path.join(PROJECT_ROOT, "data", "samples", "sample2.mp3")
    
    print(f"Testing diarization with audio: {test_audio}")
    
    # 3. Run diarization
    intervals = diarize_audio(test_audio, HF_TOKEN)
    
    # 4. Print the first 5 speaker turns to verify
    print(f"\n--- All Speaker Turns ---")
    for interval in intervals:
        print(f"[{interval['start']:05.2f}s -> {interval['end']:05.2f}s] {interval['speaker']}")