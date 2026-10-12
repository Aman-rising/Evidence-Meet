"""
Reproducible Diarization Benchmark for CPU-only Execution (Intel Core i5-1235U).

Profiles and compares:
1. Baseline: pyannote/speaker-diarization-3.1 (default threads, unoptimized)
2. Optimized: pyannote/speaker-diarization-3.1 (P-core 4 threads)
3. Alternative: pyannote/speaker-diarization-community-1 (4 threads)

Measures:
- Model initialization/load time (sec)
- Audio prep/conversion time (sec)
- Diarization inference latency (sec)
- Real-Time Factor (RTF = inference_time / audio_duration)
- Speaker assignments & detection quality
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from typing import Dict, Any, List

# Setup project root and ffmpeg path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_FFMPEG_BIN = r"C:\Users\Hp\Downloads\ffmpeg-9.0.2-full_build-shared\ffmpeg-9.0.2-full_build-shared\bin"
if os.path.exists(DEFAULT_FFMPEG_BIN) and DEFAULT_FFMPEG_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = DEFAULT_FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

import torch
from member1_speech.diarization.diarizer import load_diarization_pipeline, diarize_audio


def get_audio_duration(audio_path: str) -> float:
    """Uses ffprobe to obtain audio duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def run_benchmark_config(
    config_name: str,
    model_name: str,
    threads: int,
    audio_path: str,
    hf_token: str,
    batch_size: int = 32,
) -> Dict[str, Any]:
    print(f"\n==================================================")
    print(f"Running Config: {config_name}")
    print(f"Model: {model_name} | Threads: {threads} | Batch Size: {batch_size}")
    print(f"==================================================")

    # Set PyTorch thread count
    torch.set_num_threads(threads)

    # Measure model startup time
    t0_load = time.perf_counter()
    pipeline = load_diarization_pipeline(hf_token, model_name=model_name)
    model_load_time = time.perf_counter() - t0_load
    print(f"Model initialized in {model_load_time:.2f}s")

    # If pipeline exposes batch_size attributes, ensure batch size is set
    if hasattr(pipeline, "_segmentation") and hasattr(pipeline._segmentation, "batch_size"):
        pipeline._segmentation.batch_size = batch_size
    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = batch_size

    # Measure diarization execution
    intervals, timings = diarize_audio(
        audio_path,
        hf_token=hf_token,
        pipeline=pipeline,
        model_load_time=model_load_time,
        return_timings=True,
    )

    audio_dur = get_audio_duration(audio_path)
    rtf = timings["inference"] / audio_dur if audio_dur > 0 else 0.0

    unique_speakers = sorted(list(set(s["speaker"] for s in intervals)))
    
    # Calculate speaker speaking time breakdown
    speaker_durations: Dict[str, float] = {}
    for item in intervals:
        spk = item["speaker"]
        dur = item["end"] - item["start"]
        speaker_durations[spk] = round(speaker_durations.get(spk, 0.0) + dur, 2)

    result = {
        "config_name": config_name,
        "model_name": model_name,
        "threads": threads,
        "batch_size": batch_size,
        "audio_duration_sec": round(audio_dur, 2),
        "model_load_sec": round(model_load_time, 2),
        "conversion_sec": round(timings["conversion"], 2),
        "inference_sec": round(timings["inference"], 2),
        "total_time_sec": round(timings["total"], 2),
        "rtf": round(rtf, 4),
        "num_speakers": len(unique_speakers),
        "unique_speakers": unique_speakers,
        "speaker_durations_sec": speaker_durations,
        "sample_intervals": intervals[:8],
    }

    print(f"Done: {config_name}")
    print(f"  Inference Time: {result['inference_sec']}s (RTF: {result['rtf']})")
    print(f"  Total Time:     {result['total_time_sec']}s")
    print(f"  Speakers ({result['num_speakers']}): {unique_speakers}")
    print(f"  Durations: {speaker_durations}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Profile EvidenceMeet Diarization on CPU")
    parser.add_argument(
        "--audio",
        default=os.path.join(PROJECT_ROOT, "data", "samples", "Business English Conversations ESL Business Meeting Conversation.mp3"),
        help="Path to meeting audio"
    )
    parser.add_argument(
        "--token",
        default=os.getenv("hf_token"),
        help="Hugging Face token"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration in seconds to slice for rapid profiling (e.g. 60.0)"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "experiments", "diarization_benchmark_results.json"),
        help="Output JSON path for benchmark report"
    )
    args = parser.parse_args()

    audio_target = args.audio
    temp_sliced = None

    if args.duration is not None:
        temp_sliced = os.path.join(PROJECT_ROOT, "data", "samples", f"temp_bench_{int(args.duration)}s.wav")
        print(f"Creating {args.duration}s sliced audio for rapid benchmark: {temp_sliced}")
        subprocess.run([
            "ffmpeg", "-y", "-i", args.audio,
            "-t", str(args.duration),
            "-ac", "1", "-ar", "16000",
            "-loglevel", "error", temp_sliced
        ], check=True)
        audio_target = temp_sliced

    try:
        results = []

        # 1. Baseline: pyannote/speaker-diarization-3.1 with default 10 threads
        res_baseline = run_benchmark_config(
            config_name="baseline_sd31_10threads",
            model_name="pyannote/speaker-diarization-3.1",
            threads=10,
            audio_path=audio_target,
            hf_token=args.token,
            batch_size=32,
        )
        results.append(res_baseline)

        # 2. Optimized: pyannote/speaker-diarization-3.1 with 4 P-core threads
        res_opt4 = run_benchmark_config(
            config_name="optimized_sd31_4threads",
            model_name="pyannote/speaker-diarization-3.1",
            threads=4,
            audio_path=audio_target,
            hf_token=args.token,
            batch_size=32,
        )
        results.append(res_opt4)

        # 3. Alternative: pyannote/speaker-diarization-community-1 with 4 P-core threads
        res_comm1 = run_benchmark_config(
            config_name="alternative_community1_4threads",
            model_name="pyannote/speaker-diarization-community-1",
            threads=4,
            audio_path=audio_target,
            hf_token=args.token,
            batch_size=32,
        )
        results.append(res_comm1)

        # Summary comparison table
        summary = {
            "target_audio": os.path.basename(args.audio),
            "benchmark_duration_sec": get_audio_duration(audio_target),
            "configurations": results,
            "comparisons": {}
        }

        base_infer = res_baseline["inference_sec"]
        for res in results:
            speedup = round(base_infer / res["inference_sec"], 2) if res["inference_sec"] > 0 else 1.0
            summary["comparisons"][res["config_name"]] = {
                "inference_speedup_vs_baseline": f"{speedup}x",
                "inference_reduction_percent": f"{round((1.0 - res['inference_sec'] / base_infer) * 100, 1)}%",
                "num_speakers": res["num_speakers"],
            }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n==================================================")
        print("BENCHMARK SUMMARY")
        print("==================================================")
        print(json.dumps(summary["comparisons"], indent=2))
        print(f"\nSaved detailed benchmark report to: {args.output}")

    finally:
        if temp_sliced and os.path.exists(temp_sliced):
            os.remove(temp_sliced)
        for sfx in ["_converted.wav"]:
            cand = (temp_sliced or audio_target).rsplit(".", 1)[0] + sfx
            if os.path.exists(cand):
                os.remove(cand)


if __name__ == "__main__":
    main()
