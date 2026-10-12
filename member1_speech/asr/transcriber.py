import os
import torch
from faster_whisper import WhisperModel
import time

def transcribe_audio(audio_file_path):
    """
    Transcribes audio using the Faster-Whisper 'base' model.
    Optimized for CPU execution with VAD filtering and Word-Level Timestamps.

    Thread count is constrained locally to 4 for ASR (matching Whisper's optimal
    CPU parallelism on the i5-1235U P-cores).  The constraint is applied only
    for the duration of this call and restored afterward so that PyAnnote
    diarization — which runs best at ~10 threads on this chip — is not affected.
    """
    # --- CPU OPTIMIZATION FOR INTEL i5 (scoped to ASR only) ---
    # Whisper runs fastest with 4 threads on P-cores; more threads add contention.
    # We restore the previous value after transcription so PyAnnote is unaffected.
    _prev_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    # Set environment hints for BLAS/MKL libraries used by Faster-Whisper.
    # These env vars are read at library init, so setting them here is advisory
    # for any sub-process, but the torch thread control is the effective guard.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    model_size = "base"
    
    print("Loading Faster-Whisper model...")
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:
        print("Model not found locally. Downloading...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8", local_files_only=False)
    
    print(f"Starting transcription for {audio_file_path}...")
    
    # Run transcription with VAD and Word Timestamps
    segments_generator, info = model.transcribe(
        audio_file_path, 
        beam_size=5, 
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500), # Native VAD tuned
        word_timestamps=True # CRITICAL: Required for future speaker alignment
    )
    
    segments_list = []
    full_transcript = ""
    
    for segment in segments_generator:
        # We now store the segment AND its word-level breakdown
        segment_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in segment.words]
        }
        segments_list.append(segment_data)
        full_transcript += segment.text + " "
        
    print("Transcription complete.")
    # Restore previous PyTorch thread count so downstream stages (e.g. PyAnnote)
    # can run at their own optimal thread count.
    torch.set_num_threads(_prev_threads)
    return full_transcript.strip(), segments_list, info.language

# --- Simple Local Test ---
if __name__ == "__main__":
    # Point this to a real audio file on your machine to test
    test_audio = "meeting.m4a" 
    text, segments, lang = transcribe_audio(test_audio)
    print(segments[0]) # Should print the first segment with word-level timestamps
    pass