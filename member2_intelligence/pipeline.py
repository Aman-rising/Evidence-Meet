"""
Member 2 Intelligence Pipeline

Orchestrates the Member 2 processing stages:
  Stage 1: Build structured transcript from Member 1 alignment data
  Stage 2: Per-segment preprocessing (preprocess_transcript)
  Stage 3: Extract concatenated cleaned text
  Stage 4: Baseline BART summarization (generate_summary)
  Stage 5: Qwen intelligence extraction + evidence retrieval + verification

This module bridges Member 1's structured transcript output and
Member 2's intelligence components. It constructs the contract-compliant
structured transcript from Member 1's in-memory outputs, then runs
each processing stage in order.

The structured transcript is the official input boundary for Member 2.
All intelligence stages receive and preserve the full segment structure
(id, speaker, start, end, raw_text, cleaned_text, text, overlap).
"""

import time
import traceback
from typing import Dict, List, Any, Optional, Callable

from member2_intelligence.preprocessing.preprocessor import preprocess_transcript
from member2_intelligence.llm.summarizer import generate_summary

from member2_intelligence.intelligence.schemas import (
    MeetingIntelligence,
    Claim,
    Decision,
    ActionItem,
    Topic,
    Contradiction,
    Evidence,
)

from member2_intelligence.intelligence.extractor import (
    extract_intelligence,
    ExtractionError,
)

from member2_intelligence.intelligence.evidence_retriever import (
    retrieve_evidence,
)

from member2_intelligence.intelligence.evidence_verifier import (
    verify_claim,
)


def format_duration(seconds: Optional[float]) -> str:
    """
    Formats a duration in seconds into a human-readable string.

    Displays milliseconds for durations under 1 second (e.g. '0.44 ms')
    and seconds for durations of 1 second or more (e.g. '21.72 s').
    """
    if seconds is None:
        return "0.00 ms"
    if seconds < 0:
        seconds = 0.0
    if seconds < 1.0:
        return f"{seconds * 1000.0:.2f} ms"
    return f"{seconds:.2f} s"


def build_structured_transcript(
    aligned_transcript: List[Dict[str, Any]],
    language: str,
    audio_filename: str,
) -> Dict[str, Any]:
    """
    Builds a contract-compliant structured transcript dictionary from
    Member 1's in-memory alignment output.

    This replicates the same JSON structure that member1_speech/main.py
    writes to disk, so that Member 2 always receives a consistent format
    regardless of whether the pipeline runs via CLI or Streamlit.

    Args:
        aligned_transcript: The merged speaker blocks from align_and_merge().
            Each block has keys: 'speaker', 'start', 'end', 'text'.

        language: ISO 639-1 language code detected by Whisper (e.g. 'en').

        audio_filename: Original audio filename (used to derive meeting_id).

    Returns:
        A structured transcript dict matching the Member 1 contract:
        {
            meeting_id,
            language,
            segments[
                {
                    id,
                    speaker,
                    start,
                    end,
                    raw_text,
                    cleaned_text,
                    text,
                    overlap
                }
            ]
        }
    """

    structured = {
        "meeting_id": (
            audio_filename.split(".")[0]
            if "." in audio_filename
            else audio_filename
        ),
        "language": language,
        "segments": [],
    }

    for idx, block in enumerate(aligned_transcript, start=1):
        structured["segments"].append(
            {
                "id": idx,
                "speaker": block["speaker"],
                "start": block["start"],
                "end": block["end"],
                "raw_text": block["text"],
                "cleaned_text": "",
                "text": block["text"],
                "overlap": False,
            }
        )

    return structured


def _get_query_text(item: Any) -> str:
    """
    Extracts the natural-language text to use as a retrieval query
    from an intelligence item.

    Supported item types:
        Claim
        Decision
        ActionItem
        Topic
        Contradiction
    """

    if isinstance(item, Claim):
        return item.text

    elif isinstance(item, Decision):
        return item.text

    elif isinstance(item, ActionItem):
        return item.task

    elif isinstance(item, Topic):
        return item.name

    elif isinstance(item, Contradiction):
        return item.description

    return ""


def attach_evidence(
    intelligence: MeetingIntelligence,
    transcript: Dict[str, Any],
    top_k: int = 3,
) -> MeetingIntelligence:
    """
    Attaches verified evidence from the transcript to each intelligence item
    and sets its evidence support score and verification status.

    For each item:
        1. Extract its text as a retrieval query.
        2. Retrieve candidate transcript evidence segments.
        3. Structurally verify each candidate against the transcript.
        4. Measure lexical overlap support score for each verified segment.
        5. Assign:
           - support_score: Best lexical overlap score [0.0, 1.0] across verified candidates.
             (Lexical grounding strength — NOT a probability of factual truth).
           - verification_status:
               * score >= THRESHOLD_SUPPORTED (0.50): "SUPPORTED"
               * score >= THRESHOLD_PARTIAL   (0.20): "PARTIALLY_SUPPORTED"
               * score <  THRESHOLD_PARTIAL   (0.20): "UNSUPPORTED"
           - evidence: Candidates meeting "SUPPORTED" threshold.
           - partial_evidence: Candidates meeting "PARTIALLY_SUPPORTED" threshold
             (only populated when no candidates meet "SUPPORTED").

    Unsupported and partially supported items honestly retain empty evidence lists
    for verified status, preventing false grounding.

    Args:
        intelligence:
            MeetingIntelligence produced by extraction.

        transcript:
            Processed structured transcript.

        top_k:
            Maximum number of evidence candidates to retrieve per item.

    Returns:
        The same MeetingIntelligence object with scores, statuses, and verified
        evidence attached.
    """

    all_items: List[Any] = (
        intelligence.claims
        + intelligence.decisions
        + intelligence.action_items
        + intelligence.topics
        + intelligence.contradictions
    )

    for item in all_items:
        query = _get_query_text(item)

        if not query:
            item.evidence = []
            item.partial_evidence = []
            item.support_score = 0.0
            item.verification_status = "UNSUPPORTED"
            continue

        # ---------------------------------------------------------------
        # Step 1: Retrieve candidate evidence
        # ---------------------------------------------------------------
        candidates = retrieve_evidence(
            transcript,
            query,
            top_k=top_k,
        )

        # If nothing was retrieved, leave evidence empty and score 0.0.
        if not candidates:
            item.evidence = []
            item.partial_evidence = []
            item.support_score = 0.0
            item.verification_status = "UNSUPPORTED"
            continue

        # ---------------------------------------------------------------
        # Step 2: Verify each candidate individually against the query.
        #
        # Reuses verify_claim() which checks:
        #   1. Structural verification (valid segment, speaker, timestamps, raw_text)
        #   2. Lexical support overlap score
        # ---------------------------------------------------------------
        candidate_verifications = []
        for cand in candidates:
            verification_claim = Claim(
                id=f"verification_{item.id}_{cand.segment_id}",
                text=query,
                evidence=[cand],
            )
            verification = verify_claim(
                verification_claim,
                transcript,
            )
            if verification.verified_evidence:
                candidate_verifications.append((cand, verification))

        if not candidate_verifications:
            item.evidence = []
            item.partial_evidence = []
            item.support_score = 0.0
            item.verification_status = "UNSUPPORTED"
            continue

        # Best lexical overlap score across structurally verified candidates
        best_score = max((v.score for _, v in candidate_verifications), default=0.0)
        item.support_score = round(float(best_score), 4)

        # Partition candidates into supported and partial
        supported_candidates = [
            cand for cand, v in candidate_verifications if v.status == "SUPPORTED"
        ]
        partial_candidates = [
            cand for cand, v in candidate_verifications if v.status == "PARTIALLY_SUPPORTED"
        ]

        if supported_candidates:
            item.verification_status = "SUPPORTED"
            item.evidence = supported_candidates
            item.partial_evidence = []
        elif partial_candidates:
            item.verification_status = "PARTIALLY_SUPPORTED"
            item.evidence = []  # Conservative: weak evidence is not claimed as verified
            item.partial_evidence = partial_candidates
        else:
            item.verification_status = "UNSUPPORTED"
            item.evidence = []
            item.partial_evidence = []

    return intelligence


def extract_verified_intelligence(
    transcript: Dict[str, Any],
    generate_fn: Optional[Callable[..., str]] = None,
    top_k: int = 3,
    stage_callback: Optional[Callable[[str, float], None]] = None,
) -> MeetingIntelligence:
    """
    Connects the existing Qwen extractor, evidence retriever, and evidence
    verifier into one verified-intelligence backend stage:

        processed structured transcript
                    ↓
              Qwen extractor
                    ↓
      Claims / Decisions / ActionItems / Topics / Contradictions
                    ↓
        evidence_retriever.retrieve_evidence()
                    ↓
        evidence_verifier.verify_claim()
                    ↓
      verified evidence attached to each intelligence item

    Rules enforced:
        - Never lets Qwen invent evidence.
        - Retrieves evidence only from the actual transcript segments.
        - Attaches only evidence that passes verification (SUPPORTED status).
        - Preserves the original raw_text as evidence.quote.
        - Keeps unsupported claims/items identifiable with empty evidence lists
          rather than silently deleting them.
        - Handles empty/no-match cases safely.
        - Deterministic execution given the Qwen extraction output.

    Args:
        transcript:
            Processed structured transcript dictionary with 'segments'.
        generate_fn:
            Optional callable for LLM generation. Defaults to Qwen via Ollama.
        top_k:
            Maximum candidate evidence segments to retrieve per item.
        stage_callback:
            Optional callback invoked after sub-stages complete:
            callback(stage_name, duration_seconds).

    Returns:
        MeetingIntelligence with verified evidence attached to each item.
    """
    if not transcript or not transcript.get("segments"):
        return MeetingIntelligence()

    # Step 1: Extract items with empty evidence via Qwen extractor
    t_start_qwen = time.perf_counter()
    intelligence = extract_intelligence(transcript, generate_fn=generate_fn)
    t_qwen = time.perf_counter() - t_start_qwen
    if stage_callback:
        stage_callback("qwen_extraction", t_qwen)

    # Step 2 & 3: Retrieve candidates, verify, and attach only verified evidence
    t_start_evidence = time.perf_counter()
    attach_evidence(intelligence, transcript, top_k=top_k)
    t_evidence = time.perf_counter() - t_start_evidence
    if stage_callback:
        stage_callback("evidence", t_evidence)

    return intelligence


def run_intelligence_pipeline(
    aligned_transcript: List[Dict[str, Any]],
    language: str,
    audio_filename: str,
    generate_fn: Optional[Callable[..., str]] = None,
    member1_timing: Optional[Dict[str, float]] = None,
    stage_callback: Optional[Callable[[str, float], None]] = None,
    summarize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """
    Runs the full Member 2 intelligence pipeline on Member 1's output.

    Stages:
        1. Build structured transcript from in-memory alignment data
        2. Per-segment preprocessing
        3. Extract concatenated cleaned text
        4. Baseline BART summarization
        5. Verified intelligence extraction (extract_verified_intelligence):
           - Qwen intelligence extraction
           - Evidence retrieval
           - Evidence verification and attachment

    Args:
        aligned_transcript:
            The merged speaker blocks from align_and_merge().

        language:
            ISO 639-1 language code from Whisper.

        audio_filename:
            Original audio filename.

        generate_fn:
            Optional callable for LLM generation.
            Defaults to Qwen via Ollama.
            Useful for testing with a mock.

        member1_timing:
            Optional dictionary with timing from Member 1 stages:
            {"transcription": float, "diarization": float, "alignment": float}.

        stage_callback:
            Optional callback invoked after each stage completes:
            callback(stage_name, duration_seconds).

        summarize_fn:
            Optional callable for text summarization.
            Defaults to baseline BART generate_summary.
            Useful for deterministic/fast testing.

    Returns:
        A dictionary containing:

        transcript:
            Fully processed structured transcript.

        cleaned_text:
            Concatenated cleaned text.

        summary:
            Baseline BART summary (preserved unchanged).

        verified_intelligence:
            MeetingIntelligence with verified evidence attached to each item.

        intelligence:
            Alias to verified_intelligence for backwards compatibility.

        intelligence_error:
            Error message if Qwen extraction fails,
            otherwise None.

        timing:
            Dictionary containing elapsed durations (in seconds) for each stage.
    """

    # ===============================================================
    # Stage 1: Build structured transcript
    # ===============================================================

    structured = build_structured_transcript(
        aligned_transcript,
        language,
        audio_filename,
    )

    # ===============================================================
    # Stage 2: Per-segment preprocessing & Stage 3: Cleaned text
    # ===============================================================

    t_start_preproc = time.perf_counter()
    processed = preprocess_transcript(structured)

    cleaned_text = " ".join(
        seg["cleaned_text"]
        for seg in processed["segments"]
        if seg["cleaned_text"]
    )
    t_preprocessing = time.perf_counter() - t_start_preproc

    if stage_callback:
        stage_callback("preprocessing", t_preprocessing)

    # ===============================================================
    # Stage 4: Baseline BART summarization
    #
    # IMPORTANT:
    # BART remains unchanged and is preserved as our baseline.
    # ===============================================================

    t_start_bart = time.perf_counter()
    _summarizer = summarize_fn if summarize_fn is not None else generate_summary
    summary_text = (
        _summarizer(cleaned_text)
        if cleaned_text
        else ""
    )
    t_bart = time.perf_counter() - t_start_bart

    if stage_callback:
        stage_callback("bart_summary", t_bart)

    # ===============================================================
    # Stage 5: Verified Intelligence Extraction
    #
    # Uses the clean extract_verified_intelligence() stage to run
    # Qwen extraction, evidence retrieval, and evidence verification.
    # ===============================================================

    intelligence = MeetingIntelligence()
    intelligence_error = None
    t_qwen = 0.0
    t_evidence = 0.0

    if processed.get("segments"):
        try:
            def _intel_callback(stage_name: str, duration: float):
                nonlocal t_qwen, t_evidence
                if stage_name == "qwen_extraction":
                    t_qwen = duration
                elif stage_name == "evidence":
                    t_evidence = duration
                if stage_callback:
                    stage_callback(stage_name, duration)

            intelligence = extract_verified_intelligence(
                processed,
                generate_fn=generate_fn,
                stage_callback=_intel_callback,
            )

        except (ExtractionError, Exception) as e:
            intelligence_error = (
                f"Intelligence extraction failed: {e}"
            )
            intelligence = MeetingIntelligence()

    # ===============================================================
    # Stage Timing Compilation
    # ===============================================================

    timing: Dict[str, float] = {}

    if member1_timing:
        for k in ("transcription", "diarization", "alignment"):
            if k in member1_timing:
                timing[k] = float(member1_timing[k])

    timing["preprocessing"] = float(t_preprocessing)
    timing["bart_summary"] = float(t_bart)
    timing["qwen_extraction"] = float(t_qwen)
    timing["evidence"] = float(t_evidence)
    timing["total"] = float(sum(timing.values()))

    # ===============================================================
    # Final pipeline result
    # ===============================================================

    return {
        "transcript": processed,
        "cleaned_text": cleaned_text,
        "summary": summary_text,
        "verified_intelligence": intelligence,
        "intelligence": intelligence,
        "intelligence_error": intelligence_error,
        "timing": timing,
    }