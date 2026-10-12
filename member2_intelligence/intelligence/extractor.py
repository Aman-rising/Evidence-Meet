"""
Meeting Intelligence Extractor

Uses a local Qwen3 model (via Ollama) to extract structured intelligence
from a processed meeting transcript. Converts the LLM's JSON output into
the existing schemas.py dataclasses.

IMPORTANT:
- Evidence lists are left EMPTY. The LLM must NOT invent segment IDs,
  timestamps, speakers, or quotes. Evidence is attached later by the
  deterministic evidence retriever + verifier.
- The extractor is conservative: it only extracts information that is
  explicitly stated or clearly agreed upon in the transcript.
- The existing BART summarizer is NOT modified or replaced.
"""
import json
from typing import Dict, List, Any, Optional, Callable

from member2_intelligence.intelligence.schemas import (
    Claim,
    Decision,
    ActionItem,
    Topic,
    Contradiction,
    MeetingIntelligence,
)
from member2_intelligence.llm.qwen_client import generate as qwen_generate


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise meeting analysis assistant. You extract structured \
information from meeting transcripts. You must be conservative:

- Extract ONLY information explicitly stated or clearly agreed upon.
- Do NOT invent facts or hallucinate details.
- Clearly distinguish requests, proposals, and suggestions from confirmed decisions and accepted action items.
- A request or suggestion must NEVER automatically become an accepted commitment.
- For action items, only classify items as accepted commitments if a speaker clearly committed to or accepted the task. If a task was only requested or suggested without acceptance, mark commitment_status as "requested" or "proposed".
- For decisions, only classify items as confirmed decisions if participants clearly agreed or finalized them. Tentative proposals or ideas must be distinguished with status "proposed".
- Preserve uncertainty when present (e.g., "possibly", "might").
- Identify apparent contradictions rather than choosing one side. Never silently resolve conflicting statements.
- Return null for unknown action-item owner or deadline. Never fabricate owners or deadlines.
- Return valid JSON only. No markdown, no commentary, no extra text.\
"""

_EXTRACTION_PROMPT_TEMPLATE = """\
Analyze the following meeting transcript and extract structured intelligence.

Extract ALL distinct claims, decisions, action items, topics, and contradictions present in the meeting. Do NOT limit any category to a single item. If multiple items of a category exist in the transcript, include all of them as separate entries in their respective array.

Return ONLY a JSON object with exactly these keys:
{{
  "claims": [
    {{"id": "c1", "text": "..."}},
    {{"id": "c2", "text": "..."}}
  ],
  "decisions": [
    {{"id": "d1", "text": "...", "status": "confirmed"}},
    {{"id": "d2", "text": "...", "status": "confirmed"}}
  ],
  "action_items": [
    {{"id": "a1", "task": "...", "owner": "..." or null, "deadline": "..." or null, "commitment_status": "committed"}},
    {{"id": "a2", "task": "...", "owner": "..." or null, "deadline": "..." or null, "commitment_status": "committed"}}
  ],
  "topics": [
    {{"id": "t1", "name": "..."}},
    {{"id": "t2", "name": "..."}}
  ],
  "contradictions": [
    {{"id": "x1", "description": "..."}},
    {{"id": "x2", "description": "..."}}
  ]
}}

Rules:
- Extract ALL distinct items present in the transcript. Never limit any category to one item.
- Use multiple array entries whenever multiple items of that category exist in the transcript.
- Keep an array empty ([]) if no items of that category exist in the transcript.
- Do NOT duplicate the same item.
- Remain conservative: extract ONLY information explicitly stated or clearly supported by the transcript. Do NOT invent, assume, or hallucinate facts.
- "claims" are factual statements, assertions, or key points made during the meeting.
- "decisions" are outcomes that were clearly agreed or decided. Set "status" to "confirmed" if clearly agreed/decided. If an item is only a tentative proposal, suggestion, or idea without confirmed consensus, set "status" to "proposed".
- "action_items" are tasks discussed. Crucially distinguish accepted commitments from mere requests:
  - If a person accepted or committed to the task, set "commitment_status" to "committed".
  - If a task was merely requested or suggested by someone without acceptance or commitment, set "commitment_status" to "requested" or "proposed". A request must NOT automatically become an accepted commitment.
  - Set "owner" to the speaker label (e.g. "SPEAKER_00") if clearly assigned and accepted, otherwise null. Never invent owners.
  - Set "deadline" to the mentioned timeframe if stated, otherwise null. Never invent deadlines.
- "topics" are the distinct main subjects or themes discussed.
- "contradictions" are conflicting statements made by different speakers or by the same speaker at different times. Do NOT resolve contradictions or choose one side; describe both conflicting perspectives.
- Optional: include "confidence" as a float in [0.0, 1.0] representing model extraction confidence (distinct from evidence support).
- Do NOT include evidence, segment IDs, timestamps, or quotes in your response.
- Return ONLY the JSON object. No markdown fences. No explanation.

TRANSCRIPT:
{transcript_text}\
"""


def _format_transcript_for_prompt(transcript: Dict[str, Any]) -> str:
    """
    Formats the structured transcript into a readable string for the LLM,
    including speaker labels and timestamps for context.
    """
    lines = []
    for seg in transcript.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        # Use cleaned_text if available, fall back to raw_text
        text = seg.get("cleaned_text") or seg.get("raw_text", "")
        lines.append(f"[{start:.1f}s-{end:.1f}s] {speaker}: {text}")
    return "\n".join(lines)


def _build_prompt(transcript: Dict[str, Any]) -> str:
    """Builds the full extraction prompt from a structured transcript."""
    transcript_text = _format_transcript_for_prompt(transcript)
    return _EXTRACTION_PROMPT_TEMPLATE.format(transcript_text=transcript_text)


# ---------------------------------------------------------------------------
# JSON parsing and schema conversion
# ---------------------------------------------------------------------------

class ExtractionError(Exception):
    """Raised when extraction fails (bad JSON, missing fields, etc.)."""
    pass


def _clean_json_response(raw: str) -> str:
    """
    Strips common LLM response artifacts to isolate JSON.
    Handles markdown fences, leading/trailing whitespace, and
    thinking tags that Qwen3 may emit.
    """
    text = raw.strip()

    # Strip <think>...</think> blocks that Qwen3 may produce
    while "<think>" in text and "</think>" in text:
        start = text.index("<think>")
        end = text.index("</think>") + len("</think>")
        text = text[:start] + text[end:]
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    return text


def _parse_llm_response(raw_response: str) -> Dict[str, Any]:
    """
    Parses the LLM's raw text response into a Python dict.

    Raises:
        ExtractionError: If the response is not valid JSON.
    """
    cleaned = _clean_json_response(raw_response)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ExtractionError(
            f"LLM returned invalid JSON: {e}\n"
            f"Cleaned response (first 500 chars): {cleaned[:500]}"
        ) from e

    if not isinstance(data, dict):
        raise ExtractionError(
            f"Expected a JSON object, got {type(data).__name__}: {str(data)[:200]}"
        )

    return data


def _parse_confidence(val: Any) -> Optional[float]:
    """Safely extracts a model confidence float in [0.0, 1.0]."""
    if val is None:
        return None
    try:
        f = float(val)
        if 0.0 <= f <= 1.0:
            return f
        if 0.0 <= f <= 100.0:
            return f / 100.0
    except (ValueError, TypeError):
        pass
    return None


def _parse_commitment_status(val: Any, task_text: str = "", owner: Optional[str] = None) -> str:
    """
    Distinguishes accepted commitments from mere requests or proposals.
    A request must NOT automatically become an accepted commitment.
    """
    if val is not None:
        v = str(val).strip().lower()
        if v in ("request", "requested", "ask", "asked"):
            return "requested"
        if v in ("proposal", "proposed", "suggestion", "suggested", "idea"):
            return "proposed"
        if v in ("committed", "accepted", "assigned", "agreed"):
            return "committed"

    lower_task = task_text.lower().strip()
    request_cues = (
        "can someone", "could someone", "can anyone", "could anyone",
        "please can", "request to", "request for", "requested that",
        "ask if", "asking to", "asking for", "someone should check",
    )
    if any(cue in lower_task for cue in request_cues):
        return "requested"

    proposal_cues = (
        "maybe we should", "perhaps we could", "suggest we", "suggestion to",
        "proposal to", "we might want to", "consider doing",
    )
    if any(cue in lower_task for cue in proposal_cues) and owner is None:
        return "proposed"

    return "committed"


def _parse_decision_status(val: Any, text: str = "") -> str:
    """
    Distinguishes confirmed decisions from tentative proposals/suggestions.
    """
    if val is not None:
        v = str(val).strip().lower()
        if v in ("proposed", "proposal", "tentative", "suggestion"):
            return "proposed"
        if v in ("confirmed", "decided", "agreed", "finalized"):
            return "confirmed"

    lower_text = text.lower().strip()
    proposal_cues = (
        "proposal to", "proposed to", "tentative decision",
        "maybe we should", "suggested to", "considering whether",
    )
    if any(cue in lower_text for cue in proposal_cues):
        return "proposed"

    return "confirmed"


def _convert_to_meeting_intelligence(data: Dict[str, Any]) -> MeetingIntelligence:
    """
    Converts the parsed LLM JSON into MeetingIntelligence dataclasses.

    Evidence lists are intentionally left EMPTY — the LLM must not
    fabricate evidence. Evidence is attached later by the deterministic
    retriever + verifier.
    """
    claims = []
    for item in data.get("claims", []):
        if isinstance(item, dict) and item.get("text"):
            claims.append(Claim(
                id=str(item.get("id", f"c{len(claims)+1}")),
                text=str(item["text"]),
                evidence=[],  # Intentionally empty — attached later
                confidence=_parse_confidence(item.get("confidence")),
            ))

    decisions = []
    for item in data.get("decisions", []):
        if isinstance(item, dict) and item.get("text"):
            decisions.append(Decision(
                id=str(item.get("id", f"d{len(decisions)+1}")),
                text=str(item["text"]),
                evidence=[],
                status=_parse_decision_status(item.get("status"), str(item["text"])),
                confidence=_parse_confidence(item.get("confidence")),
            ))

    action_items = []
    for item in data.get("action_items", []):
        if isinstance(item, dict) and item.get("task"):
            owner = item.get("owner")
            if owner is not None:
                owner = str(owner)
                if owner.lower() in ("null", "none", "unknown", ""):
                    owner = None
            deadline = item.get("deadline")
            if deadline is not None:
                deadline = str(deadline)
                if deadline.lower() in ("null", "none", "unknown", ""):
                    deadline = None
            task_str = str(item["task"])
            commitment_status = _parse_commitment_status(
                item.get("commitment_status") or item.get("status"),
                task_text=task_str,
                owner=owner,
            )
            action_items.append(ActionItem(
                id=str(item.get("id", f"a{len(action_items)+1}")),
                task=task_str,
                owner=owner,
                deadline=deadline,
                evidence=[],
                commitment_status=commitment_status,
                confidence=_parse_confidence(item.get("confidence")),
            ))

    topics = []
    for item in data.get("topics", []):
        if isinstance(item, dict) and item.get("name"):
            topics.append(Topic(
                id=str(item.get("id", f"t{len(topics)+1}")),
                name=str(item["name"]),
                evidence=[],
                confidence=_parse_confidence(item.get("confidence")),
            ))

    contradictions = []
    for item in data.get("contradictions", []):
        if isinstance(item, dict) and item.get("description"):
            contradictions.append(Contradiction(
                id=str(item.get("id", f"x{len(contradictions)+1}")),
                description=str(item["description"]),
                evidence=[],
                confidence=_parse_confidence(item.get("confidence")),
            ))

    return MeetingIntelligence(
        claims=claims,
        decisions=decisions,
        action_items=action_items,
        topics=topics,
        contradictions=contradictions,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_intelligence(
    transcript: Dict[str, Any],
    generate_fn: Optional[Callable[..., str]] = None,
) -> MeetingIntelligence:
    """
    Extracts structured meeting intelligence from a processed transcript
    using a local Qwen3 model via Ollama.

    The LLM identifies claims, decisions, action items, topics, and
    contradictions. Evidence lists are left EMPTY — evidence is attached
    later by the deterministic retriever + verifier pipeline.

    Args:
        transcript: The processed structured transcript dictionary
                    (after preprocess_transcript). Must contain 'segments'.
        generate_fn: Optional callable for LLM generation. Defaults to
                     qwen_generate (Ollama). Accepts (prompt, system) kwargs.
                     Useful for testing with a mock.

    Returns:
        A MeetingIntelligence dataclass with extracted items.
        Evidence lists are empty — to be populated by the retriever.

    Raises:
        ExtractionError: If the LLM response cannot be parsed.
    """
    segments = transcript.get("segments", [])

    # Handle empty transcript
    if not segments:
        return MeetingIntelligence()

    # Build prompt
    prompt = _build_prompt(transcript)

    # Call LLM
    if generate_fn is None:
        generate_fn = qwen_generate

    raw_response = generate_fn(prompt=prompt, system=_SYSTEM_PROMPT)

    # Parse and convert
    data = _parse_llm_response(raw_response)
    intelligence = _convert_to_meeting_intelligence(data)

    return intelligence
