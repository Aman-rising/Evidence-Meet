"""
Evidence-Based Meeting Intelligence Schemas

Defines the data structures for evidence-grounded meeting intelligence.
All intelligence outputs (claims, decisions, action items, topics,
contradictions) are linked back to the original transcript via Evidence
objects, ensuring full traceability to the source.

These schemas are data-only definitions. Extraction and verification
logic is NOT implemented here.

Uses Python standard-library dataclasses only — no external dependencies.
"""
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Evidence — the traceability link back to the original transcript
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """
    A reference to a specific segment in the original transcript that
    supports an intelligence item (claim, decision, action item, etc.).

    Attributes:
        segment_id: The 'id' of the transcript segment (from Member 1 output).
                    Must correspond to an actual segment in the transcript.
        speaker:    The speaker label from the transcript segment
                    (e.g. 'SPEAKER_00').
        start:      Segment start time in seconds from the beginning of
                    the audio.
        end:        Segment end time in seconds from the beginning of
                    the audio.
        quote:      The supporting text from the transcript segment.
                    Must be derived from the segment's raw_text — never
                    invented or paraphrased.
    """
    segment_id: int
    speaker: str
    start: float
    end: float
    quote: str

    def validate(self) -> None:
        """
        Validates that this Evidence object has all required fields
        with acceptable values.

        Raises:
            ValueError: If any field is invalid.
        """
        if not isinstance(self.segment_id, int) or self.segment_id < 1:
            raise ValueError(
                f"Evidence.segment_id must be a positive integer, "
                f"got: {self.segment_id!r}"
            )
        if not self.speaker or not isinstance(self.speaker, str):
            raise ValueError(
                f"Evidence.speaker must be a non-empty string, "
                f"got: {self.speaker!r}"
            )
        if not isinstance(self.start, (int, float)) or self.start < 0:
            raise ValueError(
                f"Evidence.start must be a non-negative number, "
                f"got: {self.start!r}"
            )
        if not isinstance(self.end, (int, float)) or self.end < self.start:
            raise ValueError(
                f"Evidence.end must be >= start ({self.start}), "
                f"got: {self.end!r}"
            )
        if not self.quote or not isinstance(self.quote, str):
            raise ValueError(
                f"Evidence.quote must be a non-empty string, "
                f"got: {self.quote!r}"
            )


# ---------------------------------------------------------------------------
# Intelligence item schemas
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """
    A factual claim, assertion, or opinion extracted from the meeting.

    Attributes:
        id:                  Unique string identifier for this claim.
        text:                The claim statement.
        evidence:            List of Evidence objects supporting this claim.
        support_score:       Evidence support score (lexical overlap grounding) in [0.0, 1.0].
        verification_status: Verification verdict ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED").
        confidence:          Optional model extraction confidence in [0.0, 1.0].
        partial_evidence:    List of Evidence objects with partial textual support.
    """
    id: str
    text: str
    evidence: List[Evidence] = field(default_factory=list)
    support_score: float = 0.0
    verification_status: str = "UNSUPPORTED"
    confidence: Optional[float] = None
    partial_evidence: List[Evidence] = field(default_factory=list)

    def validate(self) -> None:
        """Validates the claim and its evidence chain."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Claim.id must be a non-empty string, got: {self.id!r}")
        if not self.text or not isinstance(self.text, str):
            raise ValueError(f"Claim.text must be a non-empty string, got: {self.text!r}")
        if not self.evidence:
            raise ValueError(f"Claim '{self.id}' must have at least one Evidence entry.")
        for ev in self.evidence:
            ev.validate()
        for ev in self.partial_evidence:
            ev.validate()
        if not (0.0 <= self.support_score <= 1.0):
            raise ValueError(f"Claim.support_score must be in [0.0, 1.0], got {self.support_score}")
        if self.verification_status not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
            raise ValueError(f"Claim.verification_status must be SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED, got {self.verification_status}")


@dataclass
class Decision:
    """
    A decision made during the meeting.

    Attributes:
        id:                  Unique string identifier for this decision.
        text:                Description of the decision.
        evidence:            List of Evidence objects supporting this decision.
        status:              Decision classification ("confirmed" or "proposed").
        support_score:       Evidence support score (lexical overlap grounding) in [0.0, 1.0].
        verification_status: Verification verdict ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED").
        confidence:          Optional model extraction confidence in [0.0, 1.0].
        partial_evidence:    List of Evidence objects with partial textual support.
    """
    id: str
    text: str
    evidence: List[Evidence] = field(default_factory=list)
    status: str = "confirmed"
    support_score: float = 0.0
    verification_status: str = "UNSUPPORTED"
    confidence: Optional[float] = None
    partial_evidence: List[Evidence] = field(default_factory=list)

    def validate(self) -> None:
        """Validates the decision and its evidence chain."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Decision.id must be a non-empty string, got: {self.id!r}")
        if not self.text or not isinstance(self.text, str):
            raise ValueError(f"Decision.text must be a non-empty string, got: {self.text!r}")
        if not self.evidence:
            raise ValueError(f"Decision '{self.id}' must have at least one Evidence entry.")
        for ev in self.evidence:
            ev.validate()
        for ev in self.partial_evidence:
            ev.validate()
        if not (0.0 <= self.support_score <= 1.0):
            raise ValueError(f"Decision.support_score must be in [0.0, 1.0], got {self.support_score}")
        if self.verification_status not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
            raise ValueError(f"Decision.verification_status must be SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED, got {self.verification_status}")


@dataclass
class ActionItem:
    """
    A task or action item assigned during the meeting.

    Attributes:
        id:                  Unique string identifier for this action item.
        task:                Description of the task.
        owner:               Person responsible (speaker label or name). May be None
                             if the transcript does not establish ownership.
        deadline:            Deadline or timeframe mentioned. May be None if the
                             transcript does not establish a deadline.
        evidence:            List of Evidence objects supporting this action item.
        commitment_status:   Commitment classification ("committed", "requested", "proposed").
        support_score:       Evidence support score (lexical overlap grounding) in [0.0, 1.0].
        verification_status: Verification verdict ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED").
        confidence:          Optional model extraction confidence in [0.0, 1.0].
        partial_evidence:    List of Evidence objects with partial textual support.
    """
    id: str
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    commitment_status: str = "committed"
    support_score: float = 0.0
    verification_status: str = "UNSUPPORTED"
    confidence: Optional[float] = None
    partial_evidence: List[Evidence] = field(default_factory=list)

    def validate(self) -> None:
        """Validates the action item and its evidence chain."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"ActionItem.id must be a non-empty string, got: {self.id!r}")
        if not self.task or not isinstance(self.task, str):
            raise ValueError(f"ActionItem.task must be a non-empty string, got: {self.task!r}")
        if not self.evidence:
            raise ValueError(f"ActionItem '{self.id}' must have at least one Evidence entry.")
        for ev in self.evidence:
            ev.validate()
        for ev in self.partial_evidence:
            ev.validate()
        if not (0.0 <= self.support_score <= 1.0):
            raise ValueError(f"ActionItem.support_score must be in [0.0, 1.0], got {self.support_score}")
        if self.verification_status not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
            raise ValueError(f"ActionItem.verification_status must be SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED, got {self.verification_status}")


@dataclass
class Topic:
    """
    A topic or theme discussed during the meeting.

    Attributes:
        id:                  Unique string identifier for this topic.
        name:                Name or short description of the topic.
        evidence:            List of Evidence objects where this topic is discussed.
        support_score:       Evidence support score (lexical overlap grounding) in [0.0, 1.0].
        verification_status: Verification verdict ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED").
        confidence:          Optional model extraction confidence in [0.0, 1.0].
        partial_evidence:    List of Evidence objects with partial textual support.
    """
    id: str
    name: str
    evidence: List[Evidence] = field(default_factory=list)
    support_score: float = 0.0
    verification_status: str = "UNSUPPORTED"
    confidence: Optional[float] = None
    partial_evidence: List[Evidence] = field(default_factory=list)

    def validate(self) -> None:
        """Validates the topic and its evidence chain."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Topic.id must be a non-empty string, got: {self.id!r}")
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"Topic.name must be a non-empty string, got: {self.name!r}")
        if not self.evidence:
            raise ValueError(f"Topic '{self.id}' must have at least one Evidence entry.")
        for ev in self.evidence:
            ev.validate()
        for ev in self.partial_evidence:
            ev.validate()
        if not (0.0 <= self.support_score <= 1.0):
            raise ValueError(f"Topic.support_score must be in [0.0, 1.0], got {self.support_score}")
        if self.verification_status not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
            raise ValueError(f"Topic.verification_status must be SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED, got {self.verification_status}")


@dataclass
class Contradiction:
    """
    A contradiction or conflicting statement detected in the meeting.

    Attributes:
        id:                  Unique string identifier for this contradiction.
        description:         Description of the contradiction.
        evidence:            List of Evidence objects showing the conflicting statements.
        support_score:       Evidence support score (lexical overlap grounding) in [0.0, 1.0].
        verification_status: Verification verdict ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED").
        confidence:          Optional model extraction confidence in [0.0, 1.0].
        partial_evidence:    List of Evidence objects with partial textual support.
    """
    id: str
    description: str
    evidence: List[Evidence] = field(default_factory=list)
    support_score: float = 0.0
    verification_status: str = "UNSUPPORTED"
    confidence: Optional[float] = None
    partial_evidence: List[Evidence] = field(default_factory=list)

    def validate(self) -> None:
        """Validates the contradiction and its evidence chain."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError(
                f"Contradiction.id must be a non-empty string, got: {self.id!r}"
            )
        if not self.description or not isinstance(self.description, str):
            raise ValueError(
                f"Contradiction.description must be a non-empty string, "
                f"got: {self.description!r}"
            )
        if not self.evidence:
            raise ValueError(
                f"Contradiction '{self.id}' must have at least one Evidence entry."
            )
        for ev in self.evidence:
            ev.validate()
        for ev in self.partial_evidence:
            ev.validate()
        if not (0.0 <= self.support_score <= 1.0):
            raise ValueError(f"Contradiction.support_score must be in [0.0, 1.0], got {self.support_score}")
        if self.verification_status not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
            raise ValueError(f"Contradiction.verification_status must be SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED, got {self.verification_status}")


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass
class MeetingIntelligence:
    """
    The top-level container for all evidence-based intelligence extracted
    from a meeting transcript.

    Each list may be empty if no items of that type were found.

    Attributes:
        claims:         List of factual claims extracted from the meeting.
        decisions:      List of decisions made during the meeting.
        action_items:   List of tasks or action items assigned.
        topics:         List of topics or themes discussed.
        contradictions: List of contradictions detected.
    """
    claims: List[Claim] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    topics: List[Topic] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)

    def validate(self) -> None:
        """Validates all contained intelligence items and their evidence."""
        for claim in self.claims:
            claim.validate()
        for decision in self.decisions:
            decision.validate()
        for action_item in self.action_items:
            action_item.validate()
        for topic in self.topics:
            topic.validate()
        for contradiction in self.contradictions:
            contradiction.validate()
