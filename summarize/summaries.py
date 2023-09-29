import datetime
import typing as t
from dataclasses import dataclass

from cdp_backend.database import models as cdp_models

from .connection import CDPConnection
from .queries import ExpandedEvent, ExpandedMatter, ExpandedSession, ExpandedTranscript

# ------------------------------------------------------------
# Summary "Models"
# ------------------------------------------------------------

# Each summary has the original model's key, and contains (at miniumum) a
# headline (short-sentence summary) and a detail (longer summary). The original
# summarized text is also included, mostly for inspection/debugging purposes.
# Where appropriate, summaries also contain links to the original source
# content (PDFs, transcript JSONs, etc).


@dataclass(frozen=True)
class SummaryBase:
    key: str
    headline: str
    detail: str
    text: str

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "headline": self.headline,
            "detail": self.detail,
            "text": self.text,
        }


@dataclass(frozen=True)
class MatterFileSummary(SummaryBase):
    uri: str

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "uri": self.uri,
        }


@dataclass(frozen=True)
class MatterSummary(SummaryBase):
    matter_file_summaries: tuple[MatterFileSummary, ...]

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "matter_file_summaries": [
                matter_file.to_dict() for matter_file in self.matter_file_summaries
            ],
        }


@dataclass(frozen=True)
class TranscriptSummary(SummaryBase):
    uri: str

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "uri": self.uri,
        }


@dataclass(frozen=True)
class SessionSummary(SummaryBase):
    transcript_summary: TranscriptSummary | None

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "transcript_summary": self.transcript_summary.to_dict()
            if self.transcript_summary
            else None,
        }


@dataclass(frozen=True)
class EventSummary(SummaryBase):
    dt: datetime.datetime
    matter_summaries: tuple[MatterSummary, ...]
    session_summaries: tuple[SessionSummary, ...]

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "dt": self.dt.isoformat(),
            "matter_summaries": [matter.to_dict() for matter in self.matter_summaries],
            "session_summaries": [
                session.to_dict() for session in self.session_summaries
            ],
        }


# ------------------------------------------------------------
# Summarization Methods
# ------------------------------------------------------------


def summarize_matter_file(
    connection: CDPConnection,
    expanded_matter: ExpandedMatter,
    matter_file: cdp_models.MatterFile,
) -> MatterFileSummary:
    content_type, bytes = expanded_matter.get_file(connection, matter_file)
    return MatterFileSummary(
        key=matter_file.key,
        headline="TODO",
        detail="TODO",
        text="TODO",
        uri=t.cast(str, matter_file.uri),
    )


def summarize_expanded_matter(
    connection: CDPConnection, expanded_matter: ExpandedMatter
) -> MatterSummary:
    matter_file_summaries = tuple(
        summarize_matter_file(connection, expanded_matter, matter_file)
        for matter_file in expanded_matter.files
    )
    return MatterSummary(
        key=expanded_matter.matter.key,
        headline="TODO",
        detail="TODO",
        text="TODO",
        matter_file_summaries=matter_file_summaries,
    )


def summarize_expanded_transcript(
    connection: CDPConnection,
    expanded_transcript: ExpandedTranscript,
) -> TranscriptSummary:
    expanded_transcript.get_data(connection)
    return TranscriptSummary(
        key=expanded_transcript.transcript.key,
        headline="TODO",
        detail="TODO",
        text="TODO",
        uri=t.cast(str, expanded_transcript.file.uri),
    )


def summarize_expanded_session(
    connection: CDPConnection, expanded_session: ExpandedSession
) -> SessionSummary:
    transcript = expanded_session.transcript
    transcript_summary = (
        summarize_expanded_transcript(connection, transcript) if transcript else None
    )
    return SessionSummary(
        key=expanded_session.session.key,
        headline="TODO",
        detail="TODO",
        text="TODO",
        transcript_summary=transcript_summary,
    )


def summarize_expanded_event(
    connection: CDPConnection, expanded_event: ExpandedEvent
) -> EventSummary:
    """Summarize an expanded event."""
    matter_summaries = tuple(
        summarize_expanded_matter(connection, matter)
        for matter in expanded_event.matters
    )
    session_summaries = tuple(
        summarize_expanded_session(connection, session)
        for session in expanded_event.sessions
    )
    return EventSummary(
        key=expanded_event.event.key,
        dt=t.cast(datetime.datetime, expanded_event.event.event_datetime),
        headline="TODO",
        detail="TODO",
        text="TODO",
        matter_summaries=matter_summaries,
        session_summaries=session_summaries,
    )
