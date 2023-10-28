# summaries.py -- higher level code that's CDP model aware and uses
# stuff in extract.py and llm.py to build the summaries.

import datetime
import io
import logging
import typing as t
from dataclasses import dataclass

from cdp_backend.database import models as cdp_models

from .connection import CDPConnection
from .extract import extract_text_from_bytes, extract_text_from_transcript_model
from .llm import LanguageModel, SummarizationResult
from .prompts import PromptTemplates
from .queries import ExpandedEvent, ExpandedMatter, ExpandedSession, ExpandedTranscript

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict:
        """Return a dictionary representation of this object."""
        return {
            "key": self.key,
            "headline": self.headline,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class MatterFileSummary(SummaryBase):
    uri: str

    def to_dict(self) -> dict:
        """Return a dictionary representation of this object."""
        return {
            **super().to_dict(),
            "uri": self.uri,
        }


@dataclass(frozen=True)
class MatterSummary(SummaryBase):
    matter_file_summaries: tuple[MatterFileSummary, ...]

    def to_dict(self) -> dict:
        """Return a dictionary representation of this object."""
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
        """Return a dictionary representation of this object."""
        return {
            **super().to_dict(),
            "uri": self.uri,
        }


@dataclass(frozen=True)
class SessionSummary(SummaryBase):
    transcript_summary: TranscriptSummary | None

    def to_dict(self) -> dict:
        """Return a dictionary representation of this object."""
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
        """Return a dictionary representation of this object."""
        return {
            **super().to_dict(),
            "dt": self.dt.isoformat(),
            "matter_summaries": [matter.to_dict() for matter in self.matter_summaries],
            "session_summaries": [
                session.to_dict() for session in self.session_summaries
            ],
        }


# ------------------------------------------------------------
# Model summarization prompts & methods
# ------------------------------------------------------------


class CDPSummarizer:
    """Top-level summarizer implementation for CDP data."""

    llm: LanguageModel
    prompts: PromptTemplates
    connection: CDPConnection

    def __init__(
        self, llm: LanguageModel, prompts: PromptTemplates, connection: CDPConnection
    ):
        self.llm = llm
        self.prompts = prompts
        self.connection = connection

    def summarize_concise(self, text: str) -> SummarizationResult:
        """Use the concise summary style to summarize text."""
        return self.llm.summarize(text, self.prompts.concise)

    def summarize_matter(self, title: str, text: str) -> SummarizationResult:
        """Use the matter summary style to summarize text."""
        return self.llm.summarize(text, self.prompts.matter, {"title": title})

    def summarize_transcript(self, text: str) -> SummarizationResult:
        """Use the transcript summary style to summarize text."""
        return self.llm.summarize(text, self.prompts.transcript)

    def summarize_event(
        self, body_name: str, dt_fmt: str, text: str
    ) -> SummarizationResult:
        """Use the event summary style to summarize text."""
        return self.llm.summarize(
            text, self.prompts.event, {"body_name": body_name, "dt_fmt": dt_fmt}
        )

    # ------------------------------------------------------------
    # Top-Level Model Summarization Methods
    # ------------------------------------------------------------

    def summarize_matter_file(
        self,
        expanded_matter: ExpandedMatter,
        matter_file: cdp_models.MatterFile,
    ) -> MatterFileSummary:
        """
        Summarize an arbitrary matter file.

        On failure, return a summary with warning text (allowing further
        summarization to continue).
        """
        try:
            logger.info(
                "matter %s file %s :: downloading ",
                expanded_matter.matter.key,
                matter_file.key,
            )
            content_type, data = expanded_matter.get_file(self.connection, matter_file)
            logger.info(
                "matter %s file %s :: extracting",
                expanded_matter.matter.key,
                matter_file.key,
            )
            text = extract_text_from_bytes(io.BytesIO(data), content_type)
            logger.info(
                "matter %s file %s :: summarizing",
                expanded_matter.matter.key,
                matter_file.key,
            )
            result = self.summarize_concise(text)
        except Exception as e:
            logger.exception("Failed to summarize matter file %s", matter_file.key)
            return MatterFileSummary(
                key=matter_file.key,
                headline="Failed to summarize",
                detail=str(e),
                uri=t.cast(str, matter_file.uri),
            )
        else:
            return MatterFileSummary(
                key=matter_file.key,
                headline=result.headline,
                detail=result.detail,
                uri=t.cast(str, matter_file.uri),
            )

    def summarize_expanded_matter(
        self, expanded_matter: ExpandedMatter
    ) -> MatterSummary:
        """
        Summarize a legislative matter by summarizing all of its files and
        combining the results.

        On failure, return a summary with warning text (allowing further
        summarization to continue).
        """
        matter_file_summaries = tuple(
            self.summarize_matter_file(expanded_matter, matter_file)
            for matter_file in expanded_matter.files
        )
        text = "\n\n".join(
            matter_file_summary.detail for matter_file_summary in matter_file_summaries
        )
        try:
            logger.info("matter %s :: summarizing", expanded_matter.matter.key)
            result = self.summarize_matter(
                t.cast(str, expanded_matter.matter.title), text
            )
        except Exception as e:
            logger.exception(
                "Failed to summarize matter %s", expanded_matter.matter.key
            )
            return MatterSummary(
                key=expanded_matter.matter.key,
                headline="Failed to summarize",
                detail=str(e),
                matter_file_summaries=matter_file_summaries,
            )
        else:
            return MatterSummary(
                key=expanded_matter.matter.key,
                headline=result.headline,
                detail=result.detail,
                matter_file_summaries=matter_file_summaries,
            )

    def summarize_expanded_transcript(
        self,
        expanded_transcript: ExpandedTranscript,
    ) -> TranscriptSummary:
        """
        Summarize an expanded transcript.

        On failure, return a summary with warning text (allowing further
        summarization to continue).
        """
        try:
            logger.info(
                "transcript %s :: downloading ", expanded_transcript.transcript.key
            )
            tm = expanded_transcript.get_transcript_model(self.connection)
            logger.info(
                "transcript %s :: extracting", expanded_transcript.transcript.key
            )
            text = extract_text_from_transcript_model(tm)
            logger.info(
                "transcript %s :: summarizing", expanded_transcript.transcript.key
            )
            result = self.summarize_transcript(text)
        except Exception as e:
            logger.exception(
                "Failed to summarize transcript %s", expanded_transcript.transcript.key
            )
            return TranscriptSummary(
                key=expanded_transcript.transcript.key,
                headline="Failed to summarize",
                detail=str(e),
                uri=t.cast(str, expanded_transcript.file.uri),
            )
        else:
            return TranscriptSummary(
                key=expanded_transcript.transcript.key,
                headline=result.headline,
                detail=result.detail,
                uri=t.cast(str, expanded_transcript.file.uri),
            )

    def summarize_expanded_session(
        self, expanded_session: ExpandedSession
    ) -> SessionSummary:
        """Summarize an expanded session by summarizing its transcript."""
        transcript = expanded_session.transcript
        transcript_summary = (
            self.summarize_expanded_transcript(transcript) if transcript else None
        )
        # Just pass the transcript summary onward; is there anything else we
        # should be summarizing here?
        return SessionSummary(
            key=expanded_session.session.key,
            headline=transcript_summary.headline
            if transcript_summary
            else "Session has no transcript",
            detail=transcript_summary.detail if transcript_summary else "",
            transcript_summary=transcript_summary,
        )

    def summarize_expanded_event(self, expanded_event: ExpandedEvent) -> EventSummary:
        """
        Summarize an expanded event.

        On failure, return a summary with warning text (allowing further
        summarization to continue).
        """
        matter_summaries = tuple(
            self.summarize_expanded_matter(matter) for matter in expanded_event.matters
        )
        session_summaries = tuple(
            self.summarize_expanded_session(session)
            for session in expanded_event.sessions
        )
        text = (
            "\n\n".join(matter_summary.detail for matter_summary in matter_summaries)
            + "\n\n"
            + "\n\n".join(
                session_summary.detail for session_summary in session_summaries
            )
        )
        try:
            logger.info("event %s :: summarizing", expanded_event.event.key)
            body_name = expanded_event.body_name or "City Council"
            friendly_date_fmt = "%A, %B %-d, %Y"
            result = self.summarize_event(body_name, friendly_date_fmt, text)
        except Exception as e:
            logger.exception("Failed to summarize event %s", expanded_event.event.key)
            return EventSummary(
                key=expanded_event.event.key,
                headline="Failed to summarize",
                detail=str(e),
                dt=expanded_event.dt,
                matter_summaries=matter_summaries,
                session_summaries=session_summaries,
            )
        else:
            return EventSummary(
                key=expanded_event.event.key,
                headline=result.headline,
                detail=result.detail,
                dt=expanded_event.dt,
                matter_summaries=matter_summaries,
                session_summaries=session_summaries,
            )
