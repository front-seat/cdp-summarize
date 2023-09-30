# summaries.py -- higher level code that's CDP model aware and uses
# stuff in extract.py and llm.py to build the summaries.

import datetime
import io
import logging
import typing as t
from dataclasses import dataclass

from cdp_backend.database import models as cdp_models

from .connection import CDPConnection
from .extract import extract_text_from_bytes, extract_text_from_transcript
from .llm import SummarizationResult, summarize_openai
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
# Model summarization prompts & methods
# ------------------------------------------------------------

CONCISE_DETAIL = """Write a concise summary of the following text. Include the most important details:

TEXT:::
{text}
:::END_TEXT

CONCISE_SUMMARY:"""  # noqa: E501


CONCISE_HEADLINE = """Write a concise and extremely compact headline (one sentence or less) for the following text. Capture only the most salient detail or two:

TEXT:::
{text}
:::END_TEXT

CONCISE_COMPACT_HEADLINE:"""  # noqa: E501


def summarize_concise(text: str) -> SummarizationResult:
    """Use the concise summary style to summarize text."""
    return summarize_openai(
        text=text,
        detail_combine_template=CONCISE_DETAIL,
        headline_combine_template=CONCISE_HEADLINE,
    )


MATTER_DETAIL = """The following is a set of summaries of documents related to a single matter considered by a city council body. Write a concise summary of the following text, which is titled "{title}". Include the most important details:

"{text}"

CITY_COUNCIL_MATTER_SUMMARY:"""  # noqa: E501


MATTER_HEADLINE = """The following is a set of summaries of documents related to a single matter considered by a city council body. Write a concise and extremely compact headline (one sentence or less) for the action, which is titled "{title}". Capture only the most salient detail or two:

"{text}"

CITY_COUNCIL_MATTER_HEADLINE:"""  # noqa: E501


def summarize_matter(title: str, text: str) -> SummarizationResult:
    """Use the matter summary style to summarize text."""
    return summarize_openai(
        text=text,
        detail_combine_template=MATTER_DETAIL,
        headline_combine_template=MATTER_HEADLINE,
        context={"title": title},
    )


TRANSCRIPT_DETAIL = """The following is a transcript of a recent city council meeting. Concisely summarize it:

"{text}"

CITY_COUNCIL_TRANSCRIPT_SUMMARY:"""  # noqa: E501


TRANSCRIPT_HEADLINE = """The following is a transcript of a recent city council meeting. Write a concise and extremely compact headline (one sentence or less):

"{text}"

CITY_COUNCIL_TRANSCRIPT_HEADLINE:"""  # noqa: E501


def summarize_transcript(text: str) -> SummarizationResult:
    """Use the transcript summary style to summarize text."""
    return summarize_openai(
        text=text,
        detail_combine_template=TRANSCRIPT_DETAIL,
        headline_combine_template=TRANSCRIPT_HEADLINE,
    )


EVENT_DETAIL = """The following is a set of summaries of documents related to a single event at the {body_name} on {dt_fmt}. Write a concise summary, including the most important details:

"{text}"

CITY_COUNCIL_EVENT_SUMMARY:"""  # noqa: E501


EVENT_HEADLINE = """The following is a set of summaries of documents related to a single event at the {body_name} on {dt_fmt}. Write a concise and extremely compact headline (one sentence or less). Capture only the most salient detail or two:

"{text}"

CITY_COUNCIL_EVENT_HEADLINE:"""  # noqa: E501


def summarize_event(body_name: str, dt_fmt: str, text: str) -> SummarizationResult:
    """Use the event summary style to summarize text."""
    return summarize_openai(
        text=text,
        detail_combine_template=EVENT_DETAIL,
        headline_combine_template=EVENT_HEADLINE,
        context={"body_name": body_name, "dt_fmt": dt_fmt},
    )


# ------------------------------------------------------------
# Top-Level Model Summarization Methods
# ------------------------------------------------------------


def summarize_matter_file(
    connection: CDPConnection,
    expanded_matter: ExpandedMatter,
    matter_file: cdp_models.MatterFile,
) -> MatterFileSummary:
    """
    Summarize an arbitrary matter file.

    On failure, return a summary with warning text (allowing further
    summarization to continue).
    """
    text = ""
    try:
        logger.info(
            "matter %s file %s :: downloading ",
            expanded_matter.matter.key,
            matter_file.key,
        )
        content_type, bytes = expanded_matter.get_file(connection, matter_file)
        logger.info(
            "matter %s file %s :: extracting",
            expanded_matter.matter.key,
            matter_file.key,
        )
        text = extract_text_from_bytes(io.BytesIO(bytes), content_type)
        logger.info(
            "matter %s file %s :: summarizing",
            expanded_matter.matter.key,
            matter_file.key,
        )
        result = summarize_concise(text)
    except Exception as e:
        logger.exception("Failed to summarize matter file %s", matter_file.key)
        return MatterFileSummary(
            key=matter_file.key,
            headline="Failed to summarize",
            detail=str(e),
            text=text,
            uri=t.cast(str, matter_file.uri),
        )
    else:
        logger.info(
            "matter %s file %s :: summarized:\n\tHeadline: %s\n\tDetail: %s",
            expanded_matter.matter.key,
            matter_file.key,
            result.headline,
            result.detail,
        )
        return MatterFileSummary(
            key=matter_file.key,
            headline=result.headline,
            detail=result.detail,
            text=result.text,
            uri=t.cast(str, matter_file.uri),
        )


def summarize_expanded_matter(
    connection: CDPConnection, expanded_matter: ExpandedMatter
) -> MatterSummary:
    """
    Summarize a legislative matter by summarizing all of its files and
    combining the results.

    On failure, return a summary with warning text (allowing further
    summarization to continue).
    """

    matter_file_summaries = tuple(
        summarize_matter_file(connection, expanded_matter, matter_file)
        for matter_file in expanded_matter.files
    )
    text = "\n\n".join(
        matter_file_summary.detail for matter_file_summary in matter_file_summaries
    )
    try:
        logger.info("matter %s :: summarizing", expanded_matter.matter.key)
        result = summarize_matter(t.cast(str, expanded_matter.matter.title), text)
    except Exception as e:
        logger.exception("Failed to summarize matter %s", expanded_matter.matter.key)
        return MatterSummary(
            key=expanded_matter.matter.key,
            headline="Failed to summarize",
            detail=str(e),
            text=text,
            matter_file_summaries=matter_file_summaries,
        )
    else:
        return MatterSummary(
            key=expanded_matter.matter.key,
            headline=result.headline,
            detail=result.detail,
            text=text,
            matter_file_summaries=matter_file_summaries,
        )


def summarize_expanded_transcript(
    connection: CDPConnection,
    expanded_transcript: ExpandedTranscript,
) -> TranscriptSummary:
    """
    Summarize an expanded transcript.

    On failure, return a summary with warning text (allowing further
    summarization to continue).
    """
    text = ""
    try:
        logger.info("transcript %s :: downloading ", expanded_transcript.transcript.key)
        data = expanded_transcript.get_data(connection)
        logger.info("transcript %s :: extracting", expanded_transcript.transcript.key)
        text = extract_text_from_transcript(data)
        logger.info("transcript %s :: summarizing", expanded_transcript.transcript.key)
        result = summarize_transcript(text)
    except Exception as e:
        logger.exception(
            "Failed to summarize transcript %s", expanded_transcript.transcript.key
        )
        return TranscriptSummary(
            key=expanded_transcript.transcript.key,
            headline="Failed to summarize",
            detail=str(e),
            text=text,
            uri=t.cast(str, expanded_transcript.file.uri),
        )
    else:
        logger.info(
            "transcript %s :: summarized:\n\tHeadline: %s\n\tDetail: %s",
            expanded_transcript.transcript.key,
            result.headline,
            result.detail,
        )
        return TranscriptSummary(
            key=expanded_transcript.transcript.key,
            headline=result.headline,
            detail=result.detail,
            text=text,
            uri=t.cast(str, expanded_transcript.file.uri),
        )


def summarize_expanded_session(
    connection: CDPConnection, expanded_session: ExpandedSession
) -> SessionSummary:
    """Summarize an expanded session by summarizing its transcript."""

    transcript = expanded_session.transcript
    transcript_summary = (
        summarize_expanded_transcript(connection, transcript) if transcript else None
    )
    # Just pass the transcript summary onward; is there anything else we
    # should be summarizing here?
    return SessionSummary(
        key=expanded_session.session.key,
        headline=transcript_summary.headline
        if transcript_summary
        else "Session has no transcript",
        detail=transcript_summary.detail if transcript_summary else "",
        text=transcript_summary.text if transcript_summary else "",
        transcript_summary=transcript_summary,
    )


def summarize_expanded_event(
    connection: CDPConnection, expanded_event: ExpandedEvent
) -> EventSummary:
    """
    Summarize an expanded event.

    On failure, return a summary with warning text (allowing further
    summarization to continue).
    """
    matter_summaries = tuple(
        summarize_expanded_matter(connection, matter)
        for matter in expanded_event.matters
    )
    session_summaries = tuple(
        summarize_expanded_session(connection, session)
        for session in expanded_event.sessions
    )
    text = (
        "\n\n".join(matter_summary.detail for matter_summary in matter_summaries)
        + "\n\n"
        + "\n\n".join(session_summary.detail for session_summary in session_summaries)
    )
    try:
        logger.info("event %s :: summarizing", expanded_event.event.key)
        body_name = expanded_event.body_name or "City Council"
        friendly_date_fmt = "%A, %B %-d, %Y"
        result = summarize_event(body_name, friendly_date_fmt, text)
    except Exception as e:
        logger.exception("Failed to summarize event %s", expanded_event.event.key)
        return EventSummary(
            key=expanded_event.event.key,
            headline="Failed to summarize",
            detail=str(e),
            text=text,
            dt=expanded_event.dt,
            matter_summaries=matter_summaries,
            session_summaries=session_summaries,
        )
    else:
        return EventSummary(
            key=expanded_event.event.key,
            headline=result.headline,
            detail=result.detail,
            text=text,
            dt=expanded_event.dt,
            matter_summaries=matter_summaries,
            session_summaries=session_summaries,
        )
