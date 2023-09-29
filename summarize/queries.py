import datetime
import json
import logging
import typing as t
import urllib.parse
import urllib.request
from dataclasses import dataclass

from cdp_backend.database import models as cdp_models
from fireo.queries.query_wrapper import ReferenceDocLoader

from .connection import CDPConnection

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Expanded "Models"
# ------------------------------------------------------------

# CONSIDER: cdp_data has its db_utils.py, which has tools to
# expand refs, primarily for creating pandas dataframes. What
# we want to do here is a little more narrowly scoped, but maybe
# I should have built on top of that?


@dataclass
class ExpandedTranscript:
    """A Transcript together with its related file."""

    transcript: cdp_models.Transcript
    file: cdp_models.File

    def __post_init__(self):
        """Validate our content."""
        file_scheme = urllib.parse.urlparse(t.cast(str, self.file.uri)).scheme
        if file_scheme != "gs":
            raise ValueError(
                f"Transcript file URI scheme must be gs:// not {file_scheme}"
            )

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript.to_dict(),
            "file": self.file.to_dict(),
        }

    def get_data(self, connection: CDPConnection) -> dict:
        logger.info("Fetching transcript data from %s", self.file.uri)
        with connection.file_system.open(self.file.uri, "rt") as f:
            data = json.load(f)
        return data


@dataclass
class ExpandedMatter:
    """A Matter together with its related files."""

    matter: cdp_models.Matter
    files: list[cdp_models.MatterFile]

    def __post_init__(self):
        """Validate our content."""
        for file in self.files:
            file_scheme = urllib.parse.urlparse(t.cast(str, file.uri)).scheme
            if file_scheme not in ("http", "https"):
                raise ValueError(
                    f"Matter file URI scheme must be https?:// not {file_scheme}"
                )

    def to_dict(self) -> dict:
        return {
            "matter": self.matter.to_dict(),
            "files": [file.to_dict() for file in self.files],
        }

    def get_file(
        self, connection: CDPConnection, file: cdp_models.MatterFile
    ) -> tuple[str, bytes]:
        """
        Fetch a MatterFile's content from the web.

        Return both its reported content-type and its content.
        """
        logger.info("Fetching matter file data from %s", file.uri)
        with urllib.request.urlopen(t.cast(str, file.uri)) as response:
            return (
                response.headers.get("content-type", "application/octet-stream"),
                response.read(),
            )


@dataclass
class ExpandedSession:
    """A session together with its related transcript."""

    session: cdp_models.Session
    transcript: ExpandedTranscript | None

    def to_dict(self) -> dict:
        return {
            "session": self.session.to_dict(),
            "transcript": self.transcript.to_dict() if self.transcript else None,
        }


@dataclass
class ExpandedEvent:
    """An event together with its related sessions and matters."""

    event: cdp_models.Event
    body_name: str
    dt: datetime.datetime
    sessions: list[ExpandedSession]
    matters: list[ExpandedMatter]

    def to_dict(self) -> dict:
        return {
            "event": self.event.to_dict(),
            "body_name": self.body_name,
            "dt": self.dt.isoformat(),
            "sessions": [session.to_dict() for session in self.sessions],
            "matters": [matter.to_dict() for matter in self.matters],
        }


# ------------------------------------------------------------
# Model expansion utilities
# ------------------------------------------------------------


def expand_event(
    connection: CDPConnection,
    event: cdp_models.Event,
) -> ExpandedEvent:
    """Expand an event into its related sessions."""
    sessions = (
        cdp_models.Session.collection.filter(event_ref=event.key)
        .order("session_index")
        .fetch()
    )
    matters = matters_for_event(event)
    body = t.cast(cdp_models.Body, t.cast(ReferenceDocLoader, event.body_ref).get())
    return ExpandedEvent(
        event=event,
        body_name=t.cast(str, body.name),
        dt=t.cast(datetime.datetime, event.event_datetime),
        sessions=[
            expand_session(connection, t.cast(cdp_models.Session, session))
            for session in sessions
        ],
        matters=[expand_matter(connection, matter) for matter in matters],
    )


def matters_for_event(event: cdp_models.Event) -> t.Iterable[cdp_models.Matter]:
    """Return all matters related to a given event."""
    # TODO I assume there's a better, more firestore-database-y way to do this?
    event_minutes_items = t.cast(
        t.Iterable[cdp_models.EventMinutesItem],
        cdp_models.EventMinutesItem.collection.filter(event_ref=event.key)
        .order("index")
        .fetch(),
    )
    minutes_items = t.cast(
        t.Iterable[cdp_models.MinutesItem],
        (
            t.cast(ReferenceDocLoader, event_minutes_item.minutes_item_ref).get()
            for event_minutes_item in event_minutes_items
        ),
    )
    matters = t.cast(
        t.Iterable[cdp_models.Matter],
        (
            t.cast(ReferenceDocLoader, matter_ref).get()
            for minutes_item in minutes_items
            if (matter_ref := minutes_item.matter_ref)
        ),
    )
    return matters


def expand_session(
    connection: CDPConnection,
    session: cdp_models.Session,
) -> ExpandedSession:
    """Expand a session into its related transcript and matters."""
    transcript = (
        cdp_models.Transcript.collection.filter(session_ref=session.key)
        .order("-created")
        .get()
    )
    return ExpandedSession(
        session=session,
        transcript=expand_transcript(
            connection, t.cast(cdp_models.Transcript | None, transcript)
        ),
    )


def expand_matter(
    connection: CDPConnection,
    matter: cdp_models.Matter,
) -> ExpandedMatter:
    """Expand a matter into its related files."""
    files = cdp_models.MatterFile.collection.filter(matter_ref=matter.key).fetch()
    return ExpandedMatter(
        matter=matter, files=[t.cast(cdp_models.MatterFile, file) for file in files]
    )


def expand_transcript(
    connection: CDPConnection,
    transcript: cdp_models.Transcript | None,
) -> ExpandedTranscript | None:
    if not transcript:
        return None
    return ExpandedTranscript(
        transcript=transcript,
        file=t.cast(
            cdp_models.File, t.cast(ReferenceDocLoader, transcript.file_ref).get()
        ),
    )


# ------------------------------------------------------------
# Queries
# ------------------------------------------------------------


def get_events(
    connection: CDPConnection,
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
    event_ids: list[str] | None = None,
) -> t.Iterable[cdp_models.Event]:
    """Get all events within the provided times."""
    events = cdp_models.Event.collection
    if start_date is not None:
        events = events.filter("event_datetime", ">=", start_date)
    if end_date is not None:
        events = events.filter("event_datetime", "<", end_date)
    if event_ids:
        events = events.filter("id", "in", event_ids)
    return t.cast(t.Iterable[cdp_models.Event], events.fetch())


def get_events_for_slug(
    infrastructure_slug: str,
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
    event_ids: list[str] | None = None,
) -> t.Iterable[cdp_models.Event]:
    """
    Get all events within a provided times.

    This is probably the droid you're looking for.
    """
    connection = CDPConnection.connect(infrastructure_slug)
    return get_events(connection, start_date, end_date, event_ids)


def get_expanded_events(
    connection: CDPConnection,
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
    event_ids: list[str] | None = None,
) -> t.Iterable[ExpandedEvent]:
    """Get all events expanded with related sessions within a provided times."""
    return [
        expand_event(connection, event)
        for event in get_events(connection, start_date, end_date, event_ids)
    ]


def get_expanded_events_for_slug(
    infrastructure_slug: str,
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
    event_ids: list[str] | None = None,
) -> t.Iterable[ExpandedEvent]:
    """
    Get all events expanded with related sessions within a provided times.

    This is probably the droid you're looking for.
    """
    connection = CDPConnection.connect(infrastructure_slug)
    return get_expanded_events(connection, start_date, end_date, event_ids)
