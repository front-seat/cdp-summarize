import datetime
import json
import typing as t
from dataclasses import dataclass

import urllib3
from cdp_backend.database import models as cdp_models
from cdp_data.utils import connect_to_infrastructure
from fireo.queries.query_wrapper import ReferenceDocLoader
from gcsfs import GCSFileSystem

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


class CDPConnection:
    """
    Represents an active connection to CDP infrastructure.

    This is a bit awkward because FireO seems to assume one global connection:
    if you call `connect()` multiple times, you'll need to drop your old
    connection instances.

    (REVISIT. Maybe there's a nice way to say 'run your query on *this*
    specific connection' with FireO?)
    """

    infrastructure_slug: str
    file_system: GCSFileSystem

    def __init__(self, infrastructure_slug: str, file_system: GCSFileSystem):
        self.infrastructure_slug = infrastructure_slug
        self.file_system = file_system

    @classmethod
    def connect(cls, infrastructure_slug: str) -> t.Self:
        return cls(infrastructure_slug, connect_to_infrastructure(infrastructure_slug))


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

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript.to_dict(),
            "file": self.file.to_dict(),
        }

    def fetch(self, connection: CDPConnection) -> dict:
        with connection.file_system.open(self.file.uri, "rt") as f:
            content = json.load(f)
        return content


@dataclass
class ExpandedMatter:
    """A Matter together with its related files."""

    matter: cdp_models.Matter
    files: list[cdp_models.MatterFile]

    def to_dict(self) -> dict:
        return {
            "matter": self.matter.to_dict(),
            "files": [file.to_dict() for file in self.files],
        }

    def fetch_file(
        self, connection: CDPConnection, file: cdp_models.MatterFile
    ) -> tuple[str, bytes]:
        """
        Fetch a MatterFile's content from the web.

        Return both its reported content-type and its content.
        """
        response = urllib3.request("GET", t.cast(str, file.uri))
        return (
            response.headers.get("content-type", "application/octet-stream"),
            response.data,
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
    sessions: list[ExpandedSession]
    matters: list[ExpandedMatter]

    def to_dict(self) -> dict:
        return {
            "event": self.event.to_dict(),
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

    return ExpandedEvent(
        event=event,
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
) -> t.Iterable[cdp_models.Event]:
    """Get all events within the provided times."""
    events = cdp_models.Event.collection
    if start_date is not None:
        events = events.filter("event_datetime", ">=", start_date)
    if end_date is not None:
        events = events.filter("event_datetime", "<", end_date)
    return t.cast(t.Iterable[cdp_models.Event], events.fetch())


def get_events_for_slug(
    infrastructure_slug: str,
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
) -> t.Iterable[cdp_models.Event]:
    """
    Get all events within a provided times.

    This is probably the droid you're looking for.
    """
    connection = CDPConnection.connect(infrastructure_slug)
    return get_events(connection, start_date, end_date)


def get_expanded_events(
    connection: CDPConnection,
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
) -> t.Iterable[ExpandedEvent]:
    """Get all events expanded with related sessions within a provided times."""
    return [
        expand_event(connection, event)
        for event in get_events(connection, start_date, end_date)
    ]


def get_expanded_events_for_slug(
    infrastructure_slug: str,
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
) -> t.Iterable[ExpandedEvent]:
    """
    Get all events expanded with related sessions within a provided times.

    This is probably the droid you're looking for.
    """
    connection = CDPConnection.connect(infrastructure_slug)
    return get_expanded_events(connection, start_date, end_date)
