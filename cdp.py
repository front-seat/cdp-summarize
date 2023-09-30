#!/usr/bin/env python3


import datetime
import json
import logging
import os
import typing as t

import click
from fireo.queries.query_wrapper import ReferenceDocLoader

from summarize.connection import INSTANCE_NAMES, INSTANCES, CDPConnection
from summarize.queries import get_events, get_expanded_events
from summarize.summaries import summarize_expanded_event

FRIENDLY_DT_FORMAT = "%a %b %-d, %Y @ %-I:%M%p"


# ------------------------------------------------------------
# Utilities: human and machine-friendly output formatters
# ------------------------------------------------------------


class SupportsToDict(t.Protocol):
    def to_dict(self) -> dict:
        """Return a dictionary representation of this object."""
        ...


class FireOAwareJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that supports datetime objects and
    fireo.queries.query_wrapper.ReferenceDocLoader objects (since, alas,
    those end up in `model_instance.to_dict()` results).
    """

    def default(self, obj: t.Any) -> t.Any:
        """Encode datetime and ReferenceDocLoader objects."""
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif isinstance(obj, ReferenceDocLoader):
            return f"{obj.field.model_ref.collection_name}/{obj.ref.id}"
        else:
            return super().default(obj)


def format_jsonl(items: t.Iterable[SupportsToDict]) -> str:
    """Format an iterable of SupportsToDict as JSON lines."""
    return "\n".join(
        json.dumps(item.to_dict(), cls=FireOAwareJSONEncoder) for item in items
    )


def format_json(items: t.Iterable[SupportsToDict]) -> str:
    """Format an iterable of SupportsToDict as a pretty-printed JSON string."""
    return json.dumps(
        [item.to_dict() for item in items], cls=FireOAwareJSONEncoder, indent=2
    )


def fmt(items: t.Iterable[SupportsToDict], jsonl: bool) -> str:
    """Dump an iterable of SupportsToDict."""
    return format_jsonl(items) if jsonl else format_json(items)


# ------------------------------------------------------------
# Click CLI
# ------------------------------------------------------------


@click.group()
def cdp() -> None:
    """Create a CLI for the CDP."""
    pass


@cdp.command()
@click.option(
    "-j",
    "--jsonl",
    is_flag=True,
    default=False,
    required=False,
    help="Output as JSON lines instead of formatted JSON.",
)
def instances(jsonl: bool) -> None:
    """
    Print a list of available CDP instances.

    These can be used with the --instance (-i) option on other commands.
    """
    for name in INSTANCE_NAMES:
        print(json.dumps({"name": name, "slug": INSTANCES[name]}) if jsonl else name)


@cdp.group()
def events() -> None:
    """Fetch events from a CDP instance."""
    pass


@events.command(name="list")
@click.option(
    "-i",
    "--instance",
    type=click.Choice(INSTANCE_NAMES),
    required=True,
    help='The CDP instance name (like "Seattle") to fetch events from.',
)
@click.option(
    "-j",
    "--jsonl",
    is_flag=True,
    default=False,
    required=False,
    help="Output as JSON lines.",
)
@click.option(
    "-s",
    "--start-date",
    type=click.DateTime(),
    default=None,
    required=False,
    help="The start date at or after which to get events from.",
)
@click.option(
    "-e",
    "--end-date",
    type=click.DateTime(),
    default=None,
    required=False,
    help="The end date before which to get events from.",
)
def list_cmd(
    instance: str,
    start_date: datetime.datetime | None,
    end_date: datetime.datetime | None,
    jsonl: bool,
    **kwargs: t.Any,
) -> None:
    """Create a short list of matching events for a given CDP instance."""
    connection = CDPConnection.for_name(instance)
    events = get_events(connection, start_date, end_date)
    if jsonl:
        print(format_jsonl(events))
    else:
        # Make it human-readable
        for event in events:
            event_dt_local = t.cast(
                datetime.datetime, event.event_datetime
            ).astimezone()
            event_dt_format = event_dt_local.strftime(FRIENDLY_DT_FORMAT)
            print(f"{event.id} {event_dt_format}")


@events.command()
@click.option(
    "-i",
    "--instance",
    type=click.Choice(INSTANCE_NAMES),
    required=True,
    help='The CDP instance name (like "Seattle") to fetch events from.',
)
@click.option(
    "-j",
    "--jsonl",
    is_flag=True,
    default=False,
    required=False,
    help="Output as JSON lines.",
)
@click.option(
    "-s",
    "--start-date",
    type=click.DateTime(),
    default=None,
    required=False,
    help="The start date at or after which to get events from.",
)
@click.option(
    "-e",
    "--end-date",
    type=click.DateTime(),
    default=None,
    required=False,
    help="The end date before which to get events from.",
)
@click.option(
    "--id",
    "event_ids",
    multiple=True,
    required=False,
    help="The ID of an event (or events) to expand.",
)
def expand(
    instance: str,
    event_ids: tuple[str],
    start_date: datetime.datetime | None,
    end_date: datetime.datetime | None,
    jsonl: bool,
    **kwargs: t.Any,
) -> None:
    """Fetch and expand events from a CDP instance."""
    connection = CDPConnection.for_name(instance)
    events = get_expanded_events(connection, start_date, end_date, list(event_ids))
    print(fmt(events, jsonl))


@events.command()
@click.option(
    "-i",
    "--instance",
    type=click.Choice(INSTANCE_NAMES),
    required=True,
    help='The CDP instance name (like "Seattle") to fetch events from.',
)
@click.option(
    "-j",
    "--jsonl",
    is_flag=True,
    default=False,
    required=False,
    help="Output as JSON lines.",
)
@click.option(
    "-s",
    "--start-date",
    type=click.DateTime(),
    default=None,
    required=False,
    help="The start date at or after which to get events from.",
)
@click.option(
    "-e",
    "--end-date",
    type=click.DateTime(),
    default=None,
    required=False,
    help="The end date before which to get events from.",
)
@click.option(
    "--id",
    "event_ids",
    multiple=True,
    required=False,
    help="The ID of an event (or events) to expand.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    required=False,
    help="Log summarization progress to stderr.",
)
def summarize(
    instance: str,
    event_ids: tuple,
    start_date: datetime.datetime | None,
    end_date: datetime.datetime | None,
    jsonl: bool,
    verbose: bool,
    **kwargs: t.Any,
) -> None:
    """Fetch and summarize events from a CDP instance."""
    # Make sure we have an OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY", None):
        raise click.ClickException(
            "You must set the OPENAI_API_KEY environment variable to use this command."
        )

    if verbose:
        logging.basicConfig(level=logging.INFO)

    connection = CDPConnection.for_name(instance)
    events = get_expanded_events(connection, start_date, end_date, list(event_ids))
    summaries = (summarize_expanded_event(connection, event) for event in events)
    print(fmt(summaries, jsonl))


if __name__ == "__main__":
    cdp()
