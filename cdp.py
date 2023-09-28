#!/usr/bin/env python3


import datetime
import json
import os
import typing as t
from inspect import getmembers

import click
from cdp_data import CDPInstances
from fireo.queries.query_wrapper import ReferenceDocLoader

from summarize.queries import get_events_for_slug, get_expanded_events_for_slug

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def _get_available_cdp_instances() -> dict[str, str]:
    """
    Return a dictionary mapping a friendly name for the CLI
    (like "Seattle") to a CDP infrastructure slug (like
    "cdp-seattle-21723dcf").

    Try not to duplicate the CDPInstances class; rather, use it.
    """
    return {
        name: value
        for name, value in getmembers(CDPInstances)
        if isinstance(value, str) and not name.startswith("__")
    }


INSTANCES = _get_available_cdp_instances()
INSTANCE_NAMES = list(INSTANCES.keys())
CDP_INSTANCE_NAME: str | None = os.getenv("CDP_INSTANCE_NAME", None)


FRIENDLY_DT_FORMAT = "%a %b %-d, %Y @ %-I:%M%p"


# ------------------------------------------------------------
# Utilities: human and machine-friendly output formatters
# ------------------------------------------------------------


class SupportsToDict(t.Protocol):
    def to_dict(self) -> dict:
        ...


class FireOAwareJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that supports datetime objects and
    fireo.queries.query_wrapper.ReferenceDocLoader objects (since, alas,
    those end up in `model_instance.to_dict()` results).
    """

    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif isinstance(obj, ReferenceDocLoader):
            return f"{obj.field.model_ref.collection_name}/{obj.ref.id}"
        else:
            return super().default(obj)


def log_jsonl(items: t.Iterable[SupportsToDict]):
    """Dump an iterable of SupportsToDict as JSON lines."""
    for item in items:
        print(json.dumps(item.to_dict(), cls=FireOAwareJSONEncoder))


def log_json(items: t.Iterable[SupportsToDict]):
    """Dump an arbitrary object as a pretty-printed JSON blob."""
    print(
        json.dumps(
            [item.to_dict() for item in items], cls=FireOAwareJSONEncoder, indent=2
        )
    )


def log(items: t.Iterable[SupportsToDict], jsonl: bool):
    """Dump an arbitrary object as a pretty-printed JSON blob."""
    if jsonl:
        log_jsonl(items)
    else:
        log_json(items)


# ------------------------------------------------------------
# Click CLI
# ------------------------------------------------------------


@click.group()
def cdp():
    """A simple CDP data command line interface."""
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
def instances(jsonl: bool):
    """
    Print a list of available CDP instances.

    These can be used with the --instance (-i) option on other commands.
    """
    for name in INSTANCE_NAMES:
        print(json.dumps({"name": name, "slug": INSTANCES[name]}) if jsonl else name)


@cdp.group()
def events():
    """Fetch events from a CDP instance."""
    pass


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
def list(
    instance: str,
    start_date: datetime.datetime | None,
    end_date: datetime.datetime | None,
    jsonl: bool,
    **kwargs,
):
    """Create a short list of matching events for a given CDP instance."""
    events = get_events_for_slug(INSTANCES[instance], start_date, end_date)
    if jsonl:
        log_jsonl(events)
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
def expand(
    instance: str,
    start_date: datetime.datetime | None,
    end_date: datetime.datetime | None,
    jsonl: bool,
    **kwargs,
):
    """Fetch and expand events from a CDP instance."""
    events = get_expanded_events_for_slug(
        INSTANCES[instance], start_date=start_date, end_date=end_date
    )
    log(events, jsonl)


if __name__ == "__main__":
    cdp()
