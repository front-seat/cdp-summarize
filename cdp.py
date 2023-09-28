#!/usr/bin/env python3

#
# This is a simple command-line interface to the CDP data and infrastructure.
#
# It offers only a tiny bit of access, just enough to get started with
# summarization.
#
# Currently, you can list available CDP instances:
#
#       ./cdp.py instances
#
# And you can get events from a CDP instance between given dates:
#
#       ./cdp.py events -c Seattle -s 2023-08-01 -e 2023-08-15
#
# Events are *expanded* to include their related sessions and other
# relevant data for summarization, including transcripts, matters, and
# matter files.
#
# You can set a global instance name with the CDP_INSTANCE_NAME environment
# variable:
#
#       export CDP_INSTANCE_NAME=KingCounty
#
# or:
#
#       CDP_INSTANCE_NAME=Seattle ./cdp.py events -s 2023-08-01
#
# By default all commands output JSON lines data. That said, you can
# pretty print it with --pretty (or -p):
#
#       ./cdp.py events -d 2023-08-01 --pretty
#
# It's not *that* pretty, but hey.
#
# You can use default JSON Lines to pipe data into other tools, like jq, or
# back into the CLI for further processing.
#
# With this, you can (for example) summarize events:
#
#       ./cdp.py events -d 2023-08-01 | ./cdp.py summarize
#
# That's about all for now. It's super early-stage code. Do we like this
# general approach? Is it useful? Frustrating? Time will tell. :-)


import datetime
import json
import os
import sys
import typing as t
from inspect import getmembers

import click
from cdp_data import CDPInstances
from fireo.queries.query_wrapper import ReferenceDocLoader

from summarize.queries import get_expanded_events_for_slug

# ------------------------------------------------------------
# Utilities: infrastructure names and slugs
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


# ------------------------------------------------------------
# Utilities: human and machine-friendly output formatters
# ------------------------------------------------------------


@t.runtime_checkable
class SupportsToDict(t.Protocol):
    def to_dict(self) -> dict:
        ...


class CustomJSONEncoder(json.JSONEncoder):
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


def _log_dict(d: dict, pretty: bool, output: t.TextIO):
    """Dump a dictionary as either JSON lines or formatted JSON."""
    print(
        json.dumps(d, cls=CustomJSONEncoder, indent=2 if pretty else None), file=output
    )


def log_iterable(i: t.Iterable[SupportsToDict], pretty: bool, output: t.TextIO):
    """Dump an iterable of SupportsToDict as either JSON lines or formatted JSON."""
    if pretty:
        items = list(i)
        print(
            json.dumps(
                [item.to_dict() for item in items], cls=CustomJSONEncoder, indent=2
            ),
            file=output,
        )
    else:
        for item in i:
            _log_dict(item.to_dict(), pretty, output)


# ------------------------------------------------------------
# click-based CLI options groups
# ------------------------------------------------------------


# CLI options that apply to all sub-commands that output data.
# That's, uh, all of them!
OUTPUT_OPTIONS = [
    click.option(
        "-o",
        "--output",
        type=click.File("wt"),
        default=sys.stdout,
        help="The output file to write to.",
    ),
    click.option(
        "-p",
        "--pretty",
        is_flag=True,
        default=False,
        help="Indent and pretty-print the output JSON.",
    ),
]

# CLI options that apply to all sub-commands that consume and output data.
# That's, uh, most of them!
STD_OPTIONS = OUTPUT_OPTIONS + [
    click.option(
        "-c",
        "--instance",
        type=click.Choice(INSTANCE_NAMES),
        required=True,
        default=CDP_INSTANCE_NAME,
        help="The CDP instance to get sessions for.",
    ),
    click.option(
        "-i",
        "--input",
        type=click.File("rt"),
        default=sys.stdin,
        help="The input file to read from.",
    ),
]


def options_group(options: list[t.Callable]):
    """A decorator to add a group of options to a command."""

    def decorator(func):
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


# ------------------------------------------------------------
# click-based CLI
# ------------------------------------------------------------


@click.group()
def cdp():
    """A simple CDP data command line interface."""
    pass


@cdp.command()
@options_group(OUTPUT_OPTIONS)
def instances(output: t.TextIO, pretty: bool):
    """
    Print a list of available CDP instances.

    These can be used with the --instance (-c) option on other commands.
    """
    for name in INSTANCE_NAMES:
        out = name if pretty else json.dumps({"name": name, "slug": INSTANCES[name]})
        print(out, file=output)


@cdp.command()
@options_group(STD_OPTIONS)
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
def events(
    instance: str,
    start_date: datetime.datetime | None,
    end_date: datetime.datetime | None,
    pretty: bool,
    output: t.TextIO,
    **kwargs,
):
    """Fetch events from a CDP instance."""
    events = get_expanded_events_for_slug(
        INSTANCES[instance], start_date=start_date, end_date=end_date
    )
    log_iterable(events, pretty, output)


if __name__ == "__main__":
    cdp()
