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
# And you can get sessions from a CDP instance:
#
#       ./cdp.py sessions -c Seattle -d 2023-08-01
#
# You can set a global instance name with the CDP_INSTANCE_NAME environment
# variable:
#
#       export CDP_INSTANCE_NAME=KingCounty
#
# or:
#
#       CDP_INSTANCE_NAME=Seattle ./cdp.py sessions -d 2023-08-01
#
# All commands can output JSON Lines files with the --jsonl (or -j) option.
# You can use this to pipe data into other tools, like jq, or back into
# the CLI for further processing.
#
# With this, you can (for example) get transcripts for a set of sessions:
#
#       ./cdp.py sessions -d 2023-08-01 --jsonl | ./cdp.py transcripts
#
# You can of course also summarize sessions, which is what we're really
# here for:
#
#       ./cdp.py sessions -d 2023-08-01 --jsonl | ./cdp.py summarize
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
from cdp_backend.database import models as cdp_models
from cdp_data import CDPInstances
from cdp_data.utils import connect_to_infrastructure
from fireo.models import Model
from fireo.queries.query_iterator import QueryIterator
from fireo.queries.query_wrapper import ReferenceDocLoader

# ------------------------------------------------------------
# Utility methods and types
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


class QIJSONEncoder(json.JSONEncoder):
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


def _log_qi_jsonl(qi: QueryIterator, output: t.TextIO):
    """Log a FireO QueryIterator as JSON lines."""
    for item in qi:
        assert isinstance(item, Model)
        print(json.dumps(item.to_dict(), cls=QIJSONEncoder), file=output)


def _log_model_human(item: Model, count: int, output: t.TextIO):
    print(f"{item.key} [{count}]", file=output)
    print(f"{'-' * len(item.key)}", file=output)
    for key, value in item.to_dict().items():
        if key == "key" or key == "id":
            continue
        if isinstance(value, datetime.datetime):
            value = value.strftime("%A, %B %d, %Y %I:%M %p")
        elif isinstance(value, datetime.date):
            value = value.strftime("%A, %B %d, %Y")
        elif isinstance(value, ReferenceDocLoader):
            value = f"{value.field.model_ref.collection_name}/{value.ref.id}"
        print(f"    {key}: {value}", file=output)


def _log_i_human(i: QueryIterator, output: t.TextIO):
    """Log a FireO QueryIterator as human readable text."""
    count = 0
    for row in qi:
        assert isinstance(row, Model)
        if count:
            print("", file=output)
        count += 1
        _log_model_human(row, count, output)
    print(f"\n{count} total rows.", file=output)


def _log_qi(qi: QueryIterator, jsonl: bool, output: t.TextIO):
    """Log a FireO QueryIterator as either JSON lines or human readable text."""
    if jsonl:
        _log_qi_jsonl(qi, output)
    else:
        _log_qi_human(qi, output)


def log_i(i: t.Iterable | QueryIterator, jsonl: bool, output: t.TextIO):
    """Log an arbitrary iterable as either JSON lines or human readable text."""
    if isinstance(i, QueryIterator):
        _log_qi(i, jsonl, output)
    #
    # CONSIDER: what if some interior object is a QueryIterator, or a
    # Model?
    #
    elif json:
        for item in i:
            print(json.dumps(item, cls=QIJSONEncoder), file=output)
    else:
        print("\n".join([str(item) for item in i]), file=output)


# ------------------------------------------------------------
# Click based CLI
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
        "-j",
        "--jsonl",
        is_flag=True,
        default=False,
        help="Output events as JSON lines instead of human readable text.",
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


@click.group()
def cdp():
    """A simple CDP data command line interface."""
    pass


@cdp.command()
@options_group(OUTPUT_OPTIONS)
def instances(output: t.TextIO, jsonl: bool):
    """
    Print a list of available CDP instances.

    These can be used with the --instance (-c) option on other commands.
    """
    for name in INSTANCE_NAMES:
        out = json.dumps({"name": name, "slug": INSTANCES[name]}) if jsonl else name
        print(out, file=output)


@cdp.command()
@options_group(STD_OPTIONS)
@click.option(
    "-d",
    "--start-date",
    type=click.DateTime(),
    default=datetime.datetime.now() - datetime.timedelta(days=7),
    help="The start date after which to get events from.",
)
def sessions(
    instance: str,
    start_date: datetime.datetime,
    jsonl: bool,
    output: t.TextIO,
    **kwargs,
):
    """Fetch sessions from a CDP instance."""
    _ = connect_to_infrastructure(INSTANCES[instance])
    sessions = cdp_models.Session.collection.filter(
        "session_datetime", ">=", start_date
    ).fetch()
    log_i(sessions, jsonl, output)


@cdp.command()
@options_group(STD_OPTIONS)
@click.option(
    "--session-id",
    type=str,
    required=False,
    multiple=True,
    help="The session ID(s) to get a transcript for.",
)
def transcripts(
    instance: str, session_id: list[str], input: t.TextIO, output: t.TextIO, jsonl: bool
):
    """Fetch transcripts from a CDP instance."""
    _ = connect_to_infrastructure(INSTANCES[instance])
    if not session_id:
        session_id = [json.loads(line)["id"] for line in input.readlines()]
    connect_to_infrastructure(INSTANCES[instance])
    for sid in session_id:
        session_key = f"session/{sid}"
        transcript = cdp_models.Transcript.collection.filter(
            session_ref=session_key
        ).get()
        if transcript is None:
            print(f"No transcript found for session {sid}.", file=output)
            continue
        print(f"Transcript for session {sid}: {transcript}", file=output)


if __name__ == "__main__":
    cdp()
