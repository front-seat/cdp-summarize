# CDP-Summarize

A set of very early stage code to summarize CDP data using OpenAI's GPT-3.5-Turbo model.

This documentation is early stage, too. :-)

## Setting up

### Your dev machine

Install Python 3.11.x.

Then install dependencies and dev dependencies:

```console
> python3 -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
> pip install -r requirements-dev.txt
```

Use `pipenv shell` to activate the environment.

After that, you'll mostly use the `./cdp.py` command-line tool to do things.

### OpenAI

Get an OpenAI API key at https://platform.openai.com/.

Set the `OPENAI_API_KEY` environment variable before running `./cdp.py summarize ...`

## Example summary output

Summary output is JSON and follows a simple data structure primarily defined by the `@dataclass`es in [`summarize/summaries.py`](./summarize/summaries.py).

Every item is summarized with both a short `headline` and a paragraph-length `detail`. The returned data structure contains summary roll-ups at every level (aka an `Event` has a rolled-up summary of all its matters and sessions, etc.)

See an [example summary output](./example-summary.json) for details.

## `cdp.py` command-line

This is the main command-line tool for this project.

You can run `./cdp.py --help` to see the available subcommands.

### Getting available CDP instances

You'll probably want to start by running `./cdp.py list` to see the available CDP instance names:

```console
> ./cdp.py instances
Alameda
Albuquerque
Atlanta
Boston
Charlotte
Denver
...
```

All `./cdp.py` sub-commands accept the `--jsonl` (`-j`) variant to output JSON lines instead of something slightly more human-readable:

```console
> ./cdp.py instances --jsonl
{"name": "Alameda", "slug": "cdp-alameda-d3dabe54"}
{"name": "Albuquerque", "slug": "cdp-albuquerque-1d29496e"}
{"name": "Atlanta", "slug": "cdp-atlanta-37e7dd70"}
{"name": "Boston", "slug": "cdp-boston-c384047b"}
{"name": "Charlotte", "slug": "cdp-charlotte-98a7c348"}
```

### Listing events

You can list events for a CDP instance with `./cdp.py events list -i <instance-name>`.

For example:

```console
> ./cdp.py events list -i Seattle -s 2023-08-01
f5beff18eb58 Thu Aug 10, 2023 @ 9:30AM
ad02ed9b06e5 Tue Aug 15, 2023 @ 9:30AM
5e0e283af0ac Tue Aug 15, 2023 @ 2:00PM
cdad9f6bbb40 Tue Sep 5, 2023 @ 9:30AM
15f48b4baa68 Mon Sep 11, 2023 @ 9:30AM
bc316138545b Tue Sep 12, 2023 @ 9:30AM
```

The `--start-date` (`-s`) and `--end-date` (`-e`) filters are available but both optional.

### Expanding an event

You can expand an event to all relevant data for summarization, including associated sessions, most recent transcripts, matters, and matter files, with:

```console
> ./cdp.py events expand -i Seattle --id bc316138545b
[
  {
    "event": {
      "body_ref": "body/bd2e47dde7d3",
      "event_datetime": "2023-09-12T16:30:00+00:00",
      "static_thumbnail_ref": "file/448020369c3c",
      "hover_thumbnail_ref": "file/94f02b4f66ad",
 ...
```

You can use `--id` multiple times, or can also provide `--start-date` or `--end-date` to expand all events in a date range.

### Summarizing an event

This is the reason we're here! Much like the `expand` subcommand, you can summarize an event with:

```console
> ./cdp.py events summarize -i Seattle --id bc316138545b
...
```

Summarization can take a while. You can turn on verbose output (to `stderr`) with `--verbose`.

```
> ./cdp.py events summarize -i Seattle --id bc316138545b --verbose > /tmp/summary.json
```

The `--id` and `--start-date`/`--end-date` filters are available here, too.
