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

After that, you'll mostly use the `./cdp.py` command-line tool to do things.

### Choosing an LLM model

We currently support:

1. [OpenAI models](https://platform.openai.com/docs/models)
2. [Hugging Face Endpoints](https://huggingface.co/inference-endpoints) that support `text-generation`

By default, the `./cdp.py` tool will use OpenAI's [`gpt-3.5-turbo`](https://platform.openai.com/docs/models/gpt-3-5), a good price/performance model for summarization tasks.

You can explicitly choose a different OpenAI model with the `--openai` option. For instance, if you'd like to spend a lot of money, try `--openai gpt-4`.

If you're using OpenAI, you'll need to set the `OPENAI_API_KEY` environment variable before running `./cdp.py events summarize ...`. You can [get an OpenAI key from their platform site](https://platform.openai.com/).

If you'd like to use Hugging Face instead, you'll need to set the `HUGGINGFACEHUB_API_TOKEN` environment variable and pass the `--huggingface <endpoint_url>` parameter to `./cdp.py`.

### Altering the LLM prompts

The `./cdp.py` tool ships with a default set of prompt templates to generate summaries. These are found in [`summarize/prompts.json`](./summarize/prompts.json).

The default prompts are tuned primarily for use with GPT-3.5-Turbo. If you're using a different model &mdash; for instance, an open model like [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) or [Llama2](https://huggingface.co/meta-llama) &mdash; you may want to tune the prompts. Simply create a new `custom-prompts.json` file wherever you like, and pass the `--prompts <path_to_file>` parameter to `./cdp.py events summarize ...`.

### Caching intermediate results

Summarization is a multi-step process. Complex events with long transcripts or large numbers of matters and matter files can take a while to summarize. To speed things up, you can cache intermediate results to the filesystem. Provide the `--cachedir <path_to_directory>` parameter to `./cdp.py events summarize ...` to specify a location to store cached summaries. If you need to stop a summarization and resume it later, simply re-run the same command with the same `--cachedir` parameter. You'll be glad you did!

It is safe to use the same cache directory for multiple CDP instances and multiple events. The tool does not evict the cache, and you probably want it somewhere stable on your filesystem anyway.

### Cost estimation

Use the `--verbose` flag to output stats at the end of summarization.

The total wall clock time for summarization is always provided.

When using OpenAI, additional stats are provided. These include the number of prompt and completion tokens used, the number of LLM requests made, and the estimated cost of the summarization based on OpenAI's current rate sheet.

Direct estimation of costs for Hugging Face endpoints is not yet supported. OpenAI charges by the token; Hugging Face endpoints charge by uptime and GPU selection. To get a rough estimate of costs, simply use the wall clock time. (Going one level deeper: the specific choice of GPU and model matter a _lot_. What you really want to know is prompt and completion token consumption and generation speed for your given model and hardware.)

The four [example event summaries](./examples/) contained in this repository cost a total of $TODO to generate using `gpt-3.5-turbo`, for an average of $TODO per summary. (The most expensive summary, TODO, cost $TODO; the least expensive summary, TODO, cost $TODO.)

## Example summary output

Summary output is JSON and follows a simple data structure primarily defined by the `@dataclass`es in [`summarize/summaries.py`](./summarize/summaries.py).

Every item is summarized with both a short `headline` and a paragraph-length `detail`. The returned data structure contains summary roll-ups at every level (aka an `Event` has a rolled-up summary of all its `Matter`s and `Session`s, etc.)

Example outputs for city council events in Seattle, Boston, Oakland, and Milwaukee are found in [the `./examples` directory](./examples/). Give them a look!

Seattle bc316138545b:

Summarized 1 events in 2,532.40 seconds.
OpenAI LLM Stats:
total_tokens: 129,868
prompt_tokens: 106,502
completion_tokens: 23,366
successful_requests: 221
total_cost_usd: $0.21

Boston 319e357ca015:

Oakland d2305d5903fc:

Milwaukee 1c696af4b860:

## `cdp.py` command-line

This is the main command-line tool for this project.

You can run `./cdp.py --help` to see the available subcommands.

### Getting available CDP instances

You'll probably want to start by running `./cdp.py instances` to see the available CDP instance names:

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
