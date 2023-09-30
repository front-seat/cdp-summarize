# llm.py :: lower level code that interacts with langchain to summarize

import os
import typing as t
from dataclasses import dataclass

from langchain.base_language import BaseLanguageModel
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)


class SummarizationError(Exception):
    """Exception raised when summary fails."""

    pass


@dataclass(frozen=True)
class SummarizationResult:
    headline: str
    detail: str
    text: str


def _make_langchain_prompt(
    format_str: str,
    context: dict[str, t.Any] | None = None,
    input_variables: tuple[str] = ("text",),
) -> PromptTemplate:
    """
    Given a python-style format string, render it into a final prompt string.
    From there, wrap it in a LangChain PromptTemplate instance.
    """
    # Deal with missing context.
    context = context or {}
    # Deal with the annoying overlap between langchain and python format
    # strings.
    context["text"] = "{text}"
    rendered_prompt = format_str.format(**(context or {}))
    return PromptTemplate(
        template=rendered_prompt, input_variables=list(input_variables)
    )


def _attempt_to_split_text(text: str, chunk_size: int) -> list[str]:
    """
    Attempt to split text into chunks of at most `chunk_size`.
    """
    # We try to split the text in a few different ways. If we're lucky, one
    # of them will yield chunks entirely of size <= `chunk_size`.
    SEPARATORS = ["\n\n", "\n", ". "]

    texts = []
    for separator in SEPARATORS:
        text_splitter = CharacterTextSplitter(separator, chunk_size=chunk_size)
        texts = text_splitter.split_text(text)
        if all(len(text) <= chunk_size for text in texts):
            return texts

    # If we get here, we couldn't find a separator that worked. Just filter
    # out the long chunks.
    filtered_texts = [text for text in texts if len(text) <= chunk_size]
    if not filtered_texts:
        raise SummarizationError("Could not split text.")
    return filtered_texts


def _summarize_langchain_llm(
    text: str,
    llm: BaseLanguageModel,
    map_template: str,
    detail_combine_template: str,
    headline_combine_template: str,
    context: dict[str, t.Any] | None = None,
    chunk_size: int = 3584,
) -> SummarizationResult:
    """
    Summarize text using langchain.

    We want to produce both a "headline" summary and a "detail" summary,
    so we do a little extra work to re-use intermediate steps so as to keep
    LLM costs low.
    """
    # Fail if the text is empty.
    if not text.strip():
        raise SummarizationError("Text was empty.")

    # Attempt to split our text into chunks of at most `chunk_size`.
    # We use LangChain's `CharacterTextSplitter` for this; it can fail -- for
    # instance, if the input text was poorly extracted, it may be effectively
    # un-splittable by any sensible algorithm. If that happens, we'll raise
    # a SummarizationError.
    texts = _attempt_to_split_text(text, chunk_size)

    # LangChain documents are tuples of text and arbitrary metadata;
    # we don't use the metadata. It defaults to an empty dict.
    detail_documents = [Document(page_content=text) for text in texts]

    # Build LangChain-style PromptTemplates.
    map_prompt = _make_langchain_prompt(map_template, context)
    detail_combine_prompt = _make_langchain_prompt(detail_combine_template, context)
    headline_combine_prompt = _make_langchain_prompt(headline_combine_template, context)

    # Build a LangChain summarization chain. This one will produce the "detail"
    # summary.
    detail_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=detail_combine_prompt,
        return_intermediate_steps=True,
    )
    # Our hack below depends on this being a MapReduceDocumentsChain.
    assert isinstance(detail_chain, MapReduceDocumentsChain)

    # Run the chain.
    detail_outputs = detail_chain(detail_documents)

    # Make sure the expected output keys are available.
    # (We used the `return_intermediate_steps=True` option above, so we expect
    # to see the `intermediate_steps` key.)
    if (
        "output_text" not in detail_outputs
        or "intermediate_steps" not in detail_outputs
    ):
        raise SummarizationError("Missing expected keys from detail_outputs.")

    # Great! We now have the "detail" summary, and the intermediate chunk
    # summaries. We're going to re-use the chunk summaries to generate the
    # "headline" summary.
    detail = detail_outputs["output_text"]
    chunk_summaries = detail_outputs["intermediate_steps"]
    assert len(chunk_summaries) == len(detail_documents)

    # Now let's generate a "headline" summary by re-using the chunk summaries.
    headline_documents = [Document(page_content=text) for text in chunk_summaries]
    headline_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=headline_combine_prompt,
    )
    headline_outputs = headline_chain(headline_documents)
    if "output_text" not in headline_outputs:
        raise SummarizationError("Missing expected key from headline_outputs.")

    headline = headline_outputs["output_text"]

    # We did it!
    return SummarizationResult(headline=headline, detail=detail, text=text)


def summarize_openai(
    text: str,
    detail_combine_template: str,
    headline_combine_template: str,
    context: dict[str, t.Any] | None = None,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.4,
    chunk_size: int = 3584,
) -> SummarizationResult:
    """Summarize text using langchain and OpenAI. Low-level."""
    if OPENAI_API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    llm = ChatOpenAI(
        temperature=temperature,
        model_name=model_name,  # type: ignore -- busted type hints, alas
        openai_api_key=OPENAI_API_KEY,
        openai_organization=OPENAI_ORGANIZATION,
    )
    return _summarize_langchain_llm(
        text=text,
        llm=llm,
        # For now, always use detail templates when mapping.
        map_template=detail_combine_template,
        detail_combine_template=detail_combine_template,
        headline_combine_template=headline_combine_template,
        context=context,
        chunk_size=chunk_size,
    )
