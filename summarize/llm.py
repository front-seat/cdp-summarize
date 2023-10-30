# llm.py :: lower level code that interacts with langchain to summarize

import abc
import typing as t
from dataclasses import dataclass

from langchain.base_language import BaseLanguageModel
from langchain.callbacks import OpenAICallbackHandler, get_openai_callback
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

from .prompts import HeadlineDetailPromptTemplates
from .timeout import configure_openai_api_timeouts


class SummarizationError(Exception):
    """Exception raised when summary fails."""

    pass


@dataclass(frozen=True)
class SummarizationResult:
    headline: str
    detail: str


def _make_langchain_prompt(
    format_str: str,
    context: dict[str, t.Any] | None = None,
    input_variables: tuple[str] = ("text",),
) -> PromptTemplate:
    """Given a python-style format string, render it into a final prompt string.
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


SEPARATORS = ["\n\n", "\n", ". "]


def _attempt_to_split_text(text: str, chunk_size: int) -> list[str]:
    """Attempt to split text into chunks of at most `chunk_size`."""
    # We try to split the text in a few different ways. If we're lucky, one
    # of them will yield chunks entirely of size <= `chunk_size`.

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
    hd_prompts: HeadlineDetailPromptTemplates,
    context: dict[str, t.Any] | None = None,
    chunk_size: int = 3584,
) -> SummarizationResult:
    """Summarize text using langchain.

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

    # Build LangChain-style PromptTemplates. For now, we always use the
    # "detail" templates for mapping.
    map_prompt = _make_langchain_prompt(hd_prompts.detail, context)
    detail_combine_prompt = _make_langchain_prompt(hd_prompts.detail, context)
    headline_combine_prompt = _make_langchain_prompt(hd_prompts.headline, context)

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
    return SummarizationResult(headline=headline.strip(), detail=detail.strip())


@dataclass(frozen=True)
class OpenAIServiceStats:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost_usd: float = 0.0

    def __add__(
        self, other: t.Union["OpenAIServiceStats", OpenAICallbackHandler]
    ) -> "OpenAIServiceStats":
        """Add two ServiceStats instances together."""
        if isinstance(other, OpenAICallbackHandler):
            other = OpenAIServiceStats.from_callback_handler(other)
        return OpenAIServiceStats(
            total_tokens=self.total_tokens + other.total_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            successful_requests=self.successful_requests + other.successful_requests,
            total_cost_usd=self.total_cost_usd + other.total_cost_usd,
        )

    @classmethod
    def from_callback_handler(
        cls, handler: OpenAICallbackHandler
    ) -> "OpenAIServiceStats":
        """Create a ServiceStats instance from an OpenAICallbackHandler."""
        return cls(
            total_tokens=handler.total_tokens,
            prompt_tokens=handler.prompt_tokens,
            completion_tokens=handler.completion_tokens,
            successful_requests=handler.successful_requests,
            total_cost_usd=handler.total_cost,
        )

    def __str__(self) -> str:
        """Return a string representation of this instance."""
        return (
            f"OpenAI LLM Stats:\n"
            f"total_tokens: {self.total_tokens:,}\n"
            f"prompt_tokens: {self.prompt_tokens:,}\n"
            f"completion_tokens: {self.completion_tokens:,}\n"
            f"successful_requests: {self.successful_requests:,}\n"
            f"total_cost_usd: ${self.total_cost_usd:.2f}\n"
        )


class LanguageModel(abc.ABC):
    """
    Abstract base class for a language models.

    Here, we try to hide the underlying mechanics of using LangChain and also
    make it possible to abstract away key details of the underlying model and
    provider itself.

    For our purposes today, there's only one thing we actually need our
    language model to do: provide a summarize() method that takes text and
    returns both a "headline" and a "detail" summary for the text. This is our
    "low level" summarization method; you need to call it with appropriate
    templates to get the desired results.
    """

    chunk_size: int

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def summarize(
        self,
        text: str,
        hd_prompts: HeadlineDetailPromptTemplates,
        context: dict[str, t.Any] | None = None,
    ) -> SummarizationResult:
        """Summarize text. Return a SummarizationResult."""
        ...


class OpenAILanguageModel(LanguageModel):
    """A language model via an OpenAI endpoint."""

    DEFAULT_CHUNK_SIZE = 3584

    model_name: str
    openai_api_key: str
    openai_organization: str | None
    temperature: float

    stats: OpenAIServiceStats

    def __init__(
        self,
        model_name: str | None,
        openai_api_key: str,
        openai_organization: str | None,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
    ):
        super().__init__(chunk_size=self.DEFAULT_CHUNK_SIZE)
        self.model_name = model_name or "gpt-3.5-turbo"
        self.openai_api_key = openai_api_key
        self.openai_organization = openai_organization
        self.temperature = 0.4
        self.stats = OpenAIServiceStats()
        configure_openai_api_timeouts(connect_timeout, read_timeout)

    def summarize(
        self,
        text: str,
        hd_prompts: HeadlineDetailPromptTemplates,
        context: dict[str, t.Any] | None = None,
    ) -> SummarizationResult:
        """Summarize text using an OpenAI API endpoint."""
        llm = ChatOpenAI(
            temperature=self.temperature,
            model=self.model_name,
            openai_api_key=self.openai_api_key,
            openai_organization=self.openai_organization,
        )
        # This is such an awkward use of a context manager on langchain's
        # part but... okay, sure. This is how we count tokens.
        with get_openai_callback() as openai_callback:
            result = _summarize_langchain_llm(
                text=text,
                llm=llm,
                hd_prompts=hd_prompts,
                context=context,
                chunk_size=self.chunk_size,
            )
            self.stats = self.stats + openai_callback
        return result


class HuggingFaceLanguageModel(LanguageModel):
    """A language model via a Huggingface endpoint."""

    DEFAULT_CHUNK_SIZE = 3584

    endpoint_url: str
    huggingfacehub_api_token: str

    def __init__(self, endpoint_url: str, huggingfacehub_api_token: str):
        super().__init__(chunk_size=self.DEFAULT_CHUNK_SIZE)
        self.endpoint_url = endpoint_url
        self.huggingfacehub_api_token = huggingfacehub_api_token

    def summarize(
        self,
        text: str,
        hd_prompts: HeadlineDetailPromptTemplates,
        context: dict[str, t.Any] | None = None,
    ) -> SummarizationResult:
        """Summarize text using a Huggingface endpoint."""
        llm = HuggingFaceEndpoint(
            endpoint_url=self.endpoint_url,
            huggingfacehub_api_token=self.huggingfacehub_api_token,
            task="text-generation",
        )
        return _summarize_langchain_llm(
            text=text,
            llm=llm,
            hd_prompts=hd_prompts,
            context=context,
            chunk_size=self.chunk_size,
        )
