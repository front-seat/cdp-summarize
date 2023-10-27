import json
import pathlib
import typing as t
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadlineDetailPromptTemplates:
    """A tandem of prompts necessary to summarize text into a headline and detail."""

    headline: str
    detail: str

    def to_dict(self) -> dict:
        """Return a dictionary representation of this object."""
        return {
            "headline": self.headline,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HeadlineDetailPromptTemplates":
        """Return a new instance from a dictionary representation."""
        return cls(
            headline=d["headline"],
            detail=d["detail"],
        )


@dataclass(frozen=True)
class PromptTemplates:
    """A collection of prompts necessary to summarize CDP data."""

    DEFAULT_PATH: t.ClassVar[pathlib.Path] = (
        pathlib.Path(__file__).parent / "prompts.json"
    )

    concise: HeadlineDetailPromptTemplates
    matter: HeadlineDetailPromptTemplates
    transcript: HeadlineDetailPromptTemplates
    event: HeadlineDetailPromptTemplates

    def to_dict(self) -> dict:
        """Return a dictionary representation of this object."""
        return {
            "concise": self.concise.to_dict(),
            "matter": self.matter.to_dict(),
            "transcript": self.transcript.to_dict(),
            "event": self.event.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptTemplates":
        """Return a new instance from a dictionary representation."""
        return cls(
            concise=HeadlineDetailPromptTemplates.from_dict(d["concise"]),
            matter=HeadlineDetailPromptTemplates.from_dict(d["matter"]),
            transcript=HeadlineDetailPromptTemplates.from_dict(d["transcript"]),
            event=HeadlineDetailPromptTemplates.from_dict(d["event"]),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PromptTemplates":
        """Return a new instance from a JSON string."""
        return cls.from_dict(t.cast(dict, json.loads(json_str)))

    @classmethod
    def from_file(cls, path: pathlib.Path) -> "PromptTemplates":
        """Return a new instance from a JSON file."""
        with open(path.resolve()) as f:
            return cls.from_json(f.read())

    @classmethod
    def default(cls) -> "PromptTemplates":
        """Return a new instance from the default JSON file."""
        return cls.from_file(cls.DEFAULT_PATH)
