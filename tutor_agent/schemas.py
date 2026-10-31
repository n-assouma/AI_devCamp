"""Pydantic output schemas for the three agents.

The interesting one is :func:`prediction_schema`. The predictor's ``pick``
field is *generated from* :data:`tutor_agent.config.STYLES` rather than
written out by hand, so the set of explanation styles stays configurable in
exactly one place. Gemini receives the enumerated styles as a JSON-schema
``enum``, which means an out-of-vocabulary pick is rejected by the model's
own constrained decoding -- we never have to post-validate it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from .config import MAX_PROFILE_WORDS, STYLES

__all__ = [
    "Explanation",
    "ProfileNotes",
    "prediction_schema",
    "PredictionBase",
    "Prediction",
    "STYLE_LITERAL",
]


# --------------------------------------------------------------------------
# Static schemas
# --------------------------------------------------------------------------


class Explanation(BaseModel):
    """One explanation of one question, in one style."""

    explanation: str = Field(
        description=(
            "The explanation itself. Plain prose or a short numbered list. "
            "No preamble, no sign-off, no markdown headings."
        )
    )


class ProfileNotes(BaseModel):
    """The rewritten student profile produced by the updater."""

    notes: str = Field(
        description=(
            "The complete rewritten profile, under "
            f"{MAX_PROFILE_WORDS} words. General rules about the student "
            "only -- never a transcript of what happened."
        )
    )


class PredictionBase(BaseModel):
    """Fields shared by every generated prediction schema.

    ``pick`` is deliberately absent here: it is added by
    :func:`prediction_schema`, typed against the live ``STYLES`` list.
    """

    correct_after_pick: bool = Field(
        description=(
            "Will this student answer the follow-up check question "
            "correctly after reading the explanation you predicted they "
            "would pick? true or false."
        )
    )
    reason: str = Field(
        description=(
            "One sentence, under 25 words, naming the single fact in the "
            "profile that drove this prediction."
        )
    )


# --------------------------------------------------------------------------
# Dynamic schema generation
# --------------------------------------------------------------------------


def _style_literal(styles: tuple[str, ...]) -> Any:
    """Build ``Literal["steps", "analogy", ...]`` from a runtime tuple.

    ``Literal[tuple_of_strings]`` is the documented way to parameterise
    Literal dynamically; it is what ``typing`` itself does internally when
    it flattens subscripts.
    """
    if not styles:
        raise ValueError("STYLES must contain at least one style")
    return Literal[styles]  # type: ignore[valid-type]


@lru_cache(maxsize=None)
def _prediction_schema_cached(styles: tuple[str, ...]) -> type[PredictionBase]:
    options = ", ".join(repr(s) for s in styles)
    return create_model(
        "Prediction",
        __base__=PredictionBase,
        pick=(
            _style_literal(styles),
            Field(
                description=(
                    "Which explanation this student will CHOOSE to read. "
                    f"Exactly one of: {options}."
                )
            ),
        ),
    )


def prediction_schema(styles: list[str] | tuple[str, ...] | None = None):
    """Return the predictor's output schema for ``styles``.

    Cached per distinct style tuple, so repeated calls hand back the same
    class (ADK compares schema identity when building the model request).

    >>> prediction_schema(["a", "b"])(pick="a", correct_after_pick=True,
    ...                              reason="because").pick
    'a'
    """
    resolved = tuple(STYLES if styles is None else styles)
    return _prediction_schema_cached(resolved)


#: The ``Literal`` for the configured styles, handy for type annotations.
STYLE_LITERAL = _style_literal(tuple(STYLES))

#: The prediction schema for the configured styles.
Prediction = prediction_schema()
