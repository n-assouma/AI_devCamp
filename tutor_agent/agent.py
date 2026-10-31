"""The agents.

Three LlmAgents, deliberately kept separate:

===========  ==========================================  ================
agent        sees                                        runs
===========  ==========================================  ================
explainer    a question + one style name                 once, offline
predictor    profile notes + question text, NOTHING ELSE every round
updater      old profile + prediction + actual outcome   only on a miss
===========  ==========================================  ================

They are separate because their *information sets* must be different. If one
agent did all three jobs it would, within a single context window, know the
student's real answer at the moment it made its prediction -- and the headline
metric of this project (blind prediction accuracy) would be fraudulent.
Separation is not a code-organisation choice here; it is the experiment's
control.

``root_agent`` exists so that ``adk web`` can show judges the live reasoning
trace. It owns its own freshly-built copies of the three specialists, because
ADK assigns ``parent_agent`` on attachment and we do not want the
programmatic agents used by :mod:`tutor_agent.api` to be owned by anything.
"""

from __future__ import annotations

from typing import Literal

from google.adk.agents import LlmAgent

from . import prompts
from .config import MODEL_NAME, STYLES
from .schemas import Explanation, ProfileNotes, prediction_schema

__all__ = [
    "build_explainer",
    "build_predictor",
    "build_updater",
    "explainer_agent",
    "predictor_agent",
    "updater_agent",
    "root_agent",
]


def build_explainer(
    styles: list[str] | None = None,
    *,
    name: str = "explainer",
    include_contents: Literal["default", "none"] = "none",
) -> LlmAgent:
    """Agent that writes ONE explanation of one question in one style.

    Called once per (question, style) pair and cached to questions.json, so
    the live demo never waits on it.
    """
    return LlmAgent(
        name=name,
        model=MODEL_NAME,
        description=(
            "Writes one short GCSE maths explanation of a question in a "
            "named style (" + ", ".join(styles or STYLES) + ")."
        ),
        instruction=prompts.explainer_instruction(styles),
        output_schema=Explanation,
        output_key="explanation",
        include_contents=include_contents,
    )


def build_predictor(
    styles: list[str] | None = None,
    *,
    name: str = "predictor",
    include_contents: Literal["default", "none"] = "none",
) -> LlmAgent:
    """Agent that makes the sealed, blind forecast.

    ``output_schema`` is generated from the style list, so ``pick`` is
    constrained by the model's own decoder to a configured style.

    ``include_contents="none"`` matters: it guarantees the predictor sees
    only the single message we hand it, never a conversation history that
    might contain an earlier round's actual outcome. Blindness is enforced
    here, in the agent definition, as well as in ``api.predict``'s signature.

    This is still safe under ``adk web``: verified against ADK 2.13.0, a
    sub-agent reached via ``transfer_to_agent`` with ``include_contents="none"``
    anchors its turn on the real user input rather than on the parent's
    transfer event, so it does receive the judge's question.
    """
    return LlmAgent(
        name=name,
        model=MODEL_NAME,
        description=(
            "Given only a student profile and a question, predicts which "
            "explanation the student will pick and whether they will then "
            "answer a check question correctly. Never sees the real outcome."
        ),
        instruction=prompts.predictor_instruction(styles),
        output_schema=prediction_schema(styles),
        output_key="prediction",
        include_contents=include_contents,
    )


def build_updater(
    styles: list[str] | None = None,
    *,
    name: str = "updater",
    include_contents: Literal["default", "none"] = "none",
) -> LlmAgent:
    """Agent that rewrites the profile after a failed prediction."""
    return LlmAgent(
        name=name,
        model=MODEL_NAME,
        description=(
            "Rewrites the student profile as short general rules after the "
            "profile's prediction was contradicted, keeping what the student "
            "prefers separate from what actually helps them."
        ),
        instruction=prompts.updater_instruction(styles),
        output_schema=ProfileNotes,
        output_key="profile_notes",
        include_contents=include_contents,
    )


# --------------------------------------------------------------------------
# The instances used programmatically by tutor_agent.api
# --------------------------------------------------------------------------

explainer_agent: LlmAgent = build_explainer()
predictor_agent: LlmAgent = build_predictor()
updater_agent: LlmAgent = build_updater()


# --------------------------------------------------------------------------
# The instance `adk web` discovers (module attribute literally named
# `root_agent`). Its sub-agents are separate objects from the three above.
# --------------------------------------------------------------------------

root_agent: LlmAgent = LlmAgent(
    name="adaptive_tutor",
    model=MODEL_NAME,
    description=(
        "Adaptive GCSE maths tutor that keeps a written profile of each "
        "student, predicts their behaviour before they act, and rewrites the "
        "profile when it is wrong."
    ),
    instruction=prompts.ROOT_INSTRUCTION,
    sub_agents=[
        build_explainer(name="explainer"),
        build_predictor(name="predictor"),
        build_updater(name="updater"),
    ],
)
