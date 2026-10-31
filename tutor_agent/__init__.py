"""The tutor that learns the student.

An adaptive GCSE maths tutor built on Google ADK + Gemini. Three separate
agents with deliberately different information sets:

- ``explainer``  writes one short explanation per style, offline and cached.
- ``predictor``  makes a SEALED, BLIND forecast of what the student will do.
- ``updater``    rewrites the student profile when the forecast was wrong.

The thesis: what a student *prefers* is not what *teaches* them, and an agent
that keeps a short written profile can discover the gap.

Typical use::

    from tutor_agent import generate_explanations, predict, update, run_round

    explanations = generate_explanations("Solve 2x + 5 = 13")   # offline, cache it
    pred = predict(profile["notes"], question["question"])       # sealed
    log = run_round(1, profile, question, {"pick": "steps", "correct": False})
    profile["notes"] = log["profile_after"]

``adk web`` picks up :data:`root_agent` for the judge-facing reasoning trace.
"""

from __future__ import annotations

from .agent import (
    build_explainer,
    build_predictor,
    build_updater,
    explainer_agent,
    predictor_agent,
    root_agent,
    updater_agent,
)
from .api import (
    answers_match,
    clamp_words,
    generate_explanations,
    new_profile,
    normalise_answer,
    predict,
    run_round,
    choose_intervention,
    choose_presentation,
    recommend_style,
    style_evidence,
    update,
)
from .config import (
    MAX_EXPLANATION_WORDS,
    MAX_PROFILE_WORDS,
    MAX_REPAIR_ATTEMPTS,
    MODEL_NAME,
    STYLES,
    MissingAPIKeyError,
    has_api_key,
    require_api_key,
)
from .runner import AgentCallError, shutdown
from .schemas import Explanation, Prediction, ProfileNotes, prediction_schema

__version__ = "1.0.0"

__all__ = [
    # --- the contract other modules import -------------------------------
    "generate_explanations",
    "predict",
    "update",
    "run_round",
    "choose_intervention",
    "choose_presentation",
    "recommend_style",
    "style_evidence",
    "normalise_answer",
    # --- helpers ----------------------------------------------------------
    "answers_match",
    "clamp_words",
    "new_profile",
    # --- agents -----------------------------------------------------------
    "root_agent",
    "explainer_agent",
    "predictor_agent",
    "updater_agent",
    "build_explainer",
    "build_predictor",
    "build_updater",
    # --- config -----------------------------------------------------------
    "STYLES",
    "MODEL_NAME",
    "MAX_EXPLANATION_WORDS",
    "MAX_PROFILE_WORDS",
    "MAX_REPAIR_ATTEMPTS",
    "has_api_key",
    "require_api_key",
    # --- schemas ----------------------------------------------------------
    "Prediction",
    "Explanation",
    "ProfileNotes",
    "prediction_schema",
    # --- errors -----------------------------------------------------------
    "MissingAPIKeyError",
    "AgentCallError",
    # --- lifecycle --------------------------------------------------------
    "shutdown",
    "__version__",
]
