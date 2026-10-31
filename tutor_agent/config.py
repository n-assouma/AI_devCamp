"""Tunables for the adaptive tutor.

Everything that a judge might ask "could you change that?" about lives here.
In particular :data:`STYLES` is the single source of truth for the explanation
styles: the explainer loops over it, and the predictor's output schema is
*built from it* at import time (see :mod:`tutor_agent.schemas`). Adding a
fourth style is a one-line change to this file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Load the project .env once, at import time, so that both `adk web` and
# plain `python -c "import tutor_agent"` see the same credentials.
load_dotenv(PROJECT_ROOT / ".env")

API_KEY_ENV_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")

_MISSING_KEY_MESSAGE = (
    "No Gemini API key found.\n"
    "Put your key in the .env file at the project root:\n"
    "\n"
    "    GOOGLE_API_KEY=your_key_here\n"
    "\n"
    "Get a free one in ~30 seconds at https://aistudio.google.com/apikey"
)


class MissingAPIKeyError(RuntimeError):
    """Raised instead of letting google-genai fail deep in its stack."""


def require_api_key() -> str:
    """Return the configured Gemini API key, or raise one actionable error.

    Called before every model invocation. Without this, a missing key
    surfaces as an opaque traceback from inside google-genai's transport
    layer, which is the worst possible thing to debug five minutes before
    a demo.
    """
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
        # Vertex AI auth uses ADC, not an API key. Trust the caller.
        return ""
    for name in API_KEY_ENV_VARS:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    raise MissingAPIKeyError(_MISSING_KEY_MESSAGE)


def has_api_key() -> bool:
    """Non-raising probe, for UIs that want to show a friendly banner."""
    try:
        require_api_key()
    except MissingAPIKeyError:
        return False
    return True


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

MODEL_NAME: str = "gemini-flash-latest"

# Free-tier Gemini enforces a *per-day* request quota that is scoped PER MODEL
# (the 429 body names the model explicitly). A key that is exhausted on one
# model still has a full bucket on the next, so when a daily quota dies we roll
# to the next model rather than backing off -- the retry window is ~17 hours,
# which is longer than any hackathon.
#
# Order is "closest to MODEL_NAME first", so quality degrades gradually.
# On a billed project this list is almost never reached past the first entry.
MODEL_FALLBACKS: tuple[str, ...] = (
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
)

APP_NAME: str = "adaptive_tutor"
DEFAULT_USER_ID: str = "tutor_harness"

# --------------------------------------------------------------------------
# The styles. THE one list. Change this and nothing else.
# --------------------------------------------------------------------------

STYLES: list[str] = ["steps", "analogy", "example"]

#: Optional one-line briefs used to steer the explainer. A style with no
#: entry here still works -- it just gets a generic brief -- which is what
#: makes STYLES genuinely extensible.
STYLE_BRIEFS: dict[str, str] = {
    "steps": (
        "A numbered procedure. Short imperative steps that say what to do to "
        "the equation, in order. No motivation, no metaphor -- just the drill."
    ),
    "analogy": (
        "One concrete everyday metaphor (balance scales, undoing a parcel, a "
        "recipe) carried all the way through. Explain the idea, not the "
        "arithmetic."
    ),
    "example": (
        "A fully worked example of a DIFFERENT but structurally identical "
        "problem, solved line by line with the reasoning shown, so the "
        "student can pattern-match their own problem onto it."
    ),
    "visual": (
        "Describe a picture or diagram of the problem in words -- what is on "
        "each side, what moves where."
    ),
    "socratic": (
        "Three or four short leading questions that walk the student to the "
        "answer without ever stating it."
    ),
}


def style_brief(style: str) -> str:
    """Brief for ``style``, with a sane fallback for styles added later."""
    return STYLE_BRIEFS.get(
        style,
        f"An explanation in the '{style}' style, used consistently throughout.",
    )


# --------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------

#: Hard word cap on each generated explanation.
MAX_EXPLANATION_WORDS: int = 80

#: Hard word cap on the student profile. The cap is the point: a profile that
#: can hold everything learns nothing, because it never has to generalise.
MAX_PROFILE_WORDS: int = 80

#: How many times run_round() may ask the updater to rewrite the profile
#: before giving up and keeping the old one. See api.run_round.__doc__.
MAX_REPAIR_ATTEMPTS: int = 2

# --------------------------------------------------------------------------
# Rate limiting (free-tier Gemini)
# --------------------------------------------------------------------------

MAX_RETRIES: int = 5
BASE_BACKOFF_SECONDS: float = 2.0
MAX_BACKOFF_SECONDS: float = 45.0
BACKOFF_JITTER: float = 0.35

#: Seconds to wait on any single model call before giving up entirely.
CALL_TIMEOUT_SECONDS: float = 180.0

#: Retryable HTTP status codes: 429 quota, 500/502/503/504 transient.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

#: Substrings that mark a retryable failure when no status code is available.
RETRYABLE_MESSAGE_FRAGMENTS: tuple[str, ...] = (
    "429",
    "resource_exhausted",
    "resource exhausted",
    "quota",
    "rate limit",
    "rate_limit",
    "too many requests",
    "503",
    "unavailable",
    "overloaded",
    "deadline exceeded",
    "internal error",
)

DEFAULT_STUDENT: str = "maya"
