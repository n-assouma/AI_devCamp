"""Local demo and optional Gemini provider for the chat tutor."""

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeGuard

from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from google.genai import types


ExplanationStyle = Literal["steps", "analogy", "example"]
STYLES: tuple[ExplanationStyle, ...] = ("steps", "analogy", "example")

# Free-tier Gemini quota is per-day AND per-model, so an exhausted model is
# dead for ~17 hours while its neighbours are untouched. Ordered closest-first
# so quality degrades gradually. Mirrors tutor_agent.config.MODEL_FALLBACKS.
MODEL_FALLBACKS: tuple[str, ...] = (
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
)

_dead_models: set[str] = set()


def _is_daily_quota(error: APIError) -> bool:
    """True for the per-day quota only, not a per-minute burst 429."""
    if error.code != 429:
        return False
    blob = str(error).lower()
    return "per day" in blob or "perday" in blob or "free_tier" in blob


def _next_live_model(current: str) -> str | None:
    """Next model not yet known to be exhausted, or None when all are."""
    _dead_models.add(current)
    for name in MODEL_FALLBACKS:
        if name not in _dead_models:
            return name
    return None


class TutorResponse(TypedDict):
    answers: dict[str, str]
    is_definition: bool
    check_question: str | None


TutorProvider = Callable[
    [str, str, ExplanationStyle | None],
    Mapping[str, object],
]


class TutorProviderError(RuntimeError):
    """Raised when a configured tutor provider cannot return a valid response."""


def is_style(value: object) -> TypeGuard[ExplanationStyle]:
    return value in STYLES


def validate_tutor_response(value: Mapping[str, object]) -> TutorResponse:
    answers = value.get("answers")
    is_definition = value.get("is_definition")
    check_question = value.get("check_question")
    if not isinstance(answers, Mapping):
        raise ValueError("Tutor response 'answers' must be an object.")
    if not isinstance(is_definition, bool):
        raise ValueError("Tutor response 'is_definition' must be a boolean.")
    if check_question is not None and not isinstance(check_question, str):
        raise ValueError("Tutor response 'check_question' must be a string or null.")
    if not is_definition and not check_question:
        raise ValueError("Non-definition tutor responses need a check question.")
    validated_answers: dict[str, str] = {}
    for style in STYLES:
        answer = answers.get(style)
        if answer is not None:
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Tutor response answer for {style!r} must be non-empty.")
            validated_answers[style] = answer.strip()
    if not validated_answers:
        raise ValueError("Tutor response must include at least one explanation.")
    if is_definition and check_question:
        check_question = None
    return {
        "answers": validated_answers,
        "is_definition": is_definition,
        "check_question": check_question,
    }


class DemoTutor:
    """Offline UI provider. Its generated text is illustrative, not model output."""

    def __call__(
        self,
        prompt: str,
        profile_notes: str,
        style: ExplanationStyle | None = None,
    ) -> TutorResponse:
        del profile_notes
        is_definition = bool(
            re.search(
                r"\b(?:define\b|what\s+(?:is|are)\b|what\s+does\b.*\bmean\b|meaning\s+of\b)",
                prompt,
                re.IGNORECASE,
            )
        )
        answers = {
            "steps": (
                f"Let's work through **{prompt}** step by step.\n\n"
                "1. Identify what the question is asking you to find.\n"
                "2. Write down the relevant facts, rule, or equation.\n"
                "3. Apply one operation at a time, showing how each step follows.\n"
                "4. Check that the result answers the original question."
            ),
            "analogy": (
                f"Think of **{prompt}** as a puzzle. The question gives you some "
                "clues and asks you to find a missing piece. Keep the clues balanced: "
                "whenever you change one side of an equation, make the same change "
                "to the other side. Then check that your piece fits."
            ),
            "example": (
                f"**A way to approach a question like “{prompt}”:**\n\n"
                "Write what you know, choose the rule that connects it to what you "
                "need, and apply that rule in small steps. For an equation, undo "
                "operations in reverse order and substitute your answer back to check."
            ),
        }
        selected_answers = {style: answers[style]} if style else answers
        check_question = (
            None
            if is_definition
            else f"What is the first step or key idea you would use for: {prompt}?"
        )
        return validate_tutor_response(
            {
                "answers": selected_answers,
                "is_definition": is_definition,
                "check_question": check_question,
            }
        )


class GeminiTutor:
    """Gemini-backed tutor using GOOGLE_API_KEY from the runtime environment."""

    def __init__(
        self,
        env_file: Path,
        api_key: str | None = None,
        model: str | None = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        load_dotenv(dotenv_path=env_file, override=False)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative.")
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.on_retry: Callable[[int, int, float], None] | None = None
        if not self.api_key:
            raise TutorProviderError(
                f"GOOGLE_API_KEY was not found in the process environment or {env_file}."
            )
        self.client = genai.Client(api_key=self.api_key)

    def __call__(
        self,
        prompt: str,
        profile_notes: str,
        style: ExplanationStyle | None = None,
    ) -> TutorResponse:
        style_instruction = (
            f"Only create the {style} explanation."
            if style
            else "Create all three explanation styles."
        )
        system_instruction = (
            "You are a patient educational tutor. Explain the student's question "
            "accurately and at an accessible level. Use the saved preference only "
            "to choose presentation, not to assume the student's ability. Return "
            "a JSON object with exactly these fields: answers (object with any "
            "requested keys from steps, analogy, example), is_definition (boolean), "
            "and check_question (string or null). For definitions, set "
            "is_definition=true and check_question=null. Otherwise write one short, "
            "relevant practice question. Do not include its answer in the "
            "student-facing explanation."
        )
        user_prompt = (
            f"Saved preference/profile: {profile_notes}\n"
            f"Requested format: {style or 'offer all three styles'}.\n"
            f"{style_instruction}\n"
            f"Student question: {prompt}"
        )
        attempt = 0
        while attempt < self.max_attempts:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.4,
                    ),
                )
                break
            except APIError as error:
                # Free-tier Gemini enforces a per-DAY request quota that is
                # scoped per model, and its retryDelay is ~17 hours. Backing
                # off is pointless; the next model has its own full bucket.
                # Without this the tutor dies on stage with "429
                # RESOURCE_EXHAUSTED after 3 attempts" while five other models
                # sit idle. tutor_agent/runner.py does the same thing.
                if _is_daily_quota(error):
                    nxt = _next_live_model(self.model)
                    if nxt is not None:
                        self.model = nxt
                        continue
                    raise TutorProviderError(
                        "Every fallback model has exhausted its free-tier "
                        "daily quota. Switch the sidebar's tutor provider to "
                        "Demo to keep going offline, or enable billing on "
                        "the Google Cloud project that owns this API key."
                    ) from error
                attempt += 1
                retryable = error.code in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt >= self.max_attempts:
                    status = f"{error.code} {error.status or ''}".strip()
                    attempts = (
                        f" after {attempt} attempts"
                        if retryable and attempt > 1
                        else ""
                    )
                    raise TutorProviderError(
                        f"Gemini API request failed ({status}){attempts}. Please try again."
                    ) from error
                delay = min(self.retry_delay_seconds * (2 ** (attempt - 1)), 8.0)
                if self.on_retry:
                    self.on_retry(attempt, self.max_attempts, delay)
                time.sleep(delay)
        if not response.text:
            raise TutorProviderError("Gemini returned an empty response.")
        try:
            payload: Any = json.loads(response.text)
            return validate_tutor_response(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise TutorProviderError(f"Gemini returned an invalid tutor response: {error}") from error
