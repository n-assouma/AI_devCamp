"""ADK plumbing: one agent call in, one string out.

Two problems this module exists to solve.

**1. Sync callers, async ADK.** The public API
(:func:`tutor_agent.api.predict` and friends) is plain and synchronous
because a Streamlit UI and a simulation script should not have to care that
ADK is async. But ``asyncio.run`` cannot be used from a thread that already
has a running loop -- which is exactly the situation inside Streamlit and
Jupyter. So every coroutine here is submitted to ONE dedicated event loop
living in ONE daemon thread for the lifetime of the process. That is strictly
better than ``asyncio.run`` per call: the google-genai client holds an
``httpx.AsyncClient`` whose connection pool is bound to the loop that first
used it, so round-robining loops eventually raises "Event loop is closed"
somewhere deep in the transport. One loop, forever, no affinity bugs.

**2. Free-tier rate limits.** The full Maya simulation is ~140 model calls.
On the free tier that reliably trips 429s. :func:`call_agent_async` retries
429/5xx with exponential backoff plus jitter; non-retryable errors (a bad
API key, a malformed request) fail immediately rather than burning a minute
of backoff on something that will never succeed.

Every call gets a **fresh session**. Combined with ``include_contents="none"``
on the agents, that means no call can see any previous call's messages. For
the predictor this is load-bearing: a leaked history would let it see an
outcome it is supposed to be blind to.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import uuid
from concurrent.futures import Future
from typing import Any, Coroutine, TypeVar

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from .config import (
    MODEL_FALLBACKS,
    APP_NAME,
    BACKOFF_JITTER,
    BASE_BACKOFF_SECONDS,
    CALL_TIMEOUT_SECONDS,
    DEFAULT_USER_ID,
    MAX_BACKOFF_SECONDS,
    MAX_RETRIES,
    RETRYABLE_MESSAGE_FRAGMENTS,
    RETRYABLE_STATUS_CODES,
    require_api_key,
)

__all__ = [
    "AgentCallError",
    "call_agent",
    "call_agent_async",
    "call_agent_json",
    "run_sync",
    "is_retryable",
    "shutdown",
]

T = TypeVar("T")


class AgentCallError(RuntimeError):
    """An agent call failed in a way that is not worth retrying again."""


# ==========================================================================
# The one background event loop
# ==========================================================================


class _LoopThread:
    """A singleton daemon thread running one event loop forever."""

    _lock = threading.Lock()
    _instance: "_LoopThread | None" = None

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run, name="tutor-agent-loop", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @classmethod
    def get(cls) -> "_LoopThread":
        with cls._lock:
            if cls._instance is None or not cls._instance.thread.is_alive():
                cls._instance = cls()
            return cls._instance

    def submit(self, coro: Coroutine[Any, Any, T]) -> "Future[T]":
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


def run_sync(coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
    """Run ``coro`` to completion from synchronous code, safely.

    Works from a plain script, from Streamlit (which owns a running loop on
    the calling thread), and from a notebook. Does NOT work from inside the
    tutor's own background loop -- nothing in this package calls it from
    there, and async callers should await :func:`call_agent_async` instead.
    """
    return _LoopThread.get().submit(coro).result(
        timeout=CALL_TIMEOUT_SECONDS if timeout is None else timeout
    )


def shutdown() -> None:
    """Stop the background loop. Optional; the thread is a daemon."""
    with _LoopThread._lock:
        inst = _LoopThread._instance
        _LoopThread._instance = None
    if inst is not None:
        inst.stop()


# ==========================================================================
# Retry / backoff
# ==========================================================================


def _status_code(exc: BaseException) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_retryable(exc: BaseException) -> bool:
    """True for quota (429) and transient server errors (5xx, overloaded).

    Checks the structured status code first and only falls back to string
    sniffing, because google-genai sometimes wraps the real error in a
    generic exception on the streaming path.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    code = _status_code(exc)
    if code is not None:
        return code in RETRYABLE_STATUS_CODES
    blob = f"{type(exc).__name__}: {exc}".lower()
    return any(frag in blob for frag in RETRYABLE_MESSAGE_FRAGMENTS)


logger = logging.getLogger(__name__)

_dead_models: set[str] = set()
_model_lock = threading.Lock()


def is_daily_quota_exhausted(exc: BaseException) -> bool:
    """True only for the per-day free-tier quota, not a per-minute 429.

    The two are both RESOURCE_EXHAUSTED but need opposite responses: a
    per-minute burst is worth sleeping on, a per-day exhaustion is not
    (the retryDelay is ~17 hours).
    """
    blob = f"{type(exc).__name__}: {exc}"
    if "RESOURCE_EXHAUSTED" not in blob and _status_code(exc) != 429:
        return False
    low = blob.lower()
    return "per day" in low or "perday" in low or "free_tier" in low


def _next_live_model(current: str) -> str | None:
    """The next model in MODEL_FALLBACKS not yet known to be exhausted."""
    with _model_lock:
        _dead_models.add(current)
        for name in MODEL_FALLBACKS:
            if name not in _dead_models:
                return name
    return None


def _retarget_model(name: str) -> None:
    """Point every agent we have already built at ``name``.

    All three agents draw on the same per-model quota pool, so when one dies
    they all have to move together -- otherwise the next agent rediscovers
    the dead quota and burns another failed call finding out.
    """
    with _runners_lock:
        agents = [r.agent for r in _runners.values()]
    for ag in agents:
        try:
            ag.model = name
        except Exception:  # pragma: no cover - pydantic validation guard
            logger.exception("could not retarget %s", getattr(ag, "name", ag))


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with full-ish jitter. ``attempt`` is 0-based."""
    base = min(BASE_BACKOFF_SECONDS * (2**attempt), MAX_BACKOFF_SECONDS)
    jitter = base * BACKOFF_JITTER
    return max(0.1, base + random.uniform(-jitter, jitter))


# ==========================================================================
# The call
# ==========================================================================

_runners: dict[int, InMemoryRunner] = {}
_runners_lock = threading.Lock()


def _runner_for(agent: LlmAgent) -> InMemoryRunner:
    """One runner per agent instance, reused across calls.

    Reuse is safe because every call creates its own session; what is shared
    is only the (stateless) service objects and the model client. Creating a
    runner per call would also create a genai client per call, which leaks
    sockets over 140 calls.
    """
    key = id(agent)
    with _runners_lock:
        runner = _runners.get(key)
        if runner is None:
            runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
            _runners[key] = runner
        return runner


def _final_text(event: Any) -> str:
    """Extract the model's user-facing text from an ADK event.

    ``part.thought`` parts are the model's internal reasoning and must be
    excluded, or they end up concatenated in front of the JSON payload.
    """
    content = getattr(event, "content", None)
    if content is None or not getattr(content, "parts", None):
        return ""
    return "".join(
        part.text
        for part in content.parts
        if getattr(part, "text", None) and not getattr(part, "thought", False)
    )


async def _invoke_once(agent: LlmAgent, message: str, *, user_id: str) -> str:
    runner = _runner_for(agent)
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=f"{agent.name}-{uuid.uuid4().hex}",
    )
    content = types.Content(role="user", parts=[types.Part(text=message)])

    chunks: list[str] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if getattr(event, "error_message", None) and not event.get_function_calls():
            raise AgentCallError(
                f"{agent.name}: model error "
                f"[{getattr(event, 'error_code', '?')}] {event.error_message}"
            )
        if event.author == agent.name and event.is_final_response():
            text = _final_text(event)
            if text:
                chunks.append(text)

    result = "".join(chunks).strip()
    if not result:
        raise AgentCallError(
            f"{agent.name}: model returned no text. This is usually a safety "
            "block or an empty candidate; retrying the call normally fixes it."
        )
    return result


async def call_agent_async(
    agent: LlmAgent,
    message: str,
    *,
    user_id: str = DEFAULT_USER_ID,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Invoke ``agent`` with one message; return its final text.

    Retries quota and transient errors with exponential backoff + jitter.
    """
    require_api_key()  # one clear error instead of a genai traceback

    last: BaseException | None = None
    attempt = 0
    while attempt < max_retries:
        try:
            return await _invoke_once(agent, message, user_id=user_id)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            last = exc

            # A dead *daily* quota never recovers inside a hackathon, but it
            # is scoped per model -- so switch models and retry immediately
            # instead of sleeping. Deliberately does NOT advance ``attempt``:
            # this is a first try on a different model, not a retry of a
            # model we already know is dead. The loop still terminates
            # because MODEL_FALLBACKS is finite and each switch marks one
            # more model dead, so _next_live_model eventually returns None.
            if is_daily_quota_exhausted(exc):
                nxt = _next_live_model(str(agent.model))
                if nxt is not None:
                    logger.warning(
                        "%s: daily quota exhausted on %s, switching to %s",
                        agent.name, agent.model, nxt,
                    )
                    _retarget_model(nxt)
                    continue
                raise AgentCallError(
                    f"{agent.name}: every model in MODEL_FALLBACKS has "
                    "exhausted its free-tier daily quota. Enable billing on "
                    "the Google Cloud project that owns this API key, or use "
                    "a key from a billed project.\n"
                    f"Original error: {exc}"
                ) from exc

            attempt += 1
            if not is_retryable(exc) or attempt >= max_retries:
                break
            await asyncio.sleep(_backoff_delay(attempt - 1))

    assert last is not None
    raise AgentCallError(
        f"{agent.name} failed after {max_retries} attempt(s): "
        f"{type(last).__name__}: {last}"
    ) from last


def call_agent(
    agent: LlmAgent,
    message: str,
    *,
    user_id: str = DEFAULT_USER_ID,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Synchronous :func:`call_agent_async`. Safe inside Streamlit."""
    return run_sync(
        call_agent_async(
            agent, message, user_id=user_id, max_retries=max_retries
        )
    )


def call_agent_json(
    agent: LlmAgent,
    message: str,
    *,
    user_id: str = DEFAULT_USER_ID,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Like :func:`call_agent`, but parse the response as a JSON object.

    Agents here all set ``output_schema``, so Gemini returns strict JSON.
    The defensive brace-slicing is for the rare case where a model prefixes
    a stray token despite constrained decoding.
    """
    raw = call_agent(agent, message, user_id=user_id, max_retries=max_retries)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise AgentCallError(
            f"{agent.name} returned non-JSON output: {raw[:300]!r}"
        ) from None
