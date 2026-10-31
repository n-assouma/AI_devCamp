"""The public API. Other people's code imports these names, so they are a
contract: do not rename them.

    generate_explanations(question)              -> {style: text, ...}
    predict(profile_notes, question)             -> {pick, correct_after_pick, reason}
    update(profile_notes, prediction, actual)    -> new profile notes (str)
    run_round(...)                               -> one scored round log
    normalise_answer(s)                          -> comparable answer string

All four are plain synchronous functions. The async ADK machinery underneath
is driven by :mod:`tutor_agent.runner`, which owns a dedicated event loop so
these work unchanged inside Streamlit, a notebook, or a bare script.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from .agent import explainer_agent, predictor_agent, updater_agent
from .config import (
    DEFAULT_STUDENT,
    MAX_EXPLANATION_WORDS,
    MAX_PROFILE_WORDS,
    MAX_REPAIR_ATTEMPTS,
    STYLES,
)
from .prompts import explainer_request, predictor_request, updater_request
from .runner import AgentCallError, call_agent_json
from .schemas import prediction_schema

__all__ = [
    "generate_explanations",
    "predict",
    "update",
    "run_round",
    "choose_intervention",
    "choose_presentation",
    "recommend_style",
    "style_evidence",
    "normalise_answer",
    "answers_match",
    "clamp_words",
    "new_profile",
]


# ==========================================================================
# Answer normalisation
# ==========================================================================

_VAR_EQUALS = re.compile(r"^[a-z]\s*=\s*")
_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-"})


def normalise_answer(s: Any) -> str:
    """Reduce a typed answer to a comparable canonical form.

    A live human types ``x=4``, ``x = 4``, ``4 `` or ``4.0``; all of those
    mean the same thing and all of them must score as correct, because
    "the demo marked me wrong" is not a story anyone wants to tell on stage.

    Strips whitespace and case, normalises unicode minus signs, drops a
    leading single-variable assignment (``x=``, ``y =``, ``n=``), drops a
    trailing full stop, and collapses integral floats.

    >>> [normalise_answer(v) for v in ("x=4", " 4 ", "x = 4", "4", "4.0")]
    ['4', '4', '4', '4', '4']
    >>> normalise_answer("x = -1/2")
    '-1/2'
    """
    if s is None:
        return ""
    text = str(s).strip().lower().translate(_DASHES)
    text = re.sub(r"\s+", "", text)
    text = _VAR_EQUALS.sub("", text)
    text = text.rstrip(".").lstrip("+")
    if text.startswith("-+"):
        text = "-" + text[2:]
    try:
        value = float(text)
    except ValueError:
        return text
    if value.is_integer():
        return str(int(value))
    return repr(value)


def answers_match(given: Any, expected: Any) -> bool:
    """Compare two answers through :func:`normalise_answer`."""
    normalised = normalise_answer(given)
    return bool(normalised) and normalised == normalise_answer(expected)


def clamp_words(text: str, limit: int) -> str:
    """Hard-truncate ``text`` to ``limit`` words.

    The prompts ask for the limit and the models mostly respect it; this is
    the belt to that braces, so a chatty generation can never blow out a
    profile or a UI card.
    """
    words = (text or "").split()
    if len(words) <= limit:
        return (text or "").strip()
    return " ".join(words[:limit]).rstrip(",;:") + "..."


def new_profile(student: str = DEFAULT_STUDENT, notes: str = "") -> dict[str, str]:
    """A blank profile. The agent is supposed to start out knowing nothing."""
    return {"student": student, "notes": notes}


# ==========================================================================
# 1. EXPLAINER
# ==========================================================================


def generate_explanations(
    question: str, styles: list[str] | None = None
) -> dict[str, str]:
    """Write one explanation of ``question`` per configured style.

    Returns ``{"steps": "...", "analogy": "...", "example": "..."}`` --
    keyed by whatever :data:`tutor_agent.config.STYLES` currently holds, so
    adding a style changes the shape of this dict and nothing else.

    One model call per style (not one call for all of them), which is what
    keeps each explanation independently inside the word budget and keeps
    this function honest when the style list grows. Run this offline once
    per question and cache the result into questions.json; the live demo
    must never wait on it.
    """
    chosen = list(STYLES if styles is None else styles)
    out: dict[str, str] = {}
    for style in chosen:
        payload = call_agent_json(
            explainer_agent, explainer_request(question, style)
        )
        text = str(payload.get("explanation", "")).strip()
        if not text:
            raise AgentCallError(
                f"explainer returned an empty explanation for style {style!r}"
            )
        out[style] = clamp_words(text, MAX_EXPLANATION_WORDS)
    return out


# ==========================================================================
# 2. PREDICTOR  --  blind by construction
# ==========================================================================


def choose_intervention(
    history: Sequence[Mapping[str, Any]],
    *,
    styles: Sequence[str] | None = None,
    every: int = 3,
) -> str | None:
    """Pick a style to FORCE on the student this round, or None to let them choose.

    Why this exists. A student with a strong preference starves the agent of
    evidence: Maya picks "steps" ~85% of the time, so in twenty rounds the
    agent may never once observe what happens after she reads a worked
    example -- and "examples are what actually help her" is precisely the
    thing the profile is supposed to discover. You cannot learn a rule from
    evidence that was never generated.

    So every ``every`` rounds the tutor stops observing and runs a small
    experiment: it shows the style it has the LEAST evidence about and
    watches what happens. That is the difference between a passive observer
    and an agent.

    Returns None on non-intervention rounds (the student chooses freely).
    """
    styles = list(STYLES if styles is None else styles)
    if every <= 0 or (len(history) + 1) % every != 0:
        return None
    seen = Counter(log.get("actual_pick") for log in history)
    # Fewest observations first; STYLES order breaks ties so the choice is
    # deterministic and a rerun of the simulation reproduces exactly.
    return min(styles, key=lambda s: (seen.get(s, 0), styles.index(s)))


def style_evidence(
    history: Sequence[Mapping[str, Any]], styles: Sequence[str] | None = None
) -> dict[str, dict[str, int | float]]:
    """Per style: how often it was READ, and how often it then worked.

    Counts what the student actually read, which on a led or forced round is
    the tutor's choice rather than their preference. That is the whole point:
    ``helps`` is the only column that says anything about learning, and it can
    only be filled for styles the student was actually shown.
    """
    styles = list(STYLES if styles is None else styles)
    out = {s: {"shown": 0, "right": 0, "helps": 0.0} for s in styles}
    for row in history:
        style = row.get("actual_pick")
        if style in out:
            out[style]["shown"] += 1
            out[style]["right"] += bool(row.get("actual_correct"))
    for cell in out.values():
        if cell["shown"]:
            cell["helps"] = cell["right"] / cell["shown"]
    return out


def recommend_style(
    history: Sequence[Mapping[str, Any]],
    *,
    styles: Sequence[str] | None = None,
    min_evidence: int = 4,
    margin: float = 0.15,
) -> str | None:
    """The style this student actually LEARNS from, or None if not yet known.

    Deliberately not "the style they pick". A recommendation built from
    preference would just hand Maya the steps she already chooses and change
    nothing; the profile exists to find the gap between those two.

    Returns None unless one style has at least ``min_evidence`` observations
    and beats the next best by ``margin``. Leading a student toward a style on
    one lucky round is worse than not leading them at all.
    """
    styles = list(STYLES if styles is None else styles)
    ev = style_evidence(history, styles)
    ranked = sorted(
        (s for s in styles if ev[s]["shown"] >= min_evidence),
        key=lambda s: (-ev[s]["helps"], styles.index(s)),
    )
    # At least two styles must have been observed. With only one, it "wins"
    # by default and the tutor would confidently lead Maya toward the steps
    # that are the only thing she has ever been shown -- recommending the
    # student's own habit back to them, which is exactly the failure this
    # whole project is about. A comparison needs something to compare to.
    if len(ranked) < 2:
        return None
    best = ranked[0]
    rest = [ev[s]["helps"] for s in ranked[1:]]
    if ev[best]["helps"] - max(rest) < margin:
        return None  # nothing clearly better; keep offering the choice
    return best


def choose_presentation(
    history: Sequence[Mapping[str, Any]],
    *,
    styles: Sequence[str] | None = None,
    probe_every: int = 3,
    min_evidence: int = 4,
    explore_every: int = 3,
) -> tuple[str, str | None]:
    """How to present this round: ``("choose"|"probe"|"lead", style)``.

    The three-way split exists because the two obvious designs are both wrong.
    Always showing one explanation leaves nothing to predict, so the agent
    stops learning after round one. Always showing three means the agent
    learns the student and then never does anything with it -- it understands
    Maya perfectly and still lets her pick the thing that does not work.

    So:

    ``probe``   force the least-evidenced style. Buys information at the
                student's expense, which is why it is rationed.
    ``lead``    open with the style the evidence says HELPS them, once there
                is enough evidence to justify it. This is the payoff: the
                tutor visibly changes its behaviour because of what it
                learned. The other explanations stay available, so the
                student still chooses and ``pick`` stays predictable.
    ``choose``  show all of them, no steer.

    One round in ``explore_every`` always falls back to ``choose`` even when a
    recommendation exists, so the preference signal never dries up entirely
    and the know-you score keeps both halves alive.
    """
    styles = list(STYLES if styles is None else styles)
    n = len(history) + 1
    ev = style_evidence(history, styles)

    # 1. Keep probing the least-seen style -- ALWAYS, not only while evidence
    #    is thin. This is the guard against the trap that leading creates:
    #    once the tutor starts leading with X, X collects every subsequent
    #    observation and the alternatives are never tested again, so a wrong
    #    recommendation becomes self-confirming and permanent. Measured on
    #    this cohort, stopping probes once each style had two observations
    #    locked iris and omar onto the wrong style for the rest of the run
    #    and made their outcomes worse than no leading at all.
    #
    #    One round in probe_every is the standing cost of keeping a
    #    recommendation falsifiable.
    if probe_every > 0 and n % probe_every == 0:
        return "probe", min(
            styles, key=lambda s: (ev[s]["shown"], styles.index(s))
        )

    best = recommend_style(history, styles=styles, min_evidence=min_evidence)
    if best is None:
        return "choose", None  # nothing worth leading with yet

    # 2. Keep checking the preference, but less and less often. A FIXED
    #    explore rate was the bug: at one round in three it never stops
    #    costing the student, and Maya spent 25 of 40 rounds reading the
    #    steps that do not work for her even though the tutor had already
    #    worked out that examples do. Exploration is an investment, and an
    #    investment you keep making at full size forever is just a tax.
    #    The period grows with evidence, so leading becomes the normal case
    #    while the pick signal never disappears entirely.
    seen = sum(int(cell["shown"]) for cell in ev.values())
    explore_period = max(explore_every, 3 + seen // 6)
    if n % explore_period == 1:
        return "choose", None
    return "lead", best


def predict(
    profile_notes: str, question: str, forced_style: str | None = None
) -> dict[str, Any]:
    """Make the sealed prediction for one round.

    Returns ``{"pick": str, "correct_after_pick": bool, "reason": str}``.

    THE SIGNATURE IS THE ENFORCEMENT MECHANISM. This function takes the
    profile and the question and *physically cannot* be passed the correct
    answer, the student's actual choice, or whether they got it right --
    there is no parameter for any of them. That blindness is the scientific
    claim of the whole project: an accuracy number is only evidence that the
    profile models the student if the prediction was made without access to
    the thing being predicted. Do not add an ``actual=`` parameter here "for
    convenience"; it would silently void every chart in the demo.

    (Two further guards back this up: the predictor agent sets
    ``include_contents="none"`` so no conversation history can leak an
    earlier outcome into it, and every call runs in a brand-new session.)
    """
    payload = call_agent_json(
        predictor_agent,
        predictor_request(profile_notes, question, forced_style),
    )
    model = prediction_schema().model_validate(payload)
    return {
        # On an intervention round the pick is the tutor's decision, not a
        # prediction, so we trust our own value over whatever the model echoed.
        "pick": forced_style or model.pick,
        "correct_after_pick": bool(model.correct_after_pick),
        "reason": model.reason.strip(),
        "forced": forced_style is not None,
    }


# ==========================================================================
# 3. UPDATER
# ==========================================================================


def update(
    profile_notes: str,
    prediction: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    question: str | None = None,
    confirming: bool = False,
) -> str:
    """Rewrite the profile after a prediction was contradicted.

    ``prediction`` is a :func:`predict` result; ``actual`` is
    ``{"pick": str, "correct": bool}``. ``question`` is optional context.
    Returns the new profile notes, clamped to the word budget.

    The returned notes are general rules, not a transcript -- see
    :func:`tutor_agent.prompts.updater_instruction` for why that is enforced
    so aggressively. A profile that logs occasions scores well on the next
    recheck and predicts nothing on the next question.
    """
    payload = call_agent_json(
        updater_agent,
        updater_request(profile_notes, dict(prediction), dict(actual), question=question),
    )
    notes = str(payload.get("notes", "")).strip()
    if not notes:
        raise AgentCallError("updater returned an empty profile")
    return clamp_words(notes, MAX_PROFILE_WORDS)


# ==========================================================================
# 4. THE ROUND LOOP  --  score first, then repair
# ==========================================================================


def _resolve_action(
    act: Mapping[str, Any] | Callable[..., Mapping[str, Any]],
    question: Mapping[str, Any],
    explanations: Mapping[str, str],
    force_style: str | None = None,
) -> dict[str, Any]:
    """Get the student's action, optionally by calling back into the caller.

    ``act`` may be a ready-made ``{"pick", "correct"}`` dict, or a callable.
    The callable form exists so that ``run_round`` can invoke the student
    *after* the prediction is sealed, inside this function, where the
    ordering is structurally guaranteed rather than merely documented. Note
    that the callback is never handed the prediction.
    """
    if callable(act):
        try:
            raw = act(question, explanations)
        except TypeError:
            raw = act(question)
    else:
        raw = act
    # On an intervention round the tutor overrode the choice, so the student's
    # own preference is not what they saw. Whatever they "picked" is discarded
    # in favour of what they were actually shown -- otherwise the profile would
    # record a style the student never read.
    pick = force_style or str(raw["pick"])
    return {"pick": pick, "correct": bool(raw["correct"])}


def run_round(
    round_number: int,
    profile: Mapping[str, Any],
    question: Mapping[str, Any],
    act: Mapping[str, Any] | Callable[..., Mapping[str, Any]],
    *,
    force_style: str | None = None,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
) -> dict[str, Any]:
    """Run one measured round: sealed prediction, score, then repair.

    ORDERING IS THE WHOLE DESIGN. It goes:

    1. **Sealed prediction.** ``predict(profile.notes, question)`` runs with
       the profile as it stood *before* this round, knowing nothing about
       what the student is about to do.
    2. **The student acts.** Supplied by the caller (simulator rules, or a
       real human in the UI).
    3. **Score it, once.** ``pick_hit`` and ``correct_hit`` are computed from
       that first blind prediction and are then frozen. They are what the
       demo chart plots, and they are NEVER recomputed.
    4. **Only then, repair the profile.** Scoring is already closed, so the
       repair loop cannot contaminate the metric.

    Why the scoring must close before the repair: the repair loop is a better
    way of *writing* the profile, not a measurement of it. It deliberately
    lets the updater see the real outcome, and then re-runs the predictor on
    the rewritten profile to check the rewrite actually bites. If the logged
    score were taken from that recheck, the agent would be grading itself on
    an answer it had already been told -- a number guaranteed to look good
    and guaranteed to mean nothing. The honest metric is step 3 and only
    step 3.

    Why the repair loop is capped at ``MAX_REPAIR_ATTEMPTS`` (2) and, on
    failure, **keeps the old profile**: the simulator injects ~15% random
    noise into the student's picks, so a meaningful share of misses are not
    signal at all. A loop that retried "until the recheck passes" would
    eventually find some rewrite that explains a coin flip, and the profile
    would fill up with rules fitted to noise -- a strictly worse model than
    the one it replaced. Failing twice is itself useful evidence that the
    observation was noise, and the correct response to noise is to do
    nothing. Hence ``while ... else: keep old profile``, not ``while True``.

    Returns the round log::

        {"round", "student", "pred_pick", "pred_correct", "pred_reason",
         "actual_pick", "actual_correct", "pick_hit", "correct_hit",
         "profile_before", "profile_after", "profile_changed",
         "repair_attempts", "repaired"}

    ``profile_after`` is the notes string to carry into the next round.
    """
    notes_before = str(profile.get("notes", "") or "")
    student = str(profile.get("student", DEFAULT_STUDENT))

    # `question` is normally a questions.json entry; a bare string is
    # accepted so the loop can be exercised without the cached explanations.
    if isinstance(question, Mapping):
        question_text = str(question.get("question", ""))
        explanations: Mapping[str, str] = question.get("explanations", {}) or {}
    else:
        question_text = str(question)
        explanations = {}

    # ---- 1. SEALED PREDICTION -------------------------------------------
    pred = predict(notes_before, question_text, force_style)

    # ---- 2. the student acts (caller's business, not ours) --------------
    actual = _resolve_action(act, question, explanations, force_style)

    # ---- 3. SCORE IT ONCE. Frozen from here. This is the chart. --------
    # On an intervention round the tutor chose the style, so "did we predict
    # the pick" is not a question that was asked -- scoring it would inflate
    # the know-you score with a guess we were handed the answer to.
    # pick_hit is None there and must be EXCLUDED from accuracy, not counted
    # as a miss. correct_hit stays live, and on these rounds it is the whole
    # point: "given she is made to read an example, does it actually help?"
    log: dict[str, Any] = {
        "round": round_number,
        "student": student,
        "forced_style": force_style,
        "pred_pick": pred["pick"],
        "pred_correct": pred["correct_after_pick"],
        "pred_reason": pred["reason"],
        "actual_pick": actual["pick"],
        "actual_correct": actual["correct"],
        "pick_hit": None if force_style else pred["pick"] == actual["pick"],
        "correct_hit": pred["correct_after_pick"] == actual["correct"],
        "profile_before": notes_before,
    }

    # ---- 4. ONLY NOW repair the profile --------------------------------
    notes_after = notes_before
    attempts = 0
    repaired = False

    # ``pick_hit is None`` means the pick was never predicted (intervention
    # round). That is not a miss, so it must not drag the profile into a
    # rewrite on its own.
    #
    # But a miss is not the only thing worth learning from. Rewriting only on
    # failure silently discards every piece of CONFIRMING evidence -- and on
    # an intervention round that is precisely the evidence we paid a round to
    # go and get. The case that matters: the tutor forces Maya to read a
    # worked example, she gets it right, the agent predicted she would, and
    # because that is a hit the profile is left untouched. "Examples are what
    # actually help Maya" is the one fact this entire project exists to
    # discover, and the passive version of this loop throws it away.
    #
    # So an intervention round always folds its observation into the profile,
    # hit or miss. Ordinary rounds still only rewrite on a miss: re-confirming
    # a preference the profile already predicted correctly costs a model call
    # and teaches nothing.
    missed = (log["pick_hit"] is False) or (log["correct_hit"] is False)
    log["confirming"] = bool(force_style) and not missed
    if missed or log["confirming"]:
        while attempts < max_repair_attempts:
            attempts += 1
            try:
                candidate = update(
                    notes_before,
                    pred,
                    actual,
                    question=question_text,
                    confirming=log["confirming"],
                )
                if log["confirming"]:
                    # Nothing to verify: the prediction was already right, so
                    # a recheck can only spend a model call to re-learn what
                    # we know. Bank the evidence and move on.
                    notes_after = candidate
                    repaired = True
                    break
                recheck = predict(candidate, question_text, force_style)
            except AgentCallError:
                # A failed rewrite is treated exactly like a rejected one:
                # keep the profile we already trust.
                break
            # The recheck must satisfy BOTH halves of the prediction, not just
            # the pick. Checking only the pick makes the loop a no-op for the
            # most important case: once the agent has learned "Maya picks
            # steps", every Maya round is pick_hit=True / correct_hit=False,
            # so a pick-only recheck passes trivially on attempt 1 and accepts
            # a rewrite that fixed nothing. Her real pattern -- prefers steps,
            # only learns from examples -- would never be verified.
            pick_ok = force_style is not None or recheck["pick"] == actual["pick"]
            if pick_ok and recheck["correct_after_pick"] == actual["correct"]:
                notes_after = candidate
                repaired = True
                break
        # else-branch semantics: no rewrite survived the recheck, so
        # notes_after is still notes_before -- the observation was probably
        # noise and the old profile stands.

    log["profile_after"] = notes_after
    log["profile_changed"] = notes_after != notes_before
    log["repair_attempts"] = attempts
    log["repaired"] = repaired
    return log
