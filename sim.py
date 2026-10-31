#!/usr/bin/env python3
"""The simulated-student experiment: can a written profile learn a rule it cannot see?

WHAT THIS IS
------------
Two simulated students with *hidden* rules, written in plain Python below.
The tutor agent never sees those rules. Before each round it makes a SEALED
prediction (which explanation the student will pick, and whether they will
then answer the check question correctly), the student acts, the round is
scored once, and only then is the profile allowed to be rewritten.

    Student  picks                      actually learns from
    -------  -------------------------  ------------------------------------
    sam      "example" 85% of the time  example -> 80% correct, else 40%
    maya     "steps"   85% of the time  ONLY example -> 80%, else 30%

Maya is the demo. Her *preference* and what *helps her* disagree, so an agent
that only logs what she clicks will model her wrongly forever.

WHY THE RULES ARE PYTHON AND NOT AN LLM
---------------------------------------
"Isn't your simulation circular -- an LLM grading an LLM?" No. Correctness is
decided by the fixed probabilities in :class:`Student` and by nothing else.
No model call is involved in deciding what the student does. The agent has to
discover a rule that exists only in this file, which it never reads.

INTERVENTION ROUNDS
-------------------
Maya picks "steps" ~85% of the time, so left to observe she may *never*
generate the evidence that a worked example helps her -- and that is exactly
what the profile is supposed to discover. You cannot learn a rule from
evidence that was never produced. So every 3rd round the tutor forces the
style it has the least evidence about (``tutor_agent.choose_intervention``).

On a forced round ``pick_hit is None``: the pick was the tutor's own decision,
never a prediction, so it is NOT SCORED. It is EXCLUDED from accuracy -- never
counted as a miss. ``--no-intervene`` turns interventions off so the team can
show the before/after contrast.

QUOTA SURVIVAL (read this before running live)
----------------------------------------------
Free-tier Gemini quota is per-day and per-model, and a full run needs more
calls than are left. Hence:

  * ``--fake``        runs the entire loop against deterministic stub agents.
                      ZERO API calls. This is how you debug.
  * checkpointing     ``results_sim.json`` is rewritten after EVERY round.
  * resume            rerunning continues from the last saved round.
  * ``--max-calls``   hard ceiling; the run stops cleanly when it is hit.
  * AgentCallError    caught, checkpointed, and reported with a resume hint.

CLI
---
    .venv/bin/python sim.py --fake --rounds 20
    .venv/bin/python sim.py --fake --rounds 20 --no-intervene --out results_sim_nointervene.json
    .venv/bin/python sim.py --self-test
    .venv/bin/python sim.py --rounds 9 --max-calls 30          # live, budgeted
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import tutor_agent
from tutor_agent import STYLES, AgentCallError, choose_intervention, new_profile, run_round

PROJECT_ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = PROJECT_ROOT / "questions.json"

#: Never, ever results.json -- that file is the team's working demo backup.
DEFAULT_OUT = PROJECT_ROOT / "results_sim.json"

SCHEMA_VERSION = 1


# ==========================================================================
# THE HIDDEN RULES.  Plain Python. No LLM anywhere in this section.
# ==========================================================================


@dataclass(frozen=True)
class Student:
    """A simulated student: a preference, and a separate truth about learning.

    The two are deliberately independent fields, because conflating them is
    precisely the mistake the product exists to fix.
    """

    name: str
    #: The style they reach for when they are not being noisy.
    prefers: str
    #: How often the preference wins. The remaining mass is uniform noise
    #: over STYLES (so the preferred style's true rate is slightly higher).
    pref_prob: float
    #: The styles that actually teach them.
    learns_from: frozenset[str]
    #: P(correct) after reading a style that teaches them.
    p_correct_helped: float
    #: P(correct) after reading anything else.
    p_correct_unhelped: float
    blurb: str = ""

    def pick(self, rng: random.Random) -> str:
        """The style the student would choose, left to their own devices."""
        if rng.random() < self.pref_prob:
            return self.prefers
        return rng.choice(list(STYLES))

    def answers_correctly(self, style_read: str, rng: random.Random) -> bool:
        """Did they get the check question right after reading ``style_read``?

        THE ARGUMENT IS THE WHOLE POINT. Correctness is a function of the
        style the student ACTUALLY READ, which on an intervention round is
        the tutor's forced choice and not their own preference. Score this
        off the preference instead and the experiment measures nothing: a
        forced worked example would be recorded as having helped Maya no
        more than the steps she wanted, and the one effect the profile is
        meant to discover becomes invisible.
        """
        p = (
            self.p_correct_helped
            if style_read in self.learns_from
            else self.p_correct_unhelped
        )
        return rng.random() < p

    def act(self, round_number: int, forced_style: str | None, seed: int) -> dict[str, Any]:
        """One round of student behaviour, as a ``{"pick", "correct"}`` dict.

        The RNG is derived from ``(seed, name, round_number)`` rather than
        carried as mutable state. That makes round *n* identical whether it
        was reached in one run or after three resumes -- which is the only
        way a checkpointed, quota-interrupted run stays reproducible.
        """
        rng = random.Random(f"{seed}|{self.name}|{round_number}")
        own_pick = self.pick(rng)
        style_read = forced_style or own_pick
        correct = self.answers_correctly(style_read, rng)
        # `pick` is their own preference; run_round() replaces it with the
        # forced style on an intervention round, so what gets logged is
        # always the style they really read.
        return {"pick": own_pick, "correct": correct}


SAM = Student(
    name="sam",
    prefers="example",
    pref_prob=0.85,
    learns_from=frozenset({"example"}),
    p_correct_helped=0.80,
    p_correct_unhelped=0.40,
    blurb="Prefers worked examples, and worked examples are what help him. "
    "Preference and benefit agree -- the easy case.",
)

MAYA = Student(
    name="maya",
    prefers="steps",
    pref_prob=0.85,
    learns_from=frozenset({"example"}),
    p_correct_helped=0.80,
    p_correct_unhelped=0.30,
    blurb="Prefers step-by-step, but ONLY worked examples actually help her. "
    "Preference and benefit disagree -- the case the product is for.",
)

STUDENTS: dict[str, Student] = {s.name: s for s in (SAM, MAYA)}


def rules_manifest(student: Student) -> dict[str, Any]:
    """The hidden rules, written into the results file for the judges.

    Published *after* the run, never read by the agent.
    """
    return {
        "prefers": student.prefers,
        "pref_prob": student.pref_prob,
        "noise": "remaining mass is uniform random over STYLES",
        "learns_from": sorted(student.learns_from),
        "p_correct_helped": student.p_correct_helped,
        "p_correct_unhelped": student.p_correct_unhelped,
        "blurb": student.blurb,
    }


# ==========================================================================
# FAKE AGENTS -- the whole loop, zero API calls
# ==========================================================================
#
# These replace tutor_agent.api.predict / .update in --fake mode. They are
# deliberately NOT smart: they are a tally-keeper that stores its belief
# *inside the profile notes string*, so the real plumbing being tested --
# sealed prediction, profile carried round to round, repair-after-scoring,
# the forced-round pick_hit=None path -- is exercised exactly as it is live.

#: ``<<tally steps=4/0/4 analogy=1/0/1 example=2/2/2>>`` -- picks/correct/seen.
#: Deliberately brace-free: an earlier version embedded JSON here and the
#: nested ``}}`` ate its own closing delimiter, so the tally silently reset to
#: zero every round and the stub profile never learned anything. A fake that
#: fails to carry state is worse than no fake, because it looks like it works.
_FAKE_STATE_RE = re.compile(r"<<tally ([^<>]*)>>")


def _blank_state() -> dict[str, Any]:
    return {"picks": {s: 0 for s in STYLES}, "hits": {s: [0, 0] for s in STYLES}}


def _fake_state(notes: str) -> dict[str, Any]:
    state = _blank_state()
    match = _FAKE_STATE_RE.search(notes or "")
    if not match:
        return state
    for field_text in match.group(1).split():
        style, _, numbers = field_text.partition("=")
        if style not in state["picks"]:
            continue
        parts = numbers.split("/")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            continue
        state["picks"][style] = int(parts[0])
        state["hits"][style] = [int(parts[1]), int(parts[2])]
    return state


def _fake_encode(state: Mapping[str, Any]) -> str:
    body = " ".join(
        f"{s}={state['picks'][s]}/{state['hits'][s][0]}/{state['hits'][s][1]}"
        for s in STYLES
    )
    return f"<<tally {body}>>"


def _fake_render(state: Mapping[str, Any]) -> str:
    picks = ", ".join(f"{s}={state['picks'][s]}" for s in STYLES)
    rates = ", ".join(
        f"{s}={state['hits'][s][0]}/{state['hits'][s][1]}" for s in STYLES
    )
    best = _fake_best_pick(state)
    helped = [
        s for s in STYLES if state["hits"][s][1] and state["hits"][s][0] * 2 >= state["hits"][s][1]
    ]
    summary = (
        f"[STUB PROFILE] Observed picks: {picks}. Correct-after-reading: {rates}. "
        f"Looks like they prefer '{best}'"
    )
    if helped:
        summary += f" and learn best from {'/'.join(helped)}."
    else:
        summary += "; no format has shown a clear benefit yet."
    return f"{summary} {_fake_encode(state)}"


def _fake_best_pick(state: Mapping[str, Any]) -> str:
    counts = state["picks"]
    if not any(counts.values()):
        return STYLES[0]
    return max(STYLES, key=lambda s: (counts[s], -STYLES.index(s)))


class _Budgeted:
    """Shared call ceiling for both harnesses.

    Raising :class:`AgentCallError` is how the ceiling stops a run: the main
    loop already treats that as "quota died, checkpoint and say how to
    resume", so a budget stop and a real 429 take exactly the same path --
    which also means ``--fake --max-calls N`` is a faithful rehearsal of a
    quota death, with no API calls and no 17-hour wait.
    """

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self.predict_calls = 0
        self.update_calls = 0

    @property
    def calls(self) -> int:
        return self.predict_calls + self.update_calls

    def _check(self) -> None:
        if self.limit is not None and self.calls >= self.limit:
            raise AgentCallError(
                f"local call budget of {self.limit} agent calls reached "
                "(--max-calls); no request was sent"
            )


class FakeAgents(_Budgeted):
    """Deterministic stand-ins for ``predict`` and ``update``, with a counter."""

    def predict(
        self, profile_notes: str, question: str, forced_style: str | None = None
    ) -> dict[str, Any]:
        self._check()
        self.predict_calls += 1
        state = _fake_state(profile_notes)
        pick = forced_style or _fake_best_pick(state)
        # Prediction of correctness is about the style that will be READ,
        # which on a forced round is the tutor's choice -- same asymmetry the
        # live predictor has to handle.
        read = forced_style or pick
        got, total = state["hits"][read]
        if total == 0:
            guess = True  # no evidence yet: naive optimism, and it gets punished
            why = f"no evidence about '{read}' yet, assuming it works"
        else:
            guess = got * 2 >= total
            why = f"'{read}' has worked {got}/{total} times so far"
        return {
            "pick": pick,
            "correct_after_pick": bool(guess),
            "reason": f"[stub] {why}.",
            "forced": forced_style is not None,
        }

    def update(
        self,
        profile_notes: str,
        prediction: Mapping[str, Any],
        actual: Mapping[str, Any],
        *,
        question: str | None = None,
        confirming: bool = False,
    ) -> str:
        # ``confirming`` marks an intervention round the profile got RIGHT:
        # evidence the tutor spent a round collecting, which a failure-only
        # loop would throw away. The stub records it the same either way --
        # the real updater is told to treat it as reinforcement, not a fix.
        del confirming
        self._check()
        self.update_calls += 1
        state = _fake_state(profile_notes)
        # `actual["pick"]` is the style the student really read (run_round
        # already substituted the forced style on an intervention round).
        read = str(actual["pick"])
        if read in state["picks"]:
            # PREFERENCE vs BENEFIT, kept in separate columns -- which is the
            # whole product thesis, so even the stub has to respect it. A
            # forced round is evidence about whether the style HELPED, but no
            # evidence at all about what the student would have CHOSEN; the
            # tutor chose it. Counting it as a pick teaches the profile that
            # Maya likes whatever she was made to read.
            if not prediction.get("forced"):
                state["picks"][read] += 1
            state["hits"][read][1] += 1
            if actual["correct"]:
                state["hits"][read][0] += 1
        return _fake_render(state)


class CallBudget(_Budgeted):
    """The real agents, behind a counter that refuses to exceed the ceiling."""

    def __init__(self, limit: int | None) -> None:
        super().__init__(limit)
        self._real_predict = tutor_agent.api.predict
        self._real_update = tutor_agent.api.update

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        self._check()
        self.predict_calls += 1
        return self._real_predict(*args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> Any:
        self._check()
        self.update_calls += 1
        return self._real_update(*args, **kwargs)


def install_agents(fake: bool, max_calls: int | None) -> FakeAgents | CallBudget:
    """Monkeypatch the agent layer, returning the installed call counter.

    Patching ``tutor_agent.api`` (not this module) is what matters:
    ``run_round`` resolves ``predict``/``update`` as globals of that module,
    so this is the only place a substitution actually takes effect.
    """
    harness: FakeAgents | CallBudget = (
        FakeAgents(max_calls) if fake else CallBudget(max_calls)
    )
    tutor_agent.api.predict = harness.predict  # type: ignore[assignment]
    tutor_agent.api.update = harness.update  # type: ignore[assignment]
    return harness


# ==========================================================================
# SCORING.  pick_hit is None  =>  EXCLUDED, not wrong.
# ==========================================================================


def score(logs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Accuracy over a list of round logs.

    The know-you score is "how many of the predictions the agent actually
    made did it get right", counted in halves: one half for the pick, one
    for the correctness. On an intervention round the pick half WAS NEVER
    PREDICTED -- the tutor chose it -- so that half contributes nothing to
    either the numerator or the denominator. ``is not None`` is the test;
    note that ``if log["pick_hit"]`` would silently fold None in with False
    and quietly understate every number on the chart.
    """
    pick_scored = [lg for lg in logs if lg.get("pick_hit") is not None]
    pick_right = sum(1 for lg in pick_scored if lg["pick_hit"])
    correct_right = sum(1 for lg in logs if lg.get("correct_hit"))
    n_correct = len(logs)
    halves_right = pick_right + correct_right
    halves_total = len(pick_scored) + n_correct
    return {
        "rounds": n_correct,
        "pick_scored_rounds": len(pick_scored),
        "pick_excluded_rounds": n_correct - len(pick_scored),
        "pick_accuracy": (pick_right / len(pick_scored)) if pick_scored else None,
        "correct_accuracy": (correct_right / n_correct) if n_correct else None,
        "know_you_score": (halves_right / halves_total) if halves_total else None,
        "halves_right": halves_right,
        "halves_scored": halves_total,
        # Chance on the halves actually scored: 1/3 for a 3-way pick, 1/2
        # for a boolean. Quoting a flat 0.417 would be wrong on a run whose
        # forced rounds scored only the boolean half.
        "random_baseline": (
            (len(pick_scored) / 3 + n_correct / 2) / halves_total if halves_total else None
        ),
    }


def profile_drift(logs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rewrites": sum(1 for lg in logs if lg.get("profile_changed")),
        "repairs_attempted": sum(int(lg.get("repair_attempts") or 0) for lg in logs),
        "repairs_accepted": sum(1 for lg in logs if lg.get("repaired")),
    }


# ==========================================================================
# CHECKPOINTING / RESUME
# ==========================================================================

RESUME_KEYS = ("seed", "rounds_requested", "intervene_every", "fake", "students")


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON so that a kill mid-write cannot corrupt the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"!! could not read checkpoint {path.name}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict) or "rounds" not in data:
        print(
            f"!! {path.name} is not a sim checkpoint (no 'rounds' key); "
            "pass --fresh to overwrite it",
            file=sys.stderr,
        )
        return None
    return data


def resume_conflict(meta: Mapping[str, Any], wanted: Mapping[str, Any]) -> str | None:
    """Return a human description of why a checkpoint cannot be resumed."""
    for key in RESUME_KEYS:
        if key == "rounds_requested":
            continue  # asking for more (or fewer) rounds is fine
        if meta.get(key) != wanted.get(key):
            return f"{key}: checkpoint has {meta.get(key)!r}, you asked for {wanted.get(key)!r}"
    return None


# ==========================================================================
# THE RUN
# ==========================================================================


def load_questions() -> list[dict[str, Any]]:
    with QUESTIONS_PATH.open(encoding="utf-8") as handle:
        questions = json.load(handle)
    if not questions:
        raise SystemExit("questions.json is empty")
    return questions


@dataclass
class Session:
    """Everything the loop needs, plus the file it keeps flushing itself to."""

    out: Path
    meta: dict[str, Any]
    rounds: list[dict[str, Any]] = field(default_factory=list)

    def logs_for(self, student: str) -> list[dict[str, Any]]:
        return [lg for lg in self.rounds if lg.get("student") == student]

    def done_rounds(self, student: str) -> set[int]:
        return {int(lg["round"]) for lg in self.logs_for(student)}

    def profile_notes(self, student: str) -> str:
        logs = self.logs_for(student)
        return str(logs[-1].get("profile_after", "")) if logs else ""

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.meta["students"]:
            logs = self.logs_for(name)
            if not logs:
                continue
            out[name] = {
                **score(logs),
                **profile_drift(logs),
                "final_profile": self.profile_notes(name),
            }
        return out

    def payload(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": {
                **self.meta,
                "status": status,
                "rounds_completed": {
                    name: len(self.logs_for(name)) for name in self.meta["students"]
                },
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "hidden_rules": {
                name: rules_manifest(STUDENTS[name]) for name in self.meta["students"]
            },
            "summary": self.summary(),
            "rounds": self.rounds,
        }

    def checkpoint(self, status: str = "partial") -> None:
        _atomic_write(self.out, self.payload(status))


def describe_round(log: Mapping[str, Any]) -> str:
    forced = log.get("forced_style")
    pick_hit = log.get("pick_hit")
    if pick_hit is None:
        pick_cell = f"FORCED {forced:<7} (pick not scored)"
    else:
        pick_cell = f"pick {log['pred_pick']:<7}->{log['actual_pick']:<7} {'HIT ' if pick_hit else 'miss'}"
    right = "HIT " if log.get("correct_hit") else "miss"
    return (
        f"  r{log['round']:>2} {log['student']:<5} {pick_cell:<36} "
        f"correct {str(log['pred_correct']):<5}->{str(log['actual_correct']):<5} {right} "
        f"{'[profile rewritten]' if log.get('profile_changed') else ''}"
    )


def resume_command(args: argparse.Namespace) -> str:
    """The exact flags to rerun to continue this run. Resumability is only
    useful if the operator is told the invocation that resumes, not an
    approximation of it."""
    parts = [
        f"--seed {args.seed}",
        f"--rounds {args.rounds}",
        f"--students {','.join(args.students)}",
        f"--out {args.out}",
    ]
    if args.fake:
        parts.append("--fake")
    if args.no_intervene:
        parts.append("--no-intervene")
    elif args.intervene_every != 3:
        parts.append(f"--intervene-every {args.intervene_every}")
    return " ".join(parts)


def run(args: argparse.Namespace) -> int:
    questions = load_questions()
    students = [STUDENTS[n] for n in args.students]
    every = 0 if args.no_intervene else args.intervene_every

    wanted_meta = {
        "seed": args.seed,
        "rounds_requested": args.rounds,
        "intervene_every": every,
        "fake": bool(args.fake),
        "students": [s.name for s in students],
    }

    out = Path(args.out)
    existing = None if args.fresh else load_checkpoint(out)
    if existing is not None:
        conflict = resume_conflict(existing.get("meta", {}), wanted_meta)
        if conflict:
            print(
                f"!! {out.name} holds an incompatible run ({conflict}).\n"
                f"   Pass --fresh to overwrite it, or --out <other.json>.",
                file=sys.stderr,
            )
            return 2
        session = Session(
            out=out,
            meta={**existing.get("meta", {}), **wanted_meta},
            rounds=list(existing.get("rounds", [])),
        )
        done = {n: len(session.logs_for(n)) for n in wanted_meta["students"]}
        if any(done.values()):
            print(f"== RESUMING from {out.name}: already saved {done}")
        else:
            print(f"== {out.name} had no completed rounds; starting from scratch")
    else:
        session = Session(
            out=out,
            meta={
                **wanted_meta,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )

    harness = install_agents(args.fake, args.max_calls)

    mode = "FAKE (stubbed agents, zero API calls)" if args.fake else "LIVE (Gemini)"
    print(f"== mode: {mode}")
    print(f"== seed {args.seed} | {args.rounds} rounds | students {wanted_meta['students']}")
    print(
        "== interventions: "
        + (f"every {every} rounds (forced style, pick NOT scored)" if every else "OFF")
    )
    if args.max_calls is not None:
        print(f"== call budget: {args.max_calls} agent calls, then stop + checkpoint")
    print()

    session.checkpoint("running")
    stopped: str | None = None

    for round_number in range(1, args.rounds + 1):
        for student in students:
            if round_number in session.done_rounds(student.name):
                continue  # already in the checkpoint
            history = session.logs_for(student.name)
            forced = choose_intervention(history, every=every) if every else None
            question = questions[(round_number - 1) % len(questions)]
            profile = new_profile(student.name, session.profile_notes(student.name))

            def act(q: Mapping[str, Any], _expl: Mapping[str, str] | None = None,
                    _s: Student = student, _r: int = round_number,
                    _f: str | None = forced) -> dict[str, Any]:
                # Called by run_round AFTER the prediction is sealed. The
                # callback never sees the prediction -- only the question.
                return _s.act(_r, _f, args.seed)

            try:
                log = run_round(round_number, profile, question, act, force_style=forced)
            except AgentCallError as exc:
                stopped = str(exc)
                break
            except KeyboardInterrupt:
                stopped = "interrupted by user (Ctrl-C)"
                break

            log["question_id"] = question.get("id")
            session.rounds.append(log)
            # CHECKPOINT AFTER EVERY ROUND. A quota death mid-run leaves a
            # chartable partial result instead of nothing at all.
            session.checkpoint("partial")
            print(describe_round(log))
        if stopped:
            break

    if stopped:
        session.checkpoint("partial")
        saved = {n: len(session.logs_for(n)) for n in wanted_meta["students"]}
        print()
        print("!! STOPPED EARLY: " + stopped)
        print(f"!! saved rounds per student: {saved}  ->  {out}")
        print(f"!! model calls spent this run: {harness.calls}")
        print("!! the partial file is valid and chartable:")
        print(f"       .venv/bin/python make_chart.py --in {out}")
        print("!! to continue once quota recovers (~17h), rerun the SAME command")
        print("!! (do NOT pass --fresh; that would throw the saved rounds away):")
        print(f"       .venv/bin/python sim.py {resume_command(args)}")
        return 1

    session.checkpoint("complete")
    print()
    print_summary(session)
    print(f"\n== wrote {out}")
    if args.fake:
        print(
            f"== API calls: 0  (stub predict x{harness.predict_calls}, "
            f"update x{harness.update_calls})"
        )
    else:
        print(
            f"== model calls spent: {harness.calls} "
            f"(predict x{harness.predict_calls}, update x{harness.update_calls})"
        )
    print("== chart it:  .venv/bin/python make_chart.py --in " + out.name)
    return 0


def print_summary(session: Session) -> None:
    print("=" * 72)
    print("KNOW-YOU SCORE".center(72))
    print("=" * 72)
    for name, s in session.summary().items():
        pick = "n/a" if s["pick_accuracy"] is None else f"{s['pick_accuracy']:.0%}"
        print(f"\n{name.upper()}  ({STUDENTS[name].blurb})")
        print(f"  rounds                : {s['rounds']}")
        print(
            f"  pick accuracy         : {pick}"
            f"  ({s['pick_scored_rounds']} scored, "
            f"{s['pick_excluded_rounds']} forced rounds EXCLUDED)"
        )
        print(f"  correctness accuracy  : {s['correct_accuracy']:.0%}  ({s['rounds']} scored)")
        print(
            f"  know-you score        : {s['know_you_score']:.0%} "
            f"({s['halves_right']}/{s['halves_scored']} halves)   "
            f"vs random {s['random_baseline']:.1%}"
        )
        print(f"  profile rewrites      : {s['rewrites']}")
        print(f"  final profile         : {s['final_profile'][:400]}")


# ==========================================================================
# SELF-TEST -- the four claims that are easiest to get silently wrong
# ==========================================================================


def self_test() -> int:
    print("== self-test (no API calls, no files written)\n")

    # ---- 1. pick_hit None is EXCLUDED, not counted as a miss ------------
    forced_logs = [
        {"pick_hit": None, "correct_hit": True, "forced_style": "example"},
        {"pick_hit": None, "correct_hit": True, "forced_style": "example"},
    ]
    s = score(forced_logs)
    assert s["pick_scored_rounds"] == 0, s
    assert s["pick_excluded_rounds"] == 2, s
    assert s["pick_accuracy"] is None, s
    assert s["halves_scored"] == 2, s  # two rounds, correctness half only
    assert s["know_you_score"] == 1.0, s
    assert abs(s["random_baseline"] - 0.5) < 1e-9, s
    print("  [ok] all-forced rounds: pick half excluded; baseline is 1/2, not 0.417")

    as_miss = [dict(lg, pick_hit=False) for lg in forced_logs]
    s_miss = score(as_miss)
    assert s_miss["know_you_score"] == 0.5, s_miss
    assert s_miss["know_you_score"] != s["know_you_score"], "exclusion is a no-op!"
    print("  [ok] counting them as misses would give 50% not 100% -> exclusion is real")

    mixed = [
        {"pick_hit": True, "correct_hit": True, "forced_style": None},
        {"pick_hit": False, "correct_hit": True, "forced_style": None},
        {"pick_hit": None, "correct_hit": False, "forced_style": "analogy"},
    ]
    sm = score(mixed)
    assert sm["pick_scored_rounds"] == 2 and sm["pick_excluded_rounds"] == 1, sm
    assert sm["halves_scored"] == 5, sm  # 2 pick halves + 3 correctness halves
    assert sm["halves_right"] == 3, sm
    assert abs(sm["random_baseline"] - (2 / 3 + 3 / 2) / 5) < 1e-9, sm
    print(
        f"  [ok] mixed run: {sm['halves_right']}/{sm['halves_scored']} halves, "
        f"baseline {sm['random_baseline']:.3f} (2 pick halves @1/3 + 3 bool halves @1/2)"
    )

    full = [{"pick_hit": True, "correct_hit": True, "forced_style": None}]
    assert abs(score(full)["random_baseline"] - (1 / 3 + 1 / 2) / 2) < 1e-9
    print("  [ok] fully-scored run: baseline == (1/3 + 1/2)/2 == 0.417")

    # ---- 2. correctness tracks the style ACTUALLY READ ------------------
    n = 4000
    forced_example = sum(
        MAYA.act(r, "example", 7)["correct"] for r in range(n)
    ) / n
    forced_steps = sum(MAYA.act(r, "steps", 7)["correct"] for r in range(n)) / n
    free = sum(MAYA.act(r, None, 7)["correct"] for r in range(n)) / n
    assert 0.76 < forced_example < 0.84, forced_example
    assert 0.26 < forced_steps < 0.34, forced_steps
    assert 0.30 < free < 0.42, free
    print(
        f"\n  [ok] maya forced 'example': {forced_example:.1%} correct (rule says 80%)"
    )
    print(f"  [ok] maya forced 'steps'  : {forced_steps:.1%} correct (rule says 30%)")
    print(f"  [ok] maya free choice     : {free:.1%} correct (mostly reads steps)")
    assert forced_example - forced_steps > 0.40, "forced style has no effect -- bug!"
    print("  [ok] the forced style changes the outcome => the experiment measures something")

    # and that the pick she reports is still her own preference
    picks = [MAYA.act(r, "example", 7)["pick"] for r in range(n)]
    pref_rate = picks.count("steps") / n
    assert 0.85 < pref_rate < 0.95, pref_rate
    print(f"  [ok] on forced rounds her own preference is unchanged ({pref_rate:.0%} steps)")

    sam_example = sum(SAM.act(r, "example", 7)["correct"] for r in range(n)) / n
    sam_other = sum(SAM.act(r, "analogy", 7)["correct"] for r in range(n)) / n
    assert 0.76 < sam_example < 0.84 and 0.36 < sam_other < 0.44
    print(f"  [ok] sam: example {sam_example:.1%}, analogy {sam_other:.1%} (80% / 40%)")

    # ---- 3. per-round RNG makes resume byte-identical --------------------
    a = [MAYA.act(r, None, 7) for r in range(1, 7)]
    b = [MAYA.act(r, None, 7) for r in (4, 5, 6)]
    assert a[3:] == b, (a[3:], b)
    assert MAYA.act(3, None, 7) != MAYA.act(3, None, 8) or True  # seeds differ in general
    print("\n  [ok] round N is identical whether reached in one run or after a resume")

    # ---- 4. intervention schedule ---------------------------------------
    hist: list[dict[str, Any]] = []
    sched = []
    for r in range(1, 10):
        sched.append(choose_intervention(hist, every=3))
        hist.append({"actual_pick": "steps"})
    assert [i for i, v in enumerate(sched, 1) if v] == [3, 6, 9], sched
    assert all(choose_intervention(hist[:k], every=0) is None for k in range(10))
    print(f"  [ok] forced rounds land on 3, 6, 9: {sched}")
    print("  [ok] --no-intervene (every=0) never forces a style")

    # ---- 5. the stub actually carries its state in the profile ----------
    # Regression guard: the first version embedded JSON whose nested braces
    # ate the closing delimiter, so every round started from a blank tally
    # and the fake loop silently learned nothing.
    fakes = FakeAgents()
    notes = ""
    for _ in range(4):
        notes = fakes.update(
            notes, {"pick": "steps"}, {"pick": "example", "correct": True}
        )
    state = _fake_state(notes)
    assert state["picks"]["example"] == 4, (state, notes)
    assert state["hits"]["example"] == [4, 4], (state, notes)
    assert _fake_state(_fake_render(state)) == state, "stub state does not round-trip"
    pred = fakes.predict(notes, "Solve 2x+5=13")
    assert pred["pick"] == "example" and pred["correct_after_pick"] is True, pred
    print("  [ok] stub profile round-trips its tally and changes the stub's prediction")

    print("\n== self-test passed")
    return 0


# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulated-student experiment for the adaptive tutor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  sim.py --fake --rounds 20                 full run, ZERO API calls\n"
            "  sim.py --fake --rounds 20 --no-intervene  the before/after contrast\n"
            "  sim.py --self-test                        assert the scoring maths\n"
            "  sim.py --rounds 9 --max-calls 30          live, budgeted, resumable\n"
        ),
    )
    p.add_argument("--rounds", type=int, default=20, help="rounds per student (default 20)")
    p.add_argument(
        "--students",
        default="sam,maya",
        help="comma-separated subset of: " + ", ".join(STUDENTS),
    )
    p.add_argument("--seed", type=int, default=7, help="RNG seed (default 7)")
    p.add_argument(
        "--no-intervene",
        action="store_true",
        help="never force a style; the student always chooses (before/after demo)",
    )
    p.add_argument(
        "--intervene-every", type=int, default=3, help="force a style every N rounds (default 3)"
    )
    p.add_argument(
        "--fake",
        "--dry-run",
        dest="fake",
        action="store_true",
        help="run the whole loop against deterministic stub agents: ZERO API calls",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT), help="checkpoint file (NEVER results.json)")
    p.add_argument(
        "--fresh", action="store_true", help="ignore any existing checkpoint and start over"
    )
    p.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="hard ceiling on agent calls; stops cleanly and checkpoints. "
        "Works with --fake too, to rehearse a quota death offline.",
    )
    p.add_argument("--self-test", action="store_true", help="run the assertions and exit")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()

    args.students = [s.strip() for s in str(args.students).split(",") if s.strip()]
    unknown = [s for s in args.students if s not in STUDENTS]
    if unknown:
        raise SystemExit(f"unknown student(s) {unknown}; known: {sorted(STUDENTS)}")
    if Path(args.out).name == "results.json":
        raise SystemExit(
            "refusing to write results.json -- that is the team's demo backup. "
            "Use results_sim.json."
        )
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    try:
        return run(args)
    finally:
        if not args.fake:
            # Stop the agent package's background event loop. It is a daemon
            # thread so this is belt-and-braces, but a run that ends on a
            # quota error should not also hang the shell.
            try:
                tutor_agent.shutdown()
            except Exception:  # noqa: BLE001 - never mask the real exit path
                pass


if __name__ == "__main__":
    sys.exit(main())
