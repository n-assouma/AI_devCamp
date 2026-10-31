"""
sim.py  (Person B: simulator + chart)

Two simulated students with hidden rules in plain Python (no LLM), a loop that
runs the tutor agent against them, results.json, and the know-you chart.

Usage
  python sim.py --mock          # test now, no API key needed (stand-in agent)
  python sim.py                 # real run, uses Person A's tutor_agent
  python sim.py --plot-only     # redraw the chart from an existing results.json

Options: --rounds 20  --seed 42  --students sam maya  --sleep 4
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

from tutor_agent import choose_presentation

STYLES = ["steps", "analogy", "example"]
NOISE = 0.15          # 15% of the time a student picks one of the other styles
WINDOW = 5            # rolling window for the know-you score
# A random guesser hits the pick 1/3 of the time and the correctness 1/2 of the time.
RANDOM_BASELINE = (1 / 3 + 1 / 2) / 2

RESULTS_FILE = Path("results.json")
CHART_FILE = Path("know_you_score.png")
QUESTIONS_FILE = Path("questions.json")


# ---------------------------------------------------------------------------
# Simulated students (the agent never sees these rules)
# ---------------------------------------------------------------------------

#: How often a student actually reads the explanation the tutor leads with.
#: A lead is a steer, not a force -- the other explanations stay on screen --
#: so this is deliberately not 1.0. If it were, "lead" would just be "probe"
#: with a nicer name and the pick would stop being a real choice.
P_TAKES_LEAD = 0.75


class SimStudent:
    def __init__(self, name, favourite, learns_from, p_correct_if_helped, p_correct_otherwise, seed):
        self.name = name
        self.favourite = favourite              # what they usually click
        self.learns_from = learns_from          # what actually helps them
        self.p_help = p_correct_if_helped
        self.p_other = p_correct_otherwise
        self.rng = random.Random(seed)          # own RNG, so runs are repeatable

    def pick(self):
        # favourite=None means no preference at all: uniform over the styles.
        # There is nothing for the agent to read, which is the point of iris.
        if self.favourite is None:
            return self.rng.choice(STYLES)
        if self.rng.random() < NOISE:
            return self.rng.choice([s for s in STYLES if s != self.favourite])
        return self.favourite

    def respond_to_lead(self, led_style):
        """What they actually read when the tutor opens with ``led_style``."""
        if self.rng.random() < P_TAKES_LEAD:
            return led_style
        return self.pick()

    def answer_check(self, pick):
        p = self.p_help if pick == self.learns_from else self.p_other
        return self.rng.random() < p


def make_students(seed):
    """The cohort. Every rule here is hidden from the agent.

    Two students is enough to tell the story but thin as evidence: 40 rounds,
    one mismatch case, and a chart that lives or dies on a handful of coin
    flips. The extra eight widen the cohort so the know-you score is measured
    across preferences the agent has to discover separately -- including
    students who have no preference to read, and students who are simply bad
    at the maths regardless of what they are shown.
    """
    return {
        # --- the two the pitch is built on ----------------------------------
        # Easy case: what he likes is what helps.
        "sam": SimStudent("sam", favourite="example", learns_from="example",
                          p_correct_if_helped=0.80, p_correct_otherwise=0.40, seed=seed + 1),
        # Killer demo: picks steps 85% of the time, but only learns from worked examples.
        "maya": SimStudent("maya", favourite="steps", learns_from="example",
                           p_correct_if_helped=0.80, p_correct_otherwise=0.30, seed=seed + 2),

        # --- more mismatches, in different directions -----------------------
        # Mismatch the other way round: loves the story, needs the procedure.
        "leo": SimStudent("leo", favourite="analogy", learns_from="steps",
                          p_correct_if_helped=0.75, p_correct_otherwise=0.30, seed=seed + 3),
        # Reaches for the worked example, but only the analogy makes it click.
        "noah": SimStudent("noah", favourite="example", learns_from="analogy",
                           p_correct_if_helped=0.78, p_correct_otherwise=0.35, seed=seed + 4),

        # --- second easy case, so "preference == benefit" is not one student -
        "ava": SimStudent("ava", favourite="steps", learns_from="steps",
                          p_correct_if_helped=0.82, p_correct_otherwise=0.40, seed=seed + 5),

        # --- the awkward ones -----------------------------------------------
        # No favourite at all: picks uniformly at random. There is no
        # preference signal to read, so the agent can only ever learn the
        # benefit half. Its pick accuracy here should sit at the 1/3 baseline
        # and that is the correct answer, not a failure.
        "iris": SimStudent("iris", favourite=None, learns_from="steps",
                           p_correct_if_helped=0.75, p_correct_otherwise=0.35, seed=seed + 6),
        # Struggling across the board: nothing helps much. A profile that
        # claims to have found her key style is overfitting noise.
        "omar": SimStudent("omar", favourite="analogy", learns_from="analogy",
                           p_correct_if_helped=0.55, p_correct_otherwise=0.25, seed=seed + 7),
        # Strong and steady: gets it right whatever you show her, so the
        # benefit half carries almost no information.
        "zoe": SimStudent("zoe", favourite="steps", learns_from="example",
                          p_correct_if_helped=0.90, p_correct_otherwise=0.80, seed=seed + 8),
    }


#: The pair the 3-minute pitch uses. Everything else is for the wider chart.
HEADLINE_STUDENTS = ["sam", "maya"]


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

FALLBACK_QUESTIONS = [
    "Solve 2x + 5 = 13", "Solve 3x - 4 = 11", "Solve 5x + 2 = 27", "Solve 4x - 7 = 9",
    "Solve 6x + 1 = 25", "Solve 2(x + 3) = 14", "Solve 7x - 5 = 2x + 10",
    "Solve x/3 + 4 = 9", "Solve 3(2x - 1) = 21", "Solve 5x + 4 = 3x + 12",
]


def load_questions():
    if QUESTIONS_FILE.exists():
        data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
        return [q["question"] for q in data]
    print("questions.json not found yet, using 10 built-in linear equations.")
    return FALLBACK_QUESTIONS


# ---------------------------------------------------------------------------
# Agent: the real one (Person A) or a stand-in for testing
# ---------------------------------------------------------------------------

class MockAgent:
    """Stand-in for tutor_agent so the loop and chart can be built before
    integration. It just counts what it has seen. Not used in the real demo."""

    def __init__(self):
        self.picks = {s: 0 for s in STYLES}
        self.right = {s: 0 for s in STYLES}

    def observe(self, pick, correct):
        self.picks[pick] += 1
        self.right[pick] += int(correct)

    def predict(self, profile_notes, question):
        if sum(self.picks.values()) == 0:
            return {"pick": "example", "correct_after_pick": True, "reason": "No data yet."}
        pick = max(self.picks, key=self.picks.get)
        rate = self.right[pick] / self.picks[pick]
        return {"pick": pick, "correct_after_pick": rate >= 0.5,
                "reason": f"Picks {pick} most often; right {rate:.0%} of the time after it."}

    def update(self, profile_notes, prediction, actual):
        parts = []
        for s in STYLES:
            if self.picks[s]:
                parts.append(f"after {s}: right {self.right[s]}/{self.picks[s]}")
        fav = max(self.picks, key=self.picks.get)
        return f"Usually picks {fav}. " + "; ".join(parts) + "."


def load_agent(use_mock):
    if use_mock:
        print("Using the MOCK agent (no Gemini calls).")
        return MockAgent()
    try:
        import tutor_agent
    except ImportError as e:
        raise SystemExit(f"Could not import tutor_agent ({e}). "
                         "Run with --mock until Person A's package is in this folder.")
    # The helpers may live in tutor_agent/__init__.py or tutor_agent/agent.py.
    mod = tutor_agent if hasattr(tutor_agent, "predict") else __import__("tutor_agent.agent", fromlist=["x"])
    print("Using the real tutor_agent (Gemini).")
    return mod


def choose_intervention(history, every=3):
    """Every ``every`` rounds, force the style we have least evidence about.

    Without this the simulation cannot show what it claims to. Maya picks
    "steps" ~85% of the time, so across twenty rounds the agent may never once
    observe what a worked example does for her -- and "examples are what
    actually help Maya" is the single fact the whole pitch rests on. You
    cannot learn a rule from evidence that was never generated; a passive
    observer of a student who only does one thing learns only that one thing.

    So the tutor periodically stops observing and runs a small experiment: it
    shows the style it knows least about and watches what happens. That is the
    difference between a recommender and an agent.

    Measured over 9 live rounds, this took Maya's profile from inventing a
    false pattern ("struggles with negative signs") to finding the real one
    ("prefers steps, learns from examples").

    Returns None on ordinary rounds, where the student chooses freely.
    """
    if every <= 0 or (len(history) + 1) % every != 0:
        return None
    seen = {s: 0 for s in STYLES}
    for row in history:
        if row.get("actual_pick") in seen:
            seen[row["actual_pick"]] += 1
    # Fewest observations first; STYLES order breaks ties, so a rerun with the
    # same seed forces exactly the same styles in the same rounds.
    return min(STYLES, key=lambda s: (seen[s], STYLES.index(s)))


def call_predict(agent, notes, question, forced):
    """Predict, passing the forced style only to agents that accept one.

    tutor_agent.predict takes it; MockAgent does not. Rather than require both
    to move in lockstep, ask for the richer call and fall back.
    """
    if forced is None:
        return agent.predict(notes, question)
    try:
        return agent.predict(notes, question, forced)
    except TypeError:
        return agent.predict(notes, question)


def with_retry(fn, *args, tries=4, wait=5):
    for attempt in range(1, tries + 1):
        try:
            return fn(*args)
        except Exception as e:  # rate limits, network blips, bad JSON
            if attempt == tries:
                raise
            print(f"  call failed ({type(e).__name__}: {e}); retrying in {wait}s")
            time.sleep(wait)
            wait *= 2


def normalise_style(value):
    v = str(value).strip().lower()
    if "example" in v:
        return "example"
    if "analog" in v:
        return "analogy"
    if "step" in v:
        return "steps"
    return v


def to_dict(obj):
    """Accept a dict, a Pydantic model or a JSON string from predict()."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, str):
        cleaned = re.sub(r"^```(json)?|```$", "", obj.strip(), flags=re.M).strip()
        return json.loads(cleaned)
    return dict(obj)


def to_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def to_notes(obj):
    """Accept a string or a dict like {"notes": ...} / {"profile": ...} from update()."""
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    if isinstance(obj, dict):
        return obj.get("notes") or obj.get("profile") or json.dumps(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run(student_names, rounds, seed, use_mock, sleep, intervene_every=3, exploit=True):
    """Run to completion, printing as it goes. The CLI entry point."""
    log = []
    for row in stream(student_names, rounds, seed, use_mock, sleep, intervene_every,
                      exploit=exploit):
        log = row["_log"]
        print(f"r{row['round']:2d} predicted {row['pred_pick']:7s}/"
              f"{'right' if row['pred_correct'] else 'wrong':5s} "
              f"actual {row['actual_pick']:7s}/"
              f"{'right' if row['actual_correct'] else 'wrong':5s} "
              f"{'--  ' if row['pick_hit'] is None else ('HIT ' if row['pick_hit'] else 'miss')} "
              f"{'HIT ' if row['correct_hit'] else 'miss'}"
              f"{'  [forced ' + row['forced_style'] + ']' if row['forced_style'] else ''}")
    print(f"\nSaved {len(log)} rounds to {RESULTS_FILE}")
    return log


def stream(student_names, rounds, seed, use_mock, sleep, intervene_every=3,
           results_file=None, exploit=True):
    """Yield one scored round at a time, as it happens.

    A generator rather than a function that returns at the end, so a UI can
    draw the know-you score climbing round by round instead of staring at a
    spinner for twenty minutes. Each yielded row carries the running log
    under "_log" so a caller can redraw the whole chart without keeping its
    own copy.
    """
    results_file = results_file or RESULTS_FILE
    questions = load_questions()
    students = make_students(seed)
    log = []
    real_agent = load_agent(use_mock)

    for name in student_names:
        student = students[name]
        # The real agent's memory is the profile notes; the mock needs a fresh copy per student.
        agent = MockAgent() if use_mock else real_agent
        profile = {"student": name, "notes": "New student. Nothing known yet."}
        print(f"\n=== {name} ===")

        for r in range(1, rounds + 1):
            question = questions[(r - 1) % len(questions)]

            # 0. How should this round be presented? Three outcomes:
            #    probe  - force the least-evidenced style (buy information)
            #    lead   - open with the style the evidence says HELPS them,
            #             the other explanations still available (the payoff)
            #    choose - show everything, no steer
            own = [row for row in log if row["student"] == name]
            if exploit:
                mode, style = choose_presentation(
                    own, probe_every=intervene_every
                )
            else:
                mode = "probe" if choose_intervention(own, intervene_every) else "choose"
                style = choose_intervention(own, intervene_every)
            forced = style if mode == "probe" else None
            led = style if mode == "lead" else None

            # 1. Sealed prediction, made before the student does anything.
            pred = to_dict(with_retry(call_predict, agent, profile["notes"],
                                      question, forced))
            pred_pick = normalise_style(pred.get("pick"))
            pred_correct = to_bool(pred.get("correct_after_pick"))

            # 2. Student acts (hidden rules). On an intervention round the
            #    tutor overrides the choice, so what they READ is the forced
            #    style -- and correctness must follow what they read, not what
            #    they would have picked. Get this backwards and the experiment
            #    measures nothing.
            if forced:
                actual_pick = forced          # no choice offered
            elif led:
                actual_pick = student.respond_to_lead(led)   # steered, not forced
            else:
                actual_pick = student.pick()
            actual_correct = student.answer_check(actual_pick)

            # 3. Compare. On a forced round the pick was never predicted, so
            #    pick_hit is None = NOT SCORED. It must be excluded from the
            #    accuracy, never counted as a miss, or the chart dips every
            #    third round for no reason.
            pick_hit = None if forced else pred_pick == actual_pick
            correct_hit = pred_correct == actual_correct

            if hasattr(agent, "observe"):
                agent.observe(actual_pick, actual_correct)

            # 4. Rewrite the profile when the prediction was wrong -- and also
            #    on a correct intervention round. Rewriting only on failure
            #    throws away every piece of CONFIRMING evidence, and on an
            #    intervention round that is exactly the evidence we spent a
            #    round going out to collect: force Maya to read an example,
            #    she gets it right, the agent predicted she would, and because
            #    that is a hit the profile never records the one fact the demo
            #    is about.
            missed = (pick_hit is False) or (correct_hit is False)
            confirming = forced is not None and not missed
            if missed or confirming:
                prediction = {"pick": pred_pick, "correct_after_pick": pred_correct,
                              "reason": pred.get("reason", "")}
                actual = {"pick": actual_pick, "correct": actual_correct}
                new_notes = with_retry(agent.update, profile["notes"], prediction, actual)
                profile["notes"] = to_notes(new_notes)

            row = {
                "round": r, "student": name,
                "forced_style": forced,
                "led_style": led,
                "mode": mode,
                "pred_pick": pred_pick, "pred_correct": pred_correct,
                "actual_pick": actual_pick, "actual_correct": actual_correct,
                "pick_hit": pick_hit, "correct_hit": correct_hit,
                "profile_after": profile["notes"],
            }
            log.append(row)
            # Save every round, so a crash or rate limit never loses the run.
            results_file.write_text(json.dumps(log, indent=2), encoding="utf-8")

            yield {**row, "_log": log}

            if sleep and not use_mock:
                time.sleep(sleep)
    return log


# ---------------------------------------------------------------------------
# Know-you score + chart
# ---------------------------------------------------------------------------

def scored_halves(row):
    """The halves of this round's prediction that were actually scored.

    ``pick_hit is None`` means an intervention round: the tutor chose the
    style, so the pick was never predicted and must be EXCLUDED rather than
    counted as a miss. Test for None explicitly -- False means "predicted and
    wrong", which still counts.
    """
    halves = []
    if row.get("pick_hit") is not None:
        halves.append(bool(row["pick_hit"]))
    if row.get("correct_hit") is not None:
        halves.append(bool(row["correct_hit"]))
    return halves


def know_you_scores(rows):
    """Rolling accuracy over the last WINDOW rounds, counted in scored halves.

    Averaged over halves rather than rounds, so a round contributing one half
    does not silently weigh as much as a round contributing two.
    """
    rolling = []
    for i in range(len(rows)):
        window = rows[max(0, i - WINDOW + 1): i + 1]
        halves = [h for row in window for h in scored_halves(row)]
        rolling.append(sum(halves) / len(halves) if halves else 0.0)
    return rolling


def summarise(log):
    for name in dict.fromkeys(r["student"] for r in log):
        rows = [r for r in log if r["student"] == name]
        scores = know_you_scores(rows)
        picks = [r["pick_hit"] for r in rows if r.get("pick_hit") is not None]
        pick_acc = sum(picks) / len(picks) if picks else 0.0
        corr_acc = sum(r["correct_hit"] for r in rows) / len(rows)
        forced_n = sum(1 for r in rows if r.get("forced_style"))
        print(f"{name:5s} pick accuracy {pick_acc:.0%} ({len(picks)} scored, "
              f"{forced_n} forced excluded), correctness accuracy {corr_acc:.0%}, "
              f"know-you score round {WINDOW}: {scores[min(WINDOW, len(scores)) - 1]:.2f} "
              f"-> last round: {scores[-1]:.2f}")
        print(f"      final profile: {rows[-1]['profile_after']}")


def plot(log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"sam": "#2a9d8f", "maya": "#6a4c93", "leo": "#e76f51",
               "noah": "#457b9d", "ava": "#b5179e", "iris": "#f4a261",
               "omar": "#8d99ae", "zoe": "#386641"}
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    names = list(dict.fromkeys(r["student"] for r in log))
    series = {n: know_you_scores([r for r in log if r["student"] == n]) for n in names}

    # Eight students drawn with equal weight is spaghetti: every line crosses
    # every other one and a judge at the back of the room reads nothing. Past
    # three, push the individuals into the background and let one bold cohort
    # mean carry the claim -- "across the whole cohort the agent beats random"
    # is the point, and the individual lines are the evidence behind it.
    crowded = len(names) > 3
    for name in names:
        rows = [r for r in log if r["student"] == name]
        x = [r["round"] for r in rows]
        if crowded:
            ax.plot(x, series[name], linewidth=1.2, alpha=0.30,
                    color=colours.get(name), label=name.capitalize(), zorder=2)
        else:
            ax.plot(x, series[name], marker="o", linewidth=2.5,
                    color=colours.get(name), label=name.capitalize(), zorder=3)

    if crowded:
        longest = max(len(v) for v in series.values())
        mean = [
            sum(v[i] for v in series.values() if i < len(v))
            / sum(1 for v in series.values() if i < len(v))
            for i in range(longest)
        ]
        ax.plot(range(1, longest + 1), mean, linewidth=3.5, color="#1d3557",
                marker="o", markersize=4, zorder=4,
                label=f"Cohort average ({len(names)} students)")

    ax.axhline(RANDOM_BASELINE, linestyle="--", color="#c1121f", linewidth=2,
               zorder=5, label=f"Random guessing ({RANDOM_BASELINE:.2f})")
    ax.set_ylim(0, 1.05)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Know-you score (rolling {WINDOW} rounds)")
    ax.set_title("How well the agent knows each student", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(CHART_FILE)
    print(f"Saved chart to {CHART_FILE}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Simulated students for the adaptive tutor.")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--students", nargs="+", default=HEADLINE_STUDENTS,
                   choices=sorted(make_students(0)) + ["all"],
                   help="which simulated students to run, or 'all' for the whole cohort")
    p.add_argument("--no-exploit", action="store_true",
                   help="never lead with the helpful style; only ever show all "
                        "three. This is the BEFORE case: the agent learns the "
                        "student and then does nothing with it.")
    p.add_argument("--intervene-every", type=int, default=3, metavar="N",
                   help="force the least-evidenced style every N rounds (0 disables). "
                        "Without this the agent may never observe what actually "
                        "helps a student with a strong preference.")
    p.add_argument("--mock", action="store_true", help="use the stand-in agent, no Gemini calls")
    p.add_argument("--sleep", type=float, default=4.0, help="seconds between rounds (real agent only)")
    p.add_argument("--plot-only", action="store_true", help="redraw the chart from results.json")
    args = p.parse_args()

    if args.plot_only:
        log = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    else:
        names = (sorted(make_students(0)) if "all" in args.students
                 else list(dict.fromkeys(args.students)))
        log = run(names, args.rounds, args.seed, args.mock, args.sleep,
                  args.intervene_every, exploit=not args.no_exploit)

    summarise(log)
    plot(log)


if __name__ == "__main__":
    main()
