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

class SimStudent:
    def __init__(self, name, favourite, learns_from, p_correct_if_helped, p_correct_otherwise, seed):
        self.name = name
        self.favourite = favourite              # what they usually click
        self.learns_from = learns_from          # what actually helps them
        self.p_help = p_correct_if_helped
        self.p_other = p_correct_otherwise
        self.rng = random.Random(seed)          # own RNG, so runs are repeatable

    def pick(self):
        if self.rng.random() < NOISE:
            return self.rng.choice([s for s in STYLES if s != self.favourite])
        return self.favourite

    def answer_check(self, pick):
        p = self.p_help if pick == self.learns_from else self.p_other
        return self.rng.random() < p


def make_students(seed):
    return {
        # Easy case: what she likes is what helps.
        "sam": SimStudent("sam", favourite="example", learns_from="example",
                          p_correct_if_helped=0.80, p_correct_otherwise=0.40, seed=seed + 1),
        # Killer demo: picks steps 85% of the time, but only learns from worked examples.
        "maya": SimStudent("maya", favourite="steps", learns_from="example",
                           p_correct_if_helped=0.80, p_correct_otherwise=0.30, seed=seed + 2),
    }


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

def run(student_names, rounds, seed, use_mock, sleep):
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

            # 1. Sealed prediction, made before the student does anything.
            pred = to_dict(with_retry(agent.predict, profile["notes"], question))
            pred_pick = normalise_style(pred.get("pick"))
            pred_correct = to_bool(pred.get("correct_after_pick"))

            # 2. Student acts (hidden rules).
            actual_pick = student.pick()
            actual_correct = student.answer_check(actual_pick)

            # 3. Compare.
            pick_hit = pred_pick == actual_pick
            correct_hit = pred_correct == actual_correct

            if hasattr(agent, "observe"):
                agent.observe(actual_pick, actual_correct)

            # 4. Rewrite the profile only when the prediction was wrong.
            if not (pick_hit and correct_hit):
                prediction = {"pick": pred_pick, "correct_after_pick": pred_correct,
                              "reason": pred.get("reason", "")}
                actual = {"pick": actual_pick, "correct": actual_correct}
                new_notes = with_retry(agent.update, profile["notes"], prediction, actual)
                profile["notes"] = to_notes(new_notes)

            row = {
                "round": r, "student": name,
                "pred_pick": pred_pick, "pred_correct": pred_correct,
                "actual_pick": actual_pick, "actual_correct": actual_correct,
                "pick_hit": pick_hit, "correct_hit": correct_hit,
                "profile_after": profile["notes"],
            }
            log.append(row)
            # Save every round, so a crash or rate limit never loses the run.
            RESULTS_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")

            print(f"r{r:2d} predicted {pred_pick:7s}/{'right' if pred_correct else 'wrong':5s} "
                  f"actual {actual_pick:7s}/{'right' if actual_correct else 'wrong':5s} "
                  f"{'HIT ' if pick_hit else 'miss'} {'HIT ' if correct_hit else 'miss'}")

            if sleep and not use_mock:
                time.sleep(sleep)

    print(f"\nSaved {len(log)} rounds to {RESULTS_FILE}")
    return log


# ---------------------------------------------------------------------------
# Know-you score + chart
# ---------------------------------------------------------------------------

def know_you_scores(rows):
    """Per round: average of the two hits (0, 0.5 or 1), then a rolling mean over WINDOW rounds."""
    per_round = [(r["pick_hit"] + r["correct_hit"]) / 2 for r in rows]
    rolling = []
    for i in range(len(per_round)):
        window = per_round[max(0, i - WINDOW + 1): i + 1]
        rolling.append(sum(window) / len(window))
    return rolling


def summarise(log):
    for name in dict.fromkeys(r["student"] for r in log):
        rows = [r for r in log if r["student"] == name]
        scores = know_you_scores(rows)
        pick_acc = sum(r["pick_hit"] for r in rows) / len(rows)
        corr_acc = sum(r["correct_hit"] for r in rows) / len(rows)
        print(f"{name:5s} pick accuracy {pick_acc:.0%}, correctness accuracy {corr_acc:.0%}, "
              f"know-you score round {WINDOW}: {scores[min(WINDOW, len(scores)) - 1]:.2f} "
              f"-> last round: {scores[-1]:.2f}")
        print(f"      final profile: {rows[-1]['profile_after']}")


def plot(log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"sam": "#2a9d8f", "maya": "#6a4c93"}
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    for name in dict.fromkeys(r["student"] for r in log):
        rows = [r for r in log if r["student"] == name]
        x = [r["round"] for r in rows]
        ax.plot(x, know_you_scores(rows), marker="o", linewidth=2.5,
                color=colours.get(name), label=name.capitalize())

    ax.axhline(RANDOM_BASELINE, linestyle="--", color="grey", linewidth=1.5,
               label=f"Random guessing ({RANDOM_BASELINE:.2f})")
    ax.set_ylim(0, 1.05)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Know-you score (rolling {WINDOW} rounds)")
    ax.set_title("How well the agent knows each student")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHART_FILE)
    print(f"Saved chart to {CHART_FILE}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Simulated students for the adaptive tutor.")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--students", nargs="+", default=["sam", "maya"], choices=["sam", "maya"])
    p.add_argument("--mock", action="store_true", help="use the stand-in agent, no Gemini calls")
    p.add_argument("--sleep", type=float, default=4.0, help="seconds between rounds (real agent only)")
    p.add_argument("--plot-only", action="store_true", help="redraw the chart from results.json")
    args = p.parse_args()

    if args.plot_only:
        log = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    else:
        log = run(args.students, args.rounds, args.seed, args.mock, args.sleep)

    summarise(log)
    plot(log)


if __name__ == "__main__":
    main()
