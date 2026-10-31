# The Tutor That Learns The Student

Adaptive GCSE maths tutor on Google ADK + Gemini.

The tutor shows **3 explanations** of each problem (`steps` / `analogy` / `example`). Before the student
touches anything, an agent makes a **sealed prediction**: which explanation they will pick, and whether
they will then get the **check question** right. Wrong prediction -> the agent rewrites the student's
profile. The demo metric is a **know-you score** climbing over 20 rounds.

**This README is the integration contract. Person B (simulator) and Person C (UI) both import the same
package. Read the two rules below before you write any scoring code.**

---

## THE TWO NON-NEGOTIABLE RULES

### 1. The predictor is blind

`predict()` sees **only** the profile notes and the question text. It must never see:

- the correct answer
- which explanation the student picked
- whether the student got it right

**Why:** the whole claim of this project is that a written profile predicts behaviour it has not seen.
Leak any of the above into the prediction and the score measures nothing — you are just reading the
answer back. Do not "help" the predictor with extra context.

### 2. Score each round exactly once, from the first blind prediction

Order is fixed: **predict (sealed) -> student acts -> score ONCE -> log -> then update if wrong.**

The profile-repair loop runs **after** the score is logged. **Never recompute the score after
`update()`.** That grades against an answer you already know and makes the know-you score meaningless.
One round = one score, and it comes from the prediction made before the student moved.

---

## Setup

```bash
# venv already exists: .venv (Python 3.14, google-adk 2.13.0 installed)
# Free key: https://aistudio.google.com/apikey
echo "GOOGLE_API_KEY=your_key_here" > .env

.venv/bin/python your_script.py     # always run via .venv/bin/python
```

`adk web` launches the **agent inspector** — every agent call and its raw output, in a browser.
That is the judge-facing reasoning trace. Have it open during the pitch.

---

## Public API

```python
from tutor_agent import generate_explanations, predict, update, normalise_answer

generate_explanations(question: str) -> dict
# {"steps": "...", "analogy": "...", "example": "..."}

predict(profile_notes: str, question: str) -> dict
# {"pick": "steps"|"analogy"|"example", "correct_after_pick": bool, "reason": "..."}

update(profile_notes: str, prediction: dict, actual: dict) -> str
# returns NEW profile notes (a string). Call only when the prediction missed.

normalise_answer(s: str) -> str
# "x = 4" -> "4",  " 4 " -> "4"
```

### Always compare answers with `normalise_answer()`

```python
correct = normalise_answer(typed) == normalise_answer(q["answer"])   # YES
correct = typed == q["answer"]                                       # NO
```

A judge in the live demo will type `x=4` or `4 ` and a raw `==` marks them wrong when they were right.
Use it in the simulator **and** in the UI.

---

## Data formats

### `questions.json` — 10 linear equations, difficulty ordered

```json
[{"id":1,"question":"Solve 2x+5=13","answer":"4",
  "check":{"question":"Solve 3x+2=14","answer":"4"},
  "explanations":{"steps":"...","analogy":"...","example":"..."}}]
```

- `answer` is a **string, bare number**: `"4"`, `"-3"`, `"2.5"`. No `x=` prefix.
- `check` tests the **same skill, different numbers**. It is how we measure whether the explanation
  actually helped. Do not swap in a check from a different question.
- `explanations` are hand-written fallbacks, all verified under 80 words. Use them when
  `generate_explanations()` is unavailable — the demo then needs **zero API calls**.

### profile

```json
{"student": "maya", "notes": "text the agent writes about the student"}
```

`notes` is free text written by the agent. Never parse it, never template it — pass it through.

### round log (one record per round)

```json
{"round":1,"student":"maya","pred_pick":"steps","pred_correct":false,
 "actual_pick":"steps","actual_correct":false,"pick_hit":true,"correct_hit":true,
 "profile_after":"..."}
```

| field | meaning |
|---|---|
| `pred_pick`, `pred_correct` | the sealed prediction |
| `actual_pick`, `actual_correct` | what the student did |
| `pick_hit` | `pred_pick == actual_pick` |
| `correct_hit` | `pred_correct == actual_correct` |
| `profile_after` | notes after the repair loop (unchanged if prediction was right) |

**know-you score** = running mean of `pick_hit` and `correct_hit`. Both computed once, before `update()`.

---

## One round — copy-paste this (Person B)

```python
import json
from tutor_agent import predict, update, normalise_answer

questions = json.load(open("questions.json"))
profile = {"student": "maya", "notes": "No data yet."}
log = []

for i, q in enumerate(questions, start=1):
    # 1. SEALED PREDICTION — profile + question text only. No answer, no student action.
    pred = predict(profile["notes"], q["question"])

    # 2. STUDENT ACTS (simulated student, or a real click in the UI)
    pick = student_picks(q)                       # "steps" | "analogy" | "example"
    typed = student_answers(q, pick)              # their answer to q["check"]["question"]
    actual_correct = normalise_answer(typed) == normalise_answer(q["check"]["answer"])
    actual = {"pick": pick, "correct_after_pick": actual_correct}

    # 3. SCORE ONCE — from the sealed prediction, before any profile change
    pick_hit = pred["pick"] == actual["pick"]
    correct_hit = pred["correct_after_pick"] == actual["correct_after_pick"]

    # 4. LOG, then repair only if the prediction missed
    notes_before = profile["notes"]
    if not (pick_hit and correct_hit):
        profile["notes"] = update(notes_before, pred, actual)   # ONE repair call

    log.append({"round": i, "student": profile["student"],
                "pred_pick": pred["pick"], "pred_correct": pred["correct_after_pick"],
                "actual_pick": actual["pick"], "actual_correct": actual["correct_after_pick"],
                "pick_hit": pick_hit, "correct_hit": correct_hit,
                "profile_after": profile["notes"]})
    # NEVER re-score here. The round is closed.
```

20 rounds = the 10 questions twice (second pass is where the score should visibly climb).

---

## The two simulated students (Person B implements these)

| Student | Picks | Actually learns from | Purpose |
|---|---|---|---|
| **Sam** | worked `example` | `example` -> 80% correct, else 40% | easy case: preference = what helps |
| **Maya** | `steps` (85%) | **ONLY** `example` -> 80% correct, else 30% | killer demo: picks != what helps |

Plus **15% random noise on picks** for both — they occasionally click something else. The noise is what
stops the agent from locking onto a trivial rule, and what the know-you score has to survive.

Maya is the pitch. She always reaches for `steps`, `steps` never helps her, and the agent has to write
that contradiction into her profile before its predictions improve.

```python
import random
def student_picks(q, student="maya"):
    pref = "example" if student == "sam" else "steps"
    if random.random() < 0.15:                      # 15% noise
        return random.choice(["steps", "analogy", "example"])
    if student == "maya" and random.random() > 0.85:  # Maya's pick is 85% steps
        return random.choice(["analogy", "example"])
    return pref
```

Seed your RNG (`random.seed(7)`) so the demo is reproducible.

---

## Rate limits — read this at minute 0, not minute 85

Free tier. A full 20-round simulation with the repair loop is **~140 Gemini calls**.

- **Run the full simulation EARLY.** Not 5 minutes before the deadline. If you hit the limit at 1:25
  you have no demo.
- **Cache the run.** Dump `log` to `run.json` and have the UI read that file. The pitch should replay a
  saved run, not call the API live.
- Skip `generate_explanations()` during the simulation — use the `explanations` already in
  `questions.json`. That cuts roughly a third of the calls for free.
- Only call `update()` when the prediction actually missed. Updating on a hit wastes quota and
  corrupts the profile.

---

## Difficulty ramp in `questions.json`

| id | equation | answer | new skill |
|---|---|---|---|
| 1 | `2x + 5 = 13` | 4 | two-step, positive |
| 2 | `5x - 7 = 18` | 5 | two-step with subtraction |
| 3 | `3x + 20 = 2` | -6 | negative solution |
| 4 | `7 - 2x = 1` | 3 | negative coefficient |
| 5 | `6x + 3 = 2x + 19` | 4 | x on both sides |
| 6 | `4x - 3 = 9 - 2x` | 2 | both sides, negative x term |
| 7 | `4x + 1 = 2x + 6` | 2.5 | non-integer answer |
| 8 | `3(x + 4) = 21` | 3 | brackets |
| 9 | `(x + 3)/4 = 5` | 17 | fraction / clear the denominator |
| 10 | `(3x + 1)/2 = 2(x - 1)` | 5 | fraction + brackets + both sides |

All 20 answers (10 main + 10 check) verified by exact rational arithmetic.

---

## Integration checklist (1:30 deadline)

- [ ] `.env` has `GOOGLE_API_KEY`
- [ ] `from tutor_agent import predict, update, normalise_answer` imports clean
- [ ] `predict()` is called **before** the student acts, and never receives `q["answer"]`
- [ ] score computed once per round, **before** `update()`
- [ ] every answer comparison goes through `normalise_answer()`
- [ ] UI reads `run.json`, not the live API
- [ ] `adk web` opens and shows the trace
