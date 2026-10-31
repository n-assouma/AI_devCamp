import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from preference_store import load_preference_store, save_preference_store
from tutor_agent import (
    answers_match,
    choose_intervention,
    choose_presentation,
    predict,
    recommend_style,
    update,
)
from tutor_provider import (
    STYLES,
    DemoTutor,
    ExplanationStyle,
    GeminiTutor,
    TutorProvider,
    TutorProviderError,
    validate_tutor_response,
)


ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results.json"
QUESTIONS_FILE = ROOT / "questions.json"
PREFERENCE_FILE = ROOT / "student_preferences.json"
STYLE_LABELS = {
    "steps": "Step by step",
    "analogy": "Analogy",
    "example": "Worked example",
}
# Each round scores up to two independent halves, with different odds of being
# right by luck: the pick is one style out of three, the outcome is right or
# wrong. The familiar 5/12 baseline is the average of the two, and it is only
# the correct baseline on a round where BOTH halves were actually scored -- so
# rolling_scores() derives the baseline from the halves in each window instead
# of assuming it.
PICK_BASELINE = 1 / 3
OUTCOME_BASELINE = 1 / 2
RANDOM_BASELINE = (PICK_BASELINE + OUTCOME_BASELINE) / 2
#: How often the tutor overrides the student's choice to run an experiment.
INTERVENTION_EVERY = 3
#: Observations of a style before the tutor will LEAD with it.
#:
#: This is a real trade-off, measured rather than guessed. At 2, a
#: recommendation forms within a handful of rounds -- good for a short live
#: demo -- but two observations cannot separate a style that helps 75% of the
#: time from one that helps 35%, and the tutor locks onto the wrong style and
#: then only ever gathers evidence about that one. On the simulated cohort
#: that made two of eight students WORSE off than no leading at all. At 4 the
#: recommendation was right for 7 of 8, at the cost of taking roughly a dozen
#: rounds to appear. Lower it for a short demo knowing what it buys.
MIN_EVIDENCE_TO_LEAD = 4
NO_PROFILE_YET = "No profile yet - this is the agent's first look at this student."


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read {path.name}: {error}")
        st.stop()


def is_scored_half(value: Any) -> bool:
    """True for a prediction half the record stores in a usable form.

    ``True``/``False`` mean the half was predicted and scored. ``None`` means
    it was never predicted -- the pick on a round where the tutor chose the
    style, or the outcome on a question with no check. Both are legal; a
    missing key is not.
    """
    return value is None or isinstance(value, bool)


def load_results() -> list[dict[str, Any]]:
    data = load_json(RESULTS_FILE)
    if not isinstance(data, list):
        st.error("results.json must contain a list of round records.")
        st.stop()
    required_booleans = ("pred_correct", "actual_correct", "correct_hit")
    for index, record in enumerate(data):
        valid = (
            isinstance(record, dict)
            and isinstance(record.get("round"), int)
            and isinstance(record.get("student"), str)
            and record.get("pred_pick") in STYLES
            and record.get("actual_pick") in STYLES
            and all(isinstance(record.get(field), bool) for field in required_booleans)
            # pick_hit is nullable: on an intervention round the tutor picked
            # the style, so the pick was never predicted.
            and "pick_hit" in record
            and is_scored_half(record["pick_hit"])
            and isinstance(record.get("profile_after"), str)
        )
        if not valid:
            st.error(f"Round record {index + 1} does not match the results.json format.")
            st.stop()
    return data


def scored_halves(record: dict[str, Any]) -> list[tuple[bool, float]]:
    """The halves of one round that were actually predicted, with their odds.

    A half that is ``None`` was never predicted, so it is excluded here rather
    than counted as a miss. Counting it as a miss would drop the score every
    intervention round and make a working agent look broken.
    """
    return [
        (bool(record.get(field)), baseline)
        for field, baseline in (
            ("pick_hit", PICK_BASELINE),
            ("correct_hit", OUTCOME_BASELINE),
        )
        if record.get(field) is not None
    ]


def rolling_scores(records: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    scores = []
    for index, record in enumerate(records):
        window = records[max(0, index - 4) : index + 1]
        halves = [half for item in window for half in scored_halves(item)]
        if not halves:
            continue
        hits = sum(1 for hit, _ in halves if hit)
        scores.append(
            {
                "Round": record["round"],
                "Know-you score": hits / len(halves),
                # The baseline has to be the luck rate of the halves that were
                # really scored, or the chart compares the agent against odds
                # it was never offered.
                "Random baseline": sum(odds for _, odds in halves) / len(halves),
            }
        )
    return scores


def show_score_chart(records: list[dict[str, Any]]) -> None:
    rows = rolling_scores(records)
    if not rows:
        st.info("No scored rounds are available.")
        return
    score_data = pd.DataFrame(rows).set_index("Round")
    st.line_chart(score_data, y=["Know-you score", "Random baseline"])
    st.caption(
        "Rolling five-round accuracy across the prediction halves that were "
        f"actually scored. Random baseline is {RANDOM_BASELINE:.1%} where both "
        "halves count, 50.0% where only the outcome does (the tutor chose the "
        "style, so the pick was never predicted)."
    )


def load_student(student_id: str) -> dict[str, Any]:
    store = load_preference_store(PREFERENCE_FILE)
    student = store["students"].get(
        student_id,
        {"preferences": [], "rounds": [], "conversations": []},
    )
    student.setdefault("conversations", [])
    # The agent-written profile. preference_store validates only preferences,
    # rounds and conversations, so an extra key round-trips untouched.
    student.setdefault("notes", "")
    return student


def persist_student(student_id: str, student: dict[str, Any]) -> None:
    store = load_preference_store(PREFERENCE_FILE)
    store["students"][student_id] = student
    save_preference_store(store, PREFERENCE_FILE)


def profile_notes(student: dict[str, Any]) -> str:
    """The profile the agent has written about this student.

    This used to be a tally of clicks, which can only ever report what a
    student *chose* -- never what actually taught them, which is the whole
    claim of the product. These notes are written by the updater agent (see
    :func:`settle_round`) every time a sealed prediction was contradicted, so
    they hold rules about the student rather than counts of their clicks.
    """
    notes = str(student.get("notes") or "").strip()
    return notes or NO_PROFILE_YET


def preferred_style(preferences: list[dict[str, Any]]) -> ExplanationStyle | None:
    counts = {style: 0 for style in STYLES}
    for preference in preferences:
        style = preference.get("style")
        if style in counts:
            counts[style] += 1
    if not any(counts.values()):
        return None
    return max(STYLES, key=lambda style: (counts[style], -STYLES.index(style)))


def round_logs(student: dict[str, Any]) -> list[dict[str, Any]]:
    """Every scored round for this student, oldest first.

    Deliberately the same shape as a results.json record, so the live chart and
    the replay chart are one piece of code reading one set of fields.
    """
    return [
        turn["score"]
        for turn in student.get("conversations", [])
        if turn.get("score")
    ]


def question_bank() -> list[dict[str, Any]]:
    """The cached question bank, or an empty list if it cannot be read.

    Unlike :func:`load_json` this never stops the app: the bank only supplies
    answer keys, and the chat still works without them.
    """
    cached = st.session_state.get("question_bank")
    if cached is None:
        try:
            data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = []
        cached = [item for item in data if isinstance(item, dict)]
        st.session_state.question_bank = cached
    return cached


def squashed(text: Any) -> str:
    """Lowercase, whitespace-free form, for matching a typed question loosely."""
    return re.sub(r"[\s?.]+", "", str(text).lower())


def bank_entry(prompt: str) -> dict[str, Any] | None:
    """The bank entry the student just asked for, if the bank has one.

    Each bank entry pairs a question with a check question *and* its answer.
    When the student asks one of them the round can be marked against a real
    key, which is what makes the outcome half of the prediction evidence rather
    than self-report. A judge types "solve 2x+5=13" or "can you solve
    2x + 5 = 13?", so the match is loose on whitespace, case and punctuation.
    """
    asked = squashed(prompt)
    if not asked:
        return None
    for entry in question_bank():
        check = entry.get("check")
        question = squashed(entry.get("question", ""))
        if not question or not isinstance(check, dict):
            continue
        if check.get("question") and check.get("answer") and question in asked:
            return entry
    return None


def seal_prediction(
    notes: str,
    question: str,
    forced_style: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """The agent's sealed, blind forecast for one round.

    ``predict`` is handed the profile and the question text and nothing else:
    not the answer key, not the student's pick, not whether they got it right.
    It has no parameter for any of them, and smuggling them in through ``notes``
    would void every number in the demo.

    Returns ``(prediction, error_message)`` and never raises. A dead free-tier
    quota is the single most likely thing to happen on stage, and it has to cost
    the demo one caption rather than a traceback.
    """
    try:
        return predict(notes, question, forced_style), None
    except Exception as error:  # noqa: BLE001
        # tutor_agent.AgentCallError is the expected failure (dead quota, bad
        # key, unparseable model output). The broad catch is deliberate: during
        # a live demo any traceback in the UI is worse than a caption.
        return None, str(error)


def rewrite_profile(
    notes: str,
    prediction: dict[str, Any],
    actual: dict[str, Any],
    question: str,
) -> tuple[str, str | None]:
    """Ask the updater agent to rewrite the profile. Keeps ``notes`` on failure."""
    try:
        return update(notes, prediction, actual, question=question), None
    except Exception as error:  # noqa: BLE001 - see seal_prediction
        return notes, str(error)


def settle_round(
    turn: dict[str, Any],
    student_id: str,
    *,
    actual_correct: bool | None,
) -> None:
    """Score the sealed prediction exactly once, then learn only from a miss.

    The score is taken from the prediction that was sealed before the student
    saw anything, and it is frozen the moment it is written -- the ``score``
    guard below means a round can never be scored twice. It is never recomputed
    after the profile is rewritten: that would be grading the agent against an
    answer it had already been told.

    Either half can be ``None``, meaning "never predicted, so never scored":
    the pick on an intervention round, where the tutor chose the style; the
    outcome on a definition, which has no check question. ``None`` is excluded
    from the know-you score and is not a miss.

    The caller persists the student afterwards.
    """
    student = st.session_state.student_data
    prediction = turn.get("prediction")
    if prediction is None or turn.get("score"):
        return  # nothing was sealed, or this round is already scored

    actual_pick = turn.get("selected_style")
    pick_hit = (
        None if turn.get("forced_style") else prediction["pick"] == actual_pick
    )
    correct_hit = (
        None
        if actual_correct is None
        else bool(prediction["correct_after_pick"]) == bool(actual_correct)
    )
    round_number = len(round_logs(student)) + 1  # this round is not logged yet
    turn["score"] = {
        "round": round_number,
        "student": student_id,
        "forced_style": turn.get("forced_style"),
        "pred_pick": prediction["pick"],
        "pred_correct": bool(prediction["correct_after_pick"]),
        "actual_pick": actual_pick,
        "actual_correct": actual_correct,
        "pick_hit": pick_hit,
        "correct_hit": correct_hit,
    }

    if pick_hit is not False and correct_hit is not False:
        return  # the profile already explains this round; leave it alone
    if actual_correct is None:
        # The pick was wrong, but this round produced no outcome at all. update()
        # reports the outcome to the updater as a boolean, so calling it here
        # would invent a fact about the student. The profile we have stands.
        turn["update_skipped"] = True
        return

    notes_after, error = rewrite_profile(
        turn.get("profile_before") or profile_notes(student),
        prediction,
        {"pick": actual_pick, "correct": bool(actual_correct)},
        turn["prompt"],
    )
    turn["update_error"] = error
    if error is None:
        student["notes"] = notes_after
        turn["profile_after"] = notes_after
    st.session_state.student_data = student


def initialize_chat_state(student_id: str) -> None:
    if st.session_state.get("student_id") == student_id:
        return
    student = load_student(student_id)
    st.session_state.student_id = student_id
    st.session_state.student_data = student


def append_prompt(
    prompt: str,
    student_id: str,
    provider: TutorProvider,
) -> bool:
    student = st.session_state.student_data
    notes = profile_notes(student)
    # Every third round the tutor stops observing and runs an experiment: it
    # forces the style it has the least evidence about. A student with a strong
    # preference never generates evidence about the other two, and "what you
    # prefer is not what teaches you" cannot be found in choices alone.
    # ...and once there is enough evidence it stops asking and LEADS with the
    # style that actually helps this student. That is the payoff: a tutor that
    # learns you and then never changes what it does has not helped you.
    # "lead" still shows the other explanations, so the pick stays a real
    # choice and stays predictable; "probe" does not.
    mode, mode_style = choose_presentation(
        round_logs(student),
        probe_every=INTERVENTION_EVERY,
        min_evidence=MIN_EVIDENCE_TO_LEAD,
    )
    forced_style = mode_style if mode == "probe" else None
    led_style = mode_style if mode == "lead" else None
    try:
        # Always all three explanations: the student has to have a real choice
        # for the predicted pick to mean anything, so a saved preference no
        # longer short-circuits it.
        response = validate_tutor_response(provider(prompt, notes, None))
        missing_styles = [item for item in STYLES if item not in response["answers"]]
        if missing_styles:
            raise ValueError(
                f"Tutor response did not include required styles: {', '.join(missing_styles)}."
            )
    except (TutorProviderError, TypeError, ValueError) as error:
        st.session_state.last_provider_error = str(error)
        return False

    check_answer = None
    entry = None if response["is_definition"] else bank_entry(prompt)
    if entry:
        # Use the bank's check question, because it is the one that comes with
        # an answer. The key never reaches the predictor -- it is stored on the
        # turn and only read when the student's typed answer is marked.
        response["check_question"] = entry["check"]["question"]
        check_answer = str(entry["check"]["answer"])

    # ---- THE SEALED PREDICTION ---------------------------------------------
    # Made here, from the profile as it stood before this round, and before a
    # single explanation reaches the screen: this runs inside the "tutor is
    # thinking" status, and the turn is first rendered on the rerun that
    # follows. The ordering is structural, not a convention.
    prediction, prediction_error = seal_prediction(notes, prompt, forced_style)

    turn = {
        "id": str(uuid.uuid4()),
        "prompt": prompt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "response": response,
        "preferred_style": None,
        # On an intervention round the style is the tutor's decision, so there
        # is nothing for the student to choose.
        "selected_style": forced_style,
        "forced_style": forced_style,
        "led_style": led_style,
        "mode": mode,
        "check_answer": check_answer,
        "profile_before": notes,
        "prediction": prediction,
        "prediction_error": prediction_error,
        "score": None,
        "check_response": None,
    }
    student["conversations"].append(turn)
    st.session_state.student_data = student
    if forced_style and not response["check_question"]:
        # The tutor chose the style and there is no check question, so neither
        # half of this prediction can ever be observed. Settle it now so the
        # round reads as "not scored" instead of waiting forever for an action
        # the student is never asked to take.
        settle_round(turn, student_id, actual_correct=None)
    persist_student(student_id, student)
    return True


def queue_chat_prompt() -> None:
    if st.session_state.get("is_generating"):
        return
    prompt = str(st.session_state.get("chat_prompt", "")).strip()
    if prompt:
        st.session_state.pending_prompt = prompt
        st.session_state.is_generating = True
        st.session_state.failed_submission = None


def select_style(turn: dict[str, Any], style: ExplanationStyle, student_id: str) -> None:
    student = st.session_state.student_data
    turn["selected_style"] = style
    preference = {
        "id": str(uuid.uuid4()),
        "style": style,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "source": "student_choice",
        "conversation_id": turn["id"],
        "prompt": turn["prompt"],
        "answer_correct": None,
    }
    student["preferences"].append(preference)
    st.session_state.student_data = student
    # A definition has no check question, so the outcome half of the sealed
    # prediction can never be observed. The pick half just was, so score it now.
    if not turn["response"]["check_question"]:
        settle_round(turn, student_id, actual_correct=None)
    persist_student(student_id, student)
    st.rerun()


def save_check_response(
    turn: dict[str, Any],
    response_text: str,
    student_id: str,
    self_marked_correct: bool | None = None,
) -> None:
    student = st.session_state.student_data
    answer_text = response_text.strip()
    expected = turn.get("check_answer")
    if expected:
        # A live judge types "x=4", "4 " or "x = 4". answers_match normalises
        # all of them to the same thing, because "the demo marked me wrong" is
        # not a story anyone wants to tell on stage.
        actual_correct: bool | None = answers_match(answer_text, expected)
        graded_by = "answer_key"
    else:
        # A free-text question has no key, so the student marks their own work.
        actual_correct = self_marked_correct
        graded_by = "student" if actual_correct is not None else None
    turn["check_response"] = {
        "text": answer_text,
        "answered_at": datetime.now(timezone.utc).isoformat(),
        "correct": actual_correct,
        "graded_by": graded_by,
    }
    student["rounds"].append(
        {
            "conversation_id": turn["id"],
            "prompt": turn["prompt"],
            "check_question": turn["response"]["check_question"],
            "student_response": answer_text,
            "answered_at": turn["check_response"]["answered_at"],
            "correct": actual_correct,
        }
    )
    for preference in student["preferences"]:
        if preference.get("conversation_id") == turn["id"]:
            preference["answer_correct"] = actual_correct
    st.session_state.student_data = student
    settle_round(turn, student_id, actual_correct=actual_correct)
    persist_student(student_id, student)
    st.rerun()


def render_prediction(turn: dict[str, Any]) -> None:
    """Show the sealed prediction -- locked until the student has acted.

    Until the round is scored the student sees only that a prediction exists.
    Revealing the pick early would tell them what to click, and an agent that
    scores itself against a student it has just coached measures nothing.
    """
    prediction = turn.get("prediction")
    if prediction is None:
        if turn.get("prediction_error"):
            st.caption(
                "The prediction agent is unavailable, so this round is not "
                "scored. The tutor itself is unaffected."
            )
            with st.expander("What went wrong"):
                st.write(turn["prediction_error"])
        return

    score = turn.get("score")
    if not score:
        st.caption(
            "Sealed: the agent has already written down which explanation it "
            "thinks you will pick and whether it will help. It unlocks when "
            "you answer."
        )
        return

    with st.container(border=True):
        st.markdown("**Sealed prediction** · written before you saw anything")
        if score["pick_hit"] is None:
            st.markdown(
                "- Which explanation you would pick: *not predicted* - the "
                f"tutor chose {STYLE_LABELS[prediction['pick']].lower()} this "
                "round, so this half is not scored."
            )
        else:
            st.markdown(
                f"- Which explanation you would pick: "
                f"**{STYLE_LABELS[prediction['pick']]}** - "
                f"{'hit' if score['pick_hit'] else 'miss'}."
            )
        if score["correct_hit"] is None:
            st.markdown(
                "- Whether it would get you the right answer: *not scored* - "
                "this round had no check question."
            )
        else:
            st.markdown(
                "- Whether it would get you the right answer: "
                f"**{'yes' if score['pred_correct'] else 'no'}** - "
                f"{'hit' if score['correct_hit'] else 'miss'}."
            )
        if prediction.get("reason"):
            st.caption(f"Its reasoning: {prediction['reason']}")

        if turn.get("profile_after"):
            st.info("**Miss** - so the agent rewrote its profile of you:")
            st.markdown(f"> {turn['profile_after']}")
        elif turn.get("update_error"):
            st.warning(
                "**Miss** - but the profile could not be rewritten just now, "
                "so the one it already had stands."
            )
            with st.expander("What went wrong"):
                st.write(turn["update_error"])
        elif turn.get("update_skipped"):
            st.warning(
                "**Miss** on the pick - but this round had no check question, "
                "so there is no outcome to learn from and the profile is "
                "unchanged."
            )
        elif score["pick_hit"] is not None or score["correct_hit"] is not None:
            st.success(
                "**Hit** - the profile already explains you, so nothing was "
                "rewritten."
            )


def render_turn(turn: dict[str, Any], student_id: str) -> None:
    response = turn["response"]
    with st.chat_message("user"):
        st.markdown(turn["prompt"])

    with st.chat_message("assistant"):
        if turn.get("forced_style"):
            style = turn["forced_style"]
            st.caption(
                f"The tutor picked {STYLE_LABELS[style].lower()} for you this "
                "round to find out whether it helps. Your choice was not "
                "predicted, so it is not scored."
            )
        elif turn["preferred_style"]:
            style = turn["preferred_style"]
            st.caption(
                f"Using your saved preference: {STYLE_LABELS[style].lower()}."
            )
        elif turn["selected_style"]:
            style = turn["selected_style"]
            st.caption(f"You chose: {STYLE_LABELS[style].lower()}.")
        else:
            style = None

        if style:
            answer = response["answers"].get(style)
            if answer:
                st.markdown(answer)
            else:
                st.warning(f"The tutor response did not include a {style} explanation.")
                return
        else:
            led = turn.get("led_style")
            # Once the agent has worked out what actually helps this student,
            # it stops presenting three equal options and opens with that one.
            # A tutor that learns you and then behaves identically forever has
            # not helped you -- this is where the profile finally does
            # something. The other two stay one click away, so the choice is
            # still real and the predicted pick still means something.
            others = [s for s in STYLES if s != led] if led else list(STYLES)

            if led:
                st.markdown(
                    f"Your profile says **{STYLE_LABELS[led].lower()}** is what "
                    "actually helps you answer correctly - even if it is not "
                    "what you would have picked. Starting there:"
                )
                with st.container(border=True):
                    st.markdown(f"**{STYLE_LABELS[led]}**")
                    st.markdown(response["answers"][led])
                    if st.button(
                        "Use this style",
                        key=f"style_{turn['id']}_{led}",
                        type="primary",
                        width="stretch",
                    ):
                        select_style(turn, led, student_id)
                expander = st.expander("Show me the other two ways")
            else:
                st.markdown(
                    "Three ways of seeing this. Which version works best for you?"
                )
                expander = None

            container = expander if expander is not None else st.container()
            with container:
                cols = st.columns(len(others))
                for col, option_style in zip(cols, others):
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**{STYLE_LABELS[option_style]}**")
                            st.markdown(response["answers"][option_style])
                            if st.button(
                                "Use this style",
                                key=f"style_{turn['id']}_{option_style}",
                                width="stretch",
                            ):
                                select_style(turn, option_style, student_id)
            render_prediction(turn)
            return

        if response["is_definition"]:
            st.caption("This is a definition, so there is no knowledge-check question.")
        else:
            st.markdown(f"**Try this:** {response['check_question']}")
            if turn["check_response"]:
                st.markdown("**Your response**")
                st.write(turn["check_response"]["text"])
                correct = turn["check_response"].get("correct")
                if correct is None:
                    st.caption("Saved for your learning history; this response is not graded.")
                elif turn["check_response"].get("graded_by") == "answer_key":
                    st.caption(
                        "Marked against the answer key: "
                        + ("correct." if correct else "not correct.")
                    )
                else:
                    st.caption(
                        "You marked this as "
                        + ("correct." if correct else "not correct.")
                    )
            else:
                with st.form(f"check_{turn['id']}"):
                    check_response = st.text_area(
                        "Your answer",
                        key=f"check_answer_{turn['id']}",
                        height=80,
                        placeholder="Just the answer is fine - x=4, x = 4 and 4 all count.",
                    )
                    self_marked = None
                    if not turn.get("check_answer"):
                        # No answer key for a free-text question, so the student
                        # marks their own. It is required: without an outcome
                        # that half of the sealed prediction cannot be scored.
                        self_marked = st.radio(
                            "Did you get it right?",
                            (True, False),
                            format_func=lambda value: "Yes" if value else "No",
                            index=None,
                            horizontal=True,
                            key=f"self_mark_{turn['id']}",
                        )
                    submitted = st.form_submit_button("Save my response")
                if submitted:
                    if not check_response.strip():
                        st.warning("Write a response before saving it.")
                    elif not turn.get("check_answer") and self_marked is None:
                        st.warning(
                            "Mark whether your answer was right, so the round "
                            "can be scored."
                        )
                    else:
                        save_check_response(
                            turn, check_response, student_id, self_marked
                        )
        render_prediction(turn)


def render_chat(student_id: str, provider: TutorProvider) -> None:
    initialize_chat_state(student_id)
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("is_generating", False)
    student = st.session_state.student_data
    logs = round_logs(student)
    if student.get("notes"):
        st.info(f"**What the agent has written about you:** {profile_notes(student)}")
    else:
        st.info(
            "Ask a question to get started. You'll get three explanations - and "
            "before you choose, the agent writes down which one it thinks you'll "
            "pick and whether it will actually help you."
        )
    style = preferred_style(student["preferences"])
    if style:
        st.caption(
            f"Most-picked explanation so far: {STYLE_LABELS[style].lower()}. What "
            "you prefer and what helps you are scored separately."
        )
    if logs:
        with st.expander(f"Know-you score · {len(logs)} scored round(s)"):
            show_score_chart(logs)

    for turn in student["conversations"]:
        render_turn(turn, student_id)

    failed_submission = st.session_state.get("failed_submission")
    if failed_submission:
        with st.chat_message("user"):
            st.markdown(failed_submission["prompt"])
        with st.chat_message("assistant"):
            st.error(failed_submission["error"])
        if st.button("Dismiss error", key="dismiss_chat_error"):
            st.session_state.failed_submission = None
            st.rerun()

    pending_prompt = st.session_state.pending_prompt
    if pending_prompt:
        with st.chat_message("user"):
            st.markdown(pending_prompt)

    st.chat_input(
        "Ask a maths question or request a definition…",
        key="chat_prompt",
        on_submit=queue_chat_prompt,
        disabled=bool(pending_prompt),
        submit_mode="disable" if pending_prompt else "submit",
    )

    if pending_prompt:
        succeeded = False
        with st.status("The tutor is thinking…", expanded=False) as thinking:
            if isinstance(provider, GeminiTutor):
                provider.on_retry = lambda attempt, total, delay: thinking.update(
                    label=(
                        f"Gemini is busy. Retrying ({attempt + 1}/{total}) "
                        f"in {delay:g}s…"
                    ),
                    state="running",
                )
            try:
                succeeded = append_prompt(pending_prompt, student_id, provider)
                if not succeeded:
                    error = st.session_state.pop(
                        "last_provider_error",
                        "Please check the tutor configuration and try again.",
                    )
                    st.session_state.failed_submission = {
                        "prompt": pending_prompt,
                        "error": f"The tutor could not answer this question: {error}",
                    }
            finally:
                if isinstance(provider, GeminiTutor):
                    provider.on_retry = None
                st.session_state.pending_prompt = None
                st.session_state.is_generating = False
                thinking.update(
                    label=(
                        "Tutor response ready."
                        if succeeded
                        else "Tutor request failed."
                    ),
                    state="complete" if succeeded else "error",
                )
        st.rerun()


def render_replay(all_results: list[dict[str, Any]]) -> None:
    if not all_results:
        st.warning("results.json has no rounds to replay yet.")
        return
    students = sorted({record["student"] for record in all_results})
    selected_student = st.sidebar.selectbox(
        "Student",
        students,
        format_func=lambda value: value.title(),
    )
    records = sorted(
        (record for record in all_results if record["student"] == selected_student),
        key=lambda record: record["round"],
    )
    round_index = st.sidebar.selectbox(
        "Replay round",
        range(len(records)),
        format_func=lambda index: f"Round {records[index]['round']}",
    )
    record = records[round_index]
    st.subheader(f"{selected_student.title()} · Round {record['round']}")
    st.markdown(f"**Selected style:** {STYLE_LABELS[record['actual_pick']]}")
    st.markdown(
        f"**Agent prediction:** {STYLE_LABELS[record['pred_pick']]} · "
        f"{'correct' if record['pred_correct'] else 'incorrect'}"
    )
    st.markdown(
        f"**Student outcome:** {'correct' if record['actual_correct'] else 'incorrect'}"
    )
    st.caption(record["profile_after"])
    st.subheader("Know-you score")
    show_score_chart(records)


def render_live_sim() -> None:
    """Watch the agent learn a cohort of simulated students, round by round.

    The replay view shows a finished run; this one shows the learning happening.
    sim.stream() is a generator, so every round is drawn as it is scored rather
    than after the whole run completes.

    Speed is the honest constraint here. The mock agent is pure Python: a
    couple of hundred rounds land in seconds and cost nothing. The real agent
    is one to three Gemini calls per round at two to four seconds each, so the
    same run takes roughly twenty minutes and several hundred calls. Both are
    offered; the sidebar says which one is running, because a chart drawn by
    the mock agent proves the mechanism works, not that Gemini learns.
    """
    import sim

    st.sidebar.markdown("---")
    use_mock = st.sidebar.radio(
        "Agent",
        ("Mock (instant, free)", "Gemini (real, slow)"),
        help="The mock agent is a plain Python tally. It proves the loop and "
             "the chart work. Only the Gemini run is evidence the agent learns.",
    ).startswith("Mock")
    names = st.sidebar.multiselect(
        "Students", sorted(sim.make_students(0)), default=sim.HEADLINE_STUDENTS
    )
    rounds = st.sidebar.slider("Rounds each", 5, 60, 30 if use_mock else 10)
    delay = st.sidebar.slider("Seconds per round", 0.0, 2.0, 0.15 if use_mock else 0.0, 0.05,
                              help="Slow it down so an audience can follow.")
    every = st.sidebar.slider("Force a style every N rounds", 0, 6, 3,
                              help="0 turns interventions off. Watch Maya's "
                                   "line with it on and off.")

    if not names:
        st.info("Pick at least one student in the sidebar.")
        return
    if not use_mock:
        st.warning(
            f"A real run is about {len(names) * rounds} rounds and "
            f"{len(names) * rounds * 2} Gemini calls. The free tier allows 20 "
            "per model per day."
        )
    if not st.button("Run the simulation", type="primary"):
        st.caption("The agent starts knowing nothing. Watch the line climb.")
        return

    # Order matters: the counters and the chart are what an audience watches,
    # so they go above the feed. Creating the columns after feed_box would
    # push them below it and off the bottom of the screen.
    boxes = [c.empty() for c in st.columns(4)]
    chart_box = st.empty()
    st.caption("Each line is one student. Flat line is random guessing.")
    feed_box = st.empty()
    feed: list[str] = []

    try:
        for row in sim.stream(names, rounds, 42, use_mock, delay, every,
                              results_file=ROOT / "results_live.json"):
            log = row["_log"]
            frame = {}
            for n in names:
                rows = [x for x in log if x["student"] == n]
                if rows:
                    frame[n.capitalize()] = sim.know_you_scores(rows)
            width = max((len(v) for v in frame.values()), default=0)
            frame = {k: v + [None] * (width - len(v)) for k, v in frame.items()}
            frame["Random baseline"] = [sim.RANDOM_BASELINE] * width
            chart_box.line_chart(frame, height=320)

            # scored_halves yields (hit, luck_rate) pairs, not bare booleans:
            # the baseline is per-half, because an intervention round scores
            # only the 1-in-2 outcome half and not the 1-in-3 pick half.
            halves = [h for x in log for h in scored_halves(x)]
            boxes[0].metric("Rounds", len(log))
            boxes[1].metric(
                "Know-you score",
                f"{sum(hit for hit, _ in halves) / len(halves):.0%}" if halves else "-",
            )
            boxes[2].metric(
                "Random baseline",
                f"{sum(luck for _, luck in halves) / len(halves):.0%}" if halves else "-",
            )
            boxes[3].metric("Interventions",
                            sum(1 for x in log if x["forced_style"]))

            mark = "--" if row["pick_hit"] is None else ("HIT" if row["pick_hit"] else "miss")
            feed.insert(0, (
                f"`r{row['round']:>2}` **{row['student']}** "
                f"{'· forced ' + row['forced_style'] + ' ' if row['forced_style'] else ''}"
                f"predicted *{row['pred_pick']}* → read *{row['actual_pick']}*, "
                f"{'got it right' if row['actual_correct'] else 'got it wrong'} "
                f"· pick {mark} · outcome {'HIT' if row['correct_hit'] else 'miss'}"
            ))
            feed_box.markdown("\n\n".join(feed[:12]))
    except Exception as error:  # noqa: BLE001 - a dead quota must not crash the demo
        st.error(f"The run stopped: {error}")
        st.caption("Switch the sidebar Agent to Mock to keep going offline.")
        return

    st.success(f"Done. {len(names) * rounds} rounds. Saved to results_live.json")
    st.caption(
        "Mock agent: this shows the loop and the chart work, not that Gemini learns."
        if use_mock else
        "Real Gemini agent: this is the result that counts."
    )


def main() -> None:
    st.set_page_config(
        page_title="The tutor that learns the student",
        page_icon="📚",
        layout="wide",
    )
    st.title("The tutor that learns the student")
    st.caption("Ask a question. Get an explanation that fits how you like to learn.")
    mode = st.sidebar.radio(
        "App view", ("Tutor chat", "Live simulation", "Replay simulation")
    )
    if mode == "Replay simulation":
        render_replay(load_results())
        return
    if mode == "Live simulation":
        render_live_sim()
        return

    student_id = st.sidebar.text_input("Student name", value="judge").strip()
    if not student_id:
        st.error("Enter a student name so your preferences can be saved.")
        st.stop()

    provider_name = st.sidebar.selectbox(
        "Tutor provider",
        ("Gemini (Google)", "Local demo (offline)"),
    )
    if provider_name == "Gemini (Google)":
        try:
            provider: TutorProvider = GeminiTutor(env_file=ROOT / ".env")
        except TutorProviderError as error:
            st.error(str(error))
            st.stop()
        st.sidebar.warning("Your questions and saved style preferences will be sent to Google Gemini.")
    else:
        provider = DemoTutor()
        st.sidebar.caption("Demo mode uses template explanations and makes no network calls.")
    render_chat(student_id, provider)


if __name__ == "__main__":
    main()
