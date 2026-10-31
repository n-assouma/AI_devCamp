import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from preference_store import load_preference_store, save_preference_store
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
PREFERENCE_FILE = ROOT / "student_preferences.json"
STYLE_LABELS = {
    "steps": "Step by step",
    "analogy": "Analogy",
    "example": "Worked example",
}
RANDOM_BASELINE = 5 / 12


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read {path.name}: {error}")
        st.stop()


def load_results() -> list[dict[str, Any]]:
    data = load_json(RESULTS_FILE)
    if not isinstance(data, list):
        st.error("results.json must contain a list of round records.")
        st.stop()
    required_booleans = ("pred_correct", "actual_correct", "pick_hit", "correct_hit")
    for index, record in enumerate(data):
        valid = (
            isinstance(record, dict)
            and isinstance(record.get("round"), int)
            and isinstance(record.get("student"), str)
            and record.get("pred_pick") in STYLES
            and record.get("actual_pick") in STYLES
            and all(isinstance(record.get(field), bool) for field in required_booleans)
            and isinstance(record.get("profile_after"), str)
        )
        if not valid:
            st.error(f"Round record {index + 1} does not match the results.json format.")
            st.stop()
    return data


def rolling_scores(records: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    scores = []
    for index, record in enumerate(records):
        window = records[max(0, index - 4) : index + 1]
        hits = sum(
            int(item["pick_hit"]) + int(item["correct_hit"]) for item in window
        )
        scores.append({"Round": record["round"], "Know-you score": hits / (2 * len(window))})
    return scores


def show_score_chart(records: list[dict[str, Any]]) -> None:
    if not records:
        st.info("No simulation rounds are available.")
        return
    score_data = pd.DataFrame(rolling_scores(records)).set_index("Round")
    score_data["Random baseline"] = RANDOM_BASELINE
    st.line_chart(score_data, y=["Know-you score", "Random baseline"])
    st.caption("Rolling five-round accuracy across both predictions. Random baseline: 41.7%.")


def load_student(student_id: str) -> dict[str, Any]:
    store = load_preference_store(PREFERENCE_FILE)
    student = store["students"].get(
        student_id,
        {"preferences": [], "rounds": [], "conversations": []},
    )
    student.setdefault("conversations", [])
    return student


def persist_student(student_id: str, student: dict[str, Any]) -> None:
    store = load_preference_store(PREFERENCE_FILE)
    store["students"][student_id] = student
    save_preference_store(store, PREFERENCE_FILE)


def profile_notes(preferences: list[dict[str, Any]]) -> str:
    counts = {style: 0 for style in STYLES}
    for preference in preferences:
        style = preference.get("style")
        if style in counts:
            counts[style] += 1
    total = sum(counts.values())
    if not total:
        return "No saved explanation preference yet."
    details = ", ".join(
        f"{STYLE_LABELS[style]}: {counts[style]}"
        for style in STYLES
        if counts[style]
    )
    return f"Explicitly selected explanation styles ({total} choices): {details}."


def preferred_style(preferences: list[dict[str, Any]]) -> ExplanationStyle | None:
    counts = {style: 0 for style in STYLES}
    for preference in preferences:
        style = preference.get("style")
        if style in counts:
            counts[style] += 1
    if not any(counts.values()):
        return None
    return max(STYLES, key=lambda style: (counts[style], -STYLES.index(style)))


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
    style = preferred_style(student["preferences"])
    try:
        response = validate_tutor_response(
            provider(prompt, profile_notes(student["preferences"]), style)
        )
        required_styles = (style,) if style else STYLES
        missing_styles = [item for item in required_styles if item not in response["answers"]]
        if missing_styles:
            raise ValueError(
                f"Tutor response did not include required styles: {', '.join(missing_styles)}."
            )
    except (TutorProviderError, TypeError, ValueError) as error:
        st.session_state.last_provider_error = str(error)
        return False

    turn = {
        "id": str(uuid.uuid4()),
        "prompt": prompt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "response": response,
        "preferred_style": style,
        "selected_style": style,
        "check_response": None,
    }
    student["conversations"].append(turn)
    st.session_state.student_data = student
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
    persist_student(student_id, student)
    st.rerun()


def save_check_response(
    turn: dict[str, Any],
    response_text: str,
    student_id: str,
) -> None:
    student = st.session_state.student_data
    turn["check_response"] = {
        "text": response_text.strip(),
        "answered_at": datetime.now(timezone.utc).isoformat(),
    }
    student["rounds"].append(
        {
            "conversation_id": turn["id"],
            "prompt": turn["prompt"],
            "check_question": turn["response"]["check_question"],
            "student_response": response_text.strip(),
            "answered_at": turn["check_response"]["answered_at"],
        }
    )
    st.session_state.student_data = student
    persist_student(student_id, student)
    st.rerun()


def render_turn(turn: dict[str, Any], student_id: str) -> None:
    response = turn["response"]
    with st.chat_message("user"):
        st.markdown(turn["prompt"])

    with st.chat_message("assistant"):
        if turn["preferred_style"]:
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
            st.markdown(
                "I don't have a saved explanation preference yet. Which version "
                "works best for you?"
            )
            cols = st.columns(3)
            for col, option_style in zip(cols, STYLES):
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
            return

        if response["is_definition"]:
            st.caption("This is a definition, so there is no knowledge-check question.")
        else:
            st.markdown(f"**Try this:** {response['check_question']}")
            if turn["check_response"]:
                st.markdown("**Your response**")
                st.write(turn["check_response"]["text"])
                st.caption("Saved for your learning history; this response is not graded.")
            else:
                with st.form(f"check_{turn['id']}"):
                    check_response = st.text_area(
                        "Your answer",
                        key=f"check_answer_{turn['id']}",
                        height=80,
                        placeholder="Write what you would try or explain your thinking.",
                    )
                    submitted = st.form_submit_button("Save my response")
                if submitted:
                    if not check_response.strip():
                        st.warning("Write a response before saving it.")
                    else:
                        save_check_response(turn, check_response, student_id)


def render_chat(student_id: str, provider: TutorProvider) -> None:
    initialize_chat_state(student_id)
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("is_generating", False)
    student = st.session_state.student_data
    style = preferred_style(student["preferences"])
    if style:
        st.info(f"Your saved preference is **{STYLE_LABELS[style].lower()}**.")
    else:
        st.info(
            "Ask a question to get started. I'll offer three explanation styles "
            "and remember the one you choose."
        )

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


def main() -> None:
    st.set_page_config(
        page_title="The tutor that learns the student",
        page_icon="📚",
        layout="wide",
    )
    st.title("The tutor that learns the student")
    st.caption("Ask a question. Get an explanation that fits how you like to learn.")
    mode = st.sidebar.radio("App view", ("Tutor chat", "Replay simulation"))
    if mode == "Replay simulation":
        render_replay(load_results())
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
            provider: TutorProvider = GeminiTutor(
                env_file=ROOT / "my_agent" / ".env"
            )
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
