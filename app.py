from __future__ import annotations

import io
import json
import uuid
from typing import Dict, List

import streamlit as st

from src.config import (
    INITIAL_GENERATION_PROMPT_PATH,
    QUESTION_GENERATION_PROMPT_PATH,
    REFINEMENT_PROMPT_PATH,
)
from src.models.schemas import (
    ClarificationQuestionSet,
    DecisionInput,
    RoundEvaluation,
    StructuredDecisionOutput,
    UserAnswer,
)
from src.models.state import StudyStateManager
from src.pipeline.initial_generator import InitialGenerator
from src.pipeline.question_generator import QuestionGenerator
from src.pipeline.refiner import Refiner
from src.utils.llm_client import OpenAILLMClient


st.set_page_config(
    page_title="Decision Articulation Study",
    page_icon="🧭",
    layout="wide",
)


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .app-subtitle {
            font-size: 1.02rem;
            color: #6b7280;
            margin-top: -0.45rem;
            margin-bottom: 1.4rem;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
        }

        .metric-chip {
            display: inline-block;
            padding: 0.38rem 0.78rem;
            border-radius: 999px;
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            color: #312e81;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            font-size: 0.9rem;
            font-weight: 600;
        }

        .small-muted {
            color: #6b7280;
            font-size: 0.92rem;
        }

        .round-badge {
            display: inline-block;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            background: #eef2ff;
            color: #3730a3;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .status-badge {
            display: inline-block;
            padding: 0.22rem 0.58rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-left: 0.45rem;
            vertical-align: middle;
        }

        .status-added {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }

        .status-revised {
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
        }

        .status-unchanged {
            background: #e0e7ff;
            color: #3730a3;
            border: 1px solid #a5b4fc;
        }

        .star-label {
            font-weight: 600;
            margin-bottom: 0.4rem;
            margin-top: 0.8rem;
        }

        .stButton > button {
            border-radius: 12px;
            padding: 0.58rem 1rem;
            font-weight: 700;
            border: none;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #4338ca, #6d28d9);
            color: white;
        }

        .stDownloadButton > button {
            border-radius: 12px;
            padding: 0.58rem 1rem;
            font-weight: 700;
            border: none;
            background: linear-gradient(135deg, #0f766e, #0891b2);
            color: white;
        }

        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #115e59, #0e7490);
            color: white;
        }

        .stTextInput > div > div > input,
        .stTextArea textarea {
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
        }

        .stSelectbox > div > div,
        .stSlider {
            border-radius: 12px !important;
        }

        div[data-testid="stExpander"] {
            border-radius: 14px !important;
            border: 1px solid #dbe2ea !important;
            background: #fafafa !important;
        }

        button[kind="secondary"] {
            border-radius: 12px !important;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        button[data-baseweb="tab"] {
            border-radius: 12px !important;
            background: #f3f4f6 !important;
            padding: 0.45rem 0.9rem !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: #e0e7ff !important;
            color: #312e81 !important;
            font-weight: 700 !important;
        }

        .star-note {
            color: #6b7280;
            font-size: 0.88rem;
            margin-top: -0.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    defaults = {
        "study_started": False,
        "study_completed": False,
        "decision_input": None,
        "manager": None,
        "current_question_set": None,
        "current_round_index": 0,
        "rating_submitted_rounds": set(),
        "answers_submitted_rounds": set(),
        "selected_model": "gpt-5.4-mini",
        "custom_model_name": "",
        "max_rounds": 2,
        "download_json_bytes": None,
        "download_csv_bytes": None,
        "download_json_filename": None,
        "download_csv_filename": None,
        "initial_generator": None,
        "question_generator": None,
        "refiner": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def get_effective_model_name() -> str:
    if st.session_state.selected_model == "Custom":
        custom_name = st.session_state.custom_model_name.strip()
        return custom_name if custom_name else "gpt-5.4-mini"
    return st.session_state.selected_model


def initialize_services(model_name: str):
    llm_client = OpenAILLMClient(model_name=model_name)
    initial_generator = InitialGenerator(llm_client, INITIAL_GENERATION_PROMPT_PATH)
    question_generator = QuestionGenerator(llm_client, QUESTION_GENERATION_PROMPT_PATH)
    refiner = Refiner(llm_client, REFINEMENT_PROMPT_PATH)
    return initial_generator, question_generator, refiner


def get_change_badge(change_type: str | None) -> str:
    if not change_type:
        return ""
    mapping = {
        "added": ('<span class="status-badge status-added">● Added</span>'),
        "revised": ('<span class="status-badge status-revised">● Revised</span>'),
        "unchanged": ('<span class="status-badge status-unchanged">● Unchanged</span>'),
    }
    return mapping.get(change_type, "")


def render_structured_output(round_index: int, structured_output: StructuredDecisionOutput) -> None:
    st.markdown(
        f'<div class="round-badge">Round {round_index}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Decision Summary")
    st.write(structured_output.decision_summary)

    alt_tab, pref_tab, unc_tab = st.tabs(["Alternatives", "Preferences", "Uncertainties"])

    with alt_tab:
        if structured_output.alternatives:
            for alt in structured_output.alternatives:
                badge = get_change_badge(alt.change_type)
                title_html = f"{alt.id}: {alt.label} {badge}"
                st.markdown(title_html, unsafe_allow_html=True)
                with st.expander("View details", expanded=True):
                    st.write(alt.description)
        else:
            st.info("No alternatives available.")

    with pref_tab:
        if structured_output.preferences:
            for pref in structured_output.preferences:
                badge = get_change_badge(pref.change_type)
                source_text = f'<span class="metric-chip">{pref.source}</span>' if pref.source else ""
                st.markdown(f"{pref.id}: {pref.label} {badge}", unsafe_allow_html=True)
                st.markdown(source_text, unsafe_allow_html=True)
                with st.expander("View details", expanded=True):
                    st.write(pref.description)
        else:
            st.info("No preferences available.")

    with unc_tab:
        if structured_output.uncertainties:
            for unc in structured_output.uncertainties:
                badge = get_change_badge(unc.change_type)
                type_text = f'<span class="metric-chip">{unc.type}</span>' if unc.type else ""
                st.markdown(f"{unc.id}: {unc.label} {badge}", unsafe_allow_html=True)
                st.markdown(type_text, unsafe_allow_html=True)
                with st.expander("View details", expanded=True):
                    st.write(unc.description)
        else:
            st.info("No uncertainties available.")

    if structured_output.missing_but_relevant_information:
        st.markdown("### Missing but Relevant Information")
        for item in structured_output.missing_but_relevant_information:
            st.write(f"- {item}")

    if structured_output.refinement_notes:
        st.markdown("### Refinement Notes")
        for note in structured_output.refinement_notes:
            st.write(f"- {note}")


def render_submitted_evaluation(round_index: int, evaluation: RoundEvaluation) -> None:
    st.markdown("### Submitted Evaluation")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Faithfulness", "★" * evaluation.faithfulness)
    col2.metric("Completeness", "★" * evaluation.completeness)
    col3.metric("Clarity", "★" * evaluation.clarity)
    col4.metric("Usefulness", "★" * evaluation.usefulness)
    col5.metric("Self-expression", "★" * evaluation.self_expression_support)

    if evaluation.notes:
        st.markdown("**Notes**")
        st.write(evaluation.notes)


def star_rating_input(label: str, key_prefix: str) -> int:
    st.markdown(f'<div class="star-label">{label}</div>', unsafe_allow_html=True)

    options = {
        "⭐": 1,
        "⭐⭐": 2,
        "⭐⭐⭐": 3,
        "⭐⭐⭐⭐": 4,
        "⭐⭐⭐⭐⭐": 5,
    }

    selected = st.radio(
        label,
        options=list(options.keys()),
        index=2,
        key=key_prefix,
        horizontal=True,
        label_visibility="collapsed",
    )
    return options[selected]


def render_rating_form(round_index: int) -> None:
    st.markdown("### Evaluation")
    st.markdown('<div class="star-note">Click the stars to rate each item from 1 to 5.</div>', unsafe_allow_html=True)

    faithfulness = star_rating_input("Faithfulness to your real situation", f"faithfulness_{round_index}")
    completeness = star_rating_input("Completeness", f"completeness_{round_index}")
    clarity = star_rating_input("Clarity", f"clarity_{round_index}")
    usefulness = star_rating_input("Usefulness for decision-making", f"usefulness_{round_index}")
    self_expression_support = star_rating_input("Helped you express what you meant", f"self_expression_support_{round_index}")

    notes = st.text_area(
        "Optional notes",
        key=f"notes_{round_index}",
        height=100,
    )

    if st.button(f"Submit Evaluation for Round {round_index}", key=f"submit_eval_{round_index}", use_container_width=True):
        evaluation = RoundEvaluation(
            round_index=round_index,
            faithfulness=faithfulness,
            completeness=completeness,
            clarity=clarity,
            usefulness=usefulness,
            self_expression_support=self_expression_support,
            notes=notes or None,
        )
        st.session_state.manager.attach_evaluation(round_index, evaluation)
        st.session_state.rating_submitted_rounds.add(round_index)
        st.success(f"Evaluation for Round {round_index} submitted.")
        st.rerun()


def render_questions(question_set: ClarificationQuestionSet, round_index: int) -> Dict[str, str]:
    st.markdown(f"### Clarification Questions for Round {round_index}")
    st.caption(question_set.round_goal)

    answers: Dict[str, str] = {}
    for q in question_set.questions:
        with st.expander(f"{q.id}: {q.question}", expanded=True):
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.caption(f"Target: {q.target}")
            with meta_col2:
                st.caption(f"Type: {q.question_type}")
            st.write(q.rationale)

            answers[q.id] = st.text_area(
                label=f"Answer for {q.id}",
                key=f"answer_round_{round_index}_{q.id}",
                height=110,
                placeholder="Type your answer here...",
            )
    return answers


def start_study(title: str, narrative: str, model_name: str) -> None:
    decision_input = DecisionInput(
        decision_id=str(uuid.uuid4()),
        title=title,
        narrative=narrative,
    )

    initial_generator, question_generator, refiner = initialize_services(model_name)

    manager = StudyStateManager(
        decision_input=decision_input,
        model_name=model_name,
    )

    initial_output = initial_generator.run(decision_input)
    manager.add_initial_round(initial_output)

    st.session_state.study_started = True
    st.session_state.study_completed = False
    st.session_state.decision_input = decision_input
    st.session_state.manager = manager
    st.session_state.current_question_set = None
    st.session_state.current_round_index = 0
    st.session_state.rating_submitted_rounds = set()
    st.session_state.answers_submitted_rounds = set()
    st.session_state.download_json_bytes = None
    st.session_state.download_csv_bytes = None
    st.session_state.download_json_filename = None
    st.session_state.download_csv_filename = None

    st.session_state.initial_generator = initial_generator
    st.session_state.question_generator = question_generator
    st.session_state.refiner = refiner


def generate_questions_for_next_round() -> None:
    manager: StudyStateManager = st.session_state.manager
    current_round = manager.get_current_round()

    if current_round is None:
        st.error("No current round found.")
        return

    if current_round.round_index not in st.session_state.rating_submitted_rounds:
        st.warning("Please submit the current round evaluation before generating the next question set.")
        return

    if current_round.round_index >= st.session_state.max_rounds:
        st.warning("Maximum number of rounds reached.")
        return

    prior_history = manager.get_prior_qa_history_as_text()

    question_set = st.session_state.question_generator.run(
        decision_title=manager.state.title,
        decision_narrative=manager.state.narrative,
        current_structured_output=current_round.structured_output,
        prior_qa_history=prior_history,
    )

    st.session_state.current_question_set = question_set


def submit_answers_and_refine(round_index: int, raw_answers: Dict[str, str]) -> None:
    manager: StudyStateManager = st.session_state.manager
    current_round = manager.get_current_round()
    question_set = st.session_state.current_question_set

    if current_round is None or question_set is None:
        st.error("Missing current round or question set.")
        return

    user_answers: List[UserAnswer] = []
    for q in question_set.questions:
        answer_text = raw_answers.get(q.id, "").strip()
        user_answers.append(
            UserAnswer(
                question_id=q.id,
                question_text=q.question,
                answer=answer_text,
            )
        )

    prior_history = manager.get_prior_qa_history_as_text()

    refined_output = st.session_state.refiner.run(
        decision_title=manager.state.title,
        decision_narrative=manager.state.narrative,
        previous_structured_output=current_round.structured_output,
        current_questions=question_set,
        user_answers=user_answers,
        prior_qa_history=prior_history,
    )

    manager.add_followup_round(
        round_index=round_index,
        structured_output=refined_output,
        question_set=question_set,
        user_answers=user_answers,
    )

    st.session_state.current_round_index = round_index
    st.session_state.current_question_set = None
    st.session_state.answers_submitted_rounds.add(round_index)


def build_json_download(manager: StudyStateManager) -> tuple[bytes, str]:
    payload = json.dumps(manager.as_dict(), ensure_ascii=False, indent=2).encode("utf-8")
    filename = f"{manager.state.decision_id}_study_results.json"
    return payload, filename


def build_csv_download(manager: StudyStateManager) -> tuple[bytes, str]:
    output = io.StringIO()
    output.write(
        "decision_id,round_index,n_alternatives,n_preferences,n_uncertainties,"
        "faithfulness,completeness,clarity,usefulness,self_expression_support,notes\n"
    )

    for round_record in manager.state.rounds:
        ev = round_record.evaluation
        row = [
            manager.state.decision_id,
            str(round_record.round_index),
            str(len(round_record.structured_output.alternatives)),
            str(len(round_record.structured_output.preferences)),
            str(len(round_record.structured_output.uncertainties)),
            str(ev.faithfulness) if ev else "",
            str(ev.completeness) if ev else "",
            str(ev.clarity) if ev else "",
            str(ev.usefulness) if ev else "",
            str(ev.self_expression_support) if ev else "",
            f"\"{(ev.notes or '').replace('\"', '\"\"')}\"" if ev else "\"\"",
        ]
        output.write(",".join(row) + "\n")

    filename = f"{manager.state.decision_id}_round_summary.csv"
    return output.getvalue().encode("utf-8"), filename


def complete_study() -> None:
    manager: StudyStateManager = st.session_state.manager
    latest_round = manager.get_current_round()

    if latest_round is None:
        st.error("No round data found.")
        return

    if latest_round.round_index not in st.session_state.rating_submitted_rounds:
        st.warning("Please submit the latest round evaluation before finishing the study.")
        return

    json_bytes, json_filename = build_json_download(manager)
    csv_bytes, csv_filename = build_csv_download(manager)

    st.session_state.download_json_bytes = json_bytes
    st.session_state.download_csv_bytes = csv_bytes
    st.session_state.download_json_filename = json_filename
    st.session_state.download_csv_filename = csv_filename
    st.session_state.study_completed = True


def main() -> None:
    initialize_session_state()
    apply_custom_styles()

    with st.sidebar:
        st.markdown("## Study Settings")

        current_model = st.session_state.selected_model
        model_options = ["gpt-5.4", "gpt-5.4-mini", "Custom"]
        selected_index = model_options.index(current_model) if current_model in model_options else 1

        model_option = st.selectbox(
            "Model",
            options=model_options,
            index=selected_index,
        )

        custom_model_name = ""
        if model_option == "Custom":
            custom_model_name = st.text_input(
                "Custom model name",
                value=st.session_state.custom_model_name,
                placeholder="Enter a model name",
            )

        max_rounds = st.slider(
            "Max Rounds",
            min_value=0,
            max_value=5,
            value=st.session_state.max_rounds,
            step=1,
        )

        st.session_state.selected_model = model_option
        st.session_state.custom_model_name = custom_model_name
        st.session_state.max_rounds = max_rounds

        if st.button("Reset Session", use_container_width=True):
            reset_session()

    st.title("Decision Articulation Study")
    st.markdown(
        '<div class="app-subtitle">Explore whether iterative LLM questioning helps users better express their decision context, preferences, and uncertainties.</div>',
        unsafe_allow_html=True,
    )

    effective_model_name = get_effective_model_name()

    if not st.session_state.study_started:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Start a New Decision Study")

        title = st.text_input(
            "Decision Title",
            placeholder="e.g. Should I invest in stocks or crypto?",
        )

        narrative = st.text_area(
            "Decision Narrative",
            height=220,
            placeholder="Describe your situation, goals, constraints, concerns, and anything else that matters.",
        )

        st.markdown(
            f"""
            <span class="metric-chip">Model: {effective_model_name}</span>
            <span class="metric-chip">Max Rounds: {st.session_state.max_rounds}</span>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Start Study", use_container_width=True):
            if not title.strip() or not narrative.strip():
                st.warning("Please provide both a decision title and a decision narrative.")
            else:
                with st.spinner("Generating initial structured decision output..."):
                    start_study(title.strip(), narrative.strip(), effective_model_name)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        return

    manager: StudyStateManager = st.session_state.manager

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## Decision Input")
    st.markdown(f"**Title:** {manager.state.title}")
    st.write(manager.state.narrative)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## Round History")

    for round_record in manager.state.rounds:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_structured_output(round_record.round_index, round_record.structured_output)

        if round_record.evaluation:
            render_submitted_evaluation(round_record.round_index, round_record.evaluation)
        elif round_record.round_index not in st.session_state.rating_submitted_rounds:
            render_rating_form(round_record.round_index)

        st.markdown("</div>", unsafe_allow_html=True)

    current_round = manager.get_current_round()
    current_round_index = current_round.round_index if current_round else 0

    if current_round_index < st.session_state.max_rounds:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("## Next Clarification Step")

        if st.session_state.current_question_set is None:
            if current_round_index in st.session_state.rating_submitted_rounds:
                if st.button(f"Generate Questions for Round {current_round_index + 1}", use_container_width=True):
                    with st.spinner("Generating clarification questions..."):
                        generate_questions_for_next_round()
                    st.rerun()
            else:
                st.info("Submit the current round evaluation to continue.")
        else:
            raw_answers = render_questions(
                st.session_state.current_question_set,
                current_round_index + 1,
            )

            if st.button(f"Submit Answers and Generate Round {current_round_index + 1} Output", use_container_width=True):
                with st.spinner("Refining structured decision output..."):
                    submit_answers_and_refine(current_round_index + 1, raw_answers)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("## Finish Study")
        st.caption("You have reached the selected maximum number of rounds.")

        if st.button("Complete Study", use_container_width=True):
            complete_study()
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.study_completed:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("## Download Results")
        st.success("Study completed. Download your results below.")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="Download JSON Results",
                data=st.session_state.download_json_bytes,
                file_name=st.session_state.download_json_filename,
                mime="application/json",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                label="Download CSV Summary",
                data=st.session_state.download_csv_bytes,
                file_name=st.session_state.download_csv_filename,
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()