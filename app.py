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
    ROUND_IMPROVEMENT_EVALUATION_PROMPT_PATH,
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
from src.pipeline.improvement_evaluator import ImprovementEvaluator
from src.pipeline.refiner import Refiner
from src.utils.llm_client import OpenAILLMClient
from src.evaluation.rubric import USER_RATING_RUBRIC


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
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .app-subtitle {
            font-size: 1.02rem;
            color: #6b7280;
            margin-top: -0.45rem;
            margin-bottom: 1.35rem;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
        }
        
        .metric-chip {
            display: inline-block;
            padding: 0.36rem 0.74rem;
            border-radius: 999px;
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            color: #312e81;
            margin-right: 0.4rem;
            margin-bottom: 0.35rem;
            font-size: 0.88rem;
            font-weight: 600;
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

        .star-label {
            font-weight: 700;
            margin-bottom: 0.35rem;
            margin-top: 0.85rem;
            font-size: 0.98rem;
        }

        .star-note {
            color: #6b7280;
            font-size: 0.88rem;
            margin-top: -0.1rem;
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
        
        .tooltip-question {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            margin-left: 6px;
            border-radius: 999px;
            background: #e0e7ff;
            color: #3730a3;
            font-size: 0.78rem;
            font-weight: 800;
            border: 1px solid #c7d2fe;
            cursor: help;
            vertical-align: middle;
        }
                
        div[data-testid="stExpander"] {
            border-radius: 14px !important;
            border: 1px solid #dbe2ea !important;
            background: #fafafa !important;
            margin-bottom: 0.7rem !important;
        }

        div[data-testid="stExpander"] summary {
            font-weight: 700 !important;
            font-size: 1rem !important;
        }
        
        div[data-testid="stExpander"] summary p {
            font-weight: 700 !important;
            font-size: 1rem !important;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.5rem;
            flex-wrap: wrap;
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

        div[data-testid="stRadio"] [role="radiogroup"] {
            gap: 0.55rem !important;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"] input {
            position: absolute !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"] {
            background: #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
            padding: 0.42rem 0.95rem !important;
            min-width: 62px;
            justify-content: center !important;
            transition: all 0.15s ease !important;
            cursor: pointer !important;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
            border-color: #818cf8 !important;
            background: #eef2ff !important;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
            border-color: #4f46e5 !important;
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.12) !important;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"] p {
            font-size: 1rem !important;
            line-height: 1.1 !important;
            color: #d97706 !important;
            font-weight: 700 !important;
            margin: 0 !important;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
            color: white !important;
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
        "download_ai_json_bytes": None,
        "download_ai_csv_bytes": None,
        "download_ai_json_filename": None,
        "download_ai_csv_filename": None,
        "initial_generator": None,
        "question_generator": None,
        "refiner": None,
        "improvement_evaluator": None,
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
    improvement_evaluator = ImprovementEvaluator(
        llm_client,
        ROUND_IMPROVEMENT_EVALUATION_PROMPT_PATH,
    )
    return initial_generator, question_generator, refiner, improvement_evaluator


def render_structured_output(round_index: int, structured_output: StructuredDecisionOutput) -> None:
    st.markdown(
        f'<div class="round-badge">Round {round_index}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Decision Summary")
    st.write(structured_output.decision_summary)

    alt_tab, pref_tab, unc_tab, ethics_tab, stakeholders_tab = st.tabs(
        ["Alternatives", "Preferences", "Uncertainties", "Ethics", "Stakeholders"]
    )

    with alt_tab:
        if structured_output.alternatives:
            for alt in structured_output.alternatives:
                header_parts = [f"{alt.id}: {alt.label}"]
                if alt.change_type:
                    header_parts.append(f"[{alt.change_type}]")
                header = " ".join(header_parts)

                with st.expander(header, expanded=True):
                    st.write(alt.description)
        else:
            st.info("No alternatives available.")

    with pref_tab:
        if structured_output.preferences:
            for pref in structured_output.preferences:
                header_parts = [f"{pref.id}: {pref.label}"]
                if pref.source:
                    header_parts.append(f"[{pref.source}]")
                if pref.change_type:
                    header_parts.append(f"[{pref.change_type}]")
                header = " ".join(header_parts)

                with st.expander(header, expanded=True):
                    st.write(pref.description)
        else:
            st.info("No preferences available.")

    with unc_tab:
        if structured_output.uncertainties:
            for unc in structured_output.uncertainties:
                header_parts = [f"{unc.id}: {unc.label}"]
                if unc.type:
                    header_parts.append(f"[{unc.type}]")
                if unc.change_type:
                    header_parts.append(f"[{unc.change_type}]")
                header = " ".join(header_parts)

                with st.expander(header, expanded=True):
                    st.write(unc.description)
        else:
            st.info("No uncertainties available.")

    with ethics_tab:
        if structured_output.ethics:
            for issue in structured_output.ethics:
                header_parts = [f"{issue.id}: {issue.label}"]
                if issue.category:
                    header_parts.append(f"[{issue.category}]")
                if issue.change_type:
                    header_parts.append(f"[{issue.change_type}]")
                header = " ".join(header_parts)

                with st.expander(header, expanded=True):
                    st.write(issue.description)
        else:
            st.info("No ethics issues available.")

    with stakeholders_tab:
        if structured_output.stakeholders:
            for stakeholder in structured_output.stakeholders:
                header_parts = [f"{stakeholder.id}: {stakeholder.label}"]
                if stakeholder.impact_type:
                    header_parts.append(f"[{stakeholder.impact_type}]")
                if stakeholder.change_type:
                    header_parts.append(f"[{stakeholder.change_type}]")
                header = " ".join(header_parts)

                with st.expander(header, expanded=True):
                    st.write(stakeholder.description)
        else:
            st.info("No stakeholders available.")

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
    col1.metric("Faithfulness", evaluation.faithfulness)
    col2.metric("Completeness", evaluation.completeness)
    col3.metric("Clarity", evaluation.clarity)
    col4.metric("Usefulness", evaluation.usefulness)
    col5.metric("Self-expression", evaluation.self_expression_support)

    if evaluation.notes:
        st.markdown("**Notes**")
        st.write(evaluation.notes)

def render_ai_evaluation(round_index: int, ai_evaluation) -> None:
    st.markdown(f"### AI Improvement Evaluation for Round {round_index}")

    top1, top2, top3 = st.columns(3)
    top1.metric("Direction", ai_evaluation.direction)
    top2.metric("Improvement Score", ai_evaluation.improvement_score)
    top3.metric("Magnitude", ai_evaluation.improvement_magnitude)

    st.markdown("**Dimension Scores**")
    cols = st.columns(5)
    cols[0].metric("Faithfulness", ai_evaluation.dimension_scores.get("faithfulness", ""))
    cols[1].metric("Completeness", ai_evaluation.dimension_scores.get("completeness", ""))
    cols[2].metric("Clarity", ai_evaluation.dimension_scores.get("clarity", ""))
    cols[3].metric("Usefulness", ai_evaluation.dimension_scores.get("usefulness", ""))
    cols[4].metric("Non-distortion", ai_evaluation.dimension_scores.get("non_distortion", ""))

    st.markdown("**Dimension Changes**")
    st.write(ai_evaluation.dimension_changes)

    if ai_evaluation.new_information_used:
        st.markdown("**New Information Used**")
        for item in ai_evaluation.new_information_used:
            st.write(f"- {item}")

    if ai_evaluation.key_improvements:
        st.markdown("**Key Improvements**")
        for item in ai_evaluation.key_improvements:
            st.write(f"- {item}")

    if ai_evaluation.remaining_issues:
        st.markdown("**Remaining Issues**")
        for item in ai_evaluation.remaining_issues:
            st.write(f"- {item}")

    st.markdown("**Reasoning Summary**")
    st.write(ai_evaluation.reasoning_summary)


def render_rating_label(label: str, rubric_key: str) -> None:
    definition = USER_RATING_RUBRIC[rubric_key]["definition"].replace('"', "&quot;")
    st.markdown(
        f"""
        <div class="star-label">
            {label}
            <span class="tooltip-question" title="{definition}">?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def numeric_rating_input(label: str, key_prefix: str, rubric_key: str) -> int:
    render_rating_label(label, rubric_key)

    selected = st.radio(
        label,
        options=[1, 2, 3, 4, 5],
        index=2,
        key=key_prefix,
        horizontal=True,
        label_visibility="collapsed",
    )
    return int(selected)


def render_rating_form(round_index: int) -> None:
    st.markdown("### Evaluation")
    st.markdown(
        '<div class="star-note">Select a score from 1 to 5 for each criterion.</div>',
        unsafe_allow_html=True,
    )

    faithfulness = numeric_rating_input(
        "Faithfulness to your real situation",
        f"faithfulness_{round_index}",
        "faithfulness",
    )
    completeness = numeric_rating_input(
        "Completeness",
        f"completeness_{round_index}",
        "completeness",
    )
    clarity = numeric_rating_input(
        "Clarity",
        f"clarity_{round_index}",
        "clarity",
    )
    usefulness = numeric_rating_input(
        "Usefulness for decision-making",
        f"usefulness_{round_index}",
        "usefulness",
    )
    self_expression_support = numeric_rating_input(
        "Helped you express what you meant",
        f"self_expression_support_{round_index}",
        "self_expression_support",
    )

    notes = st.text_area(
        "Optional notes",
        key=f"notes_{round_index}",
        height=90,
        placeholder="Any comments about this round...",
    )

    if st.button(
        f"Submit Evaluation for Round {round_index}",
        key=f"submit_eval_{round_index}",
        use_container_width=True,
    ):
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

    initial_generator, question_generator, refiner, improvement_evaluator = initialize_services(model_name)

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
    st.session_state.download_ai_json_bytes = None
    st.session_state.download_ai_csv_bytes = None
    st.session_state.download_ai_json_filename = None
    st.session_state.download_ai_csv_filename = None

    st.session_state.initial_generator = initial_generator
    st.session_state.question_generator = question_generator
    st.session_state.refiner = refiner
    st.session_state.improvement_evaluator = improvement_evaluator


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

    ai_eval = st.session_state.improvement_evaluator.run(
        decision_title=manager.state.title,
        decision_narrative=manager.state.narrative,
        previous_round_index=round_index - 1,
        previous_round_output=current_round.structured_output,
        current_round_output=refined_output,
        clarification_questions=question_set,
        user_answers=user_answers,
    )
    manager.attach_ai_evaluation(round_index, ai_eval)

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
        "decision_id,round_index,n_alternatives,n_preferences,n_uncertainties,n_ethics,n_stakeholders,"
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
            str(len(round_record.structured_output.ethics)),
            str(len(round_record.structured_output.stakeholders)),
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

def build_ai_evaluation_json_download(manager: StudyStateManager) -> tuple[bytes, str]:
    payload = {
        "decision_id": manager.state.decision_id,
        "title": manager.state.title,
        "narrative": manager.state.narrative,
        "ai_evaluations": [],
    }

    for round_record in manager.state.rounds:
        if round_record.ai_evaluation is None:
            continue

        payload["ai_evaluations"].append({
            "round_index": round_record.round_index,
            "compared_to_round": round_record.ai_evaluation.compared_to_round,
            "improved": round_record.ai_evaluation.improved,
            "improvement_score": round_record.ai_evaluation.improvement_score,
            "improvement_magnitude": round_record.ai_evaluation.improvement_magnitude,
            "dimension_scores": round_record.ai_evaluation.dimension_scores,
            "dimension_changes": round_record.ai_evaluation.dimension_changes,
            "new_information_used": round_record.ai_evaluation.new_information_used,
            "key_improvements": round_record.ai_evaluation.key_improvements,
            "remaining_issues": round_record.ai_evaluation.remaining_issues,
            "reasoning_summary": round_record.ai_evaluation.reasoning_summary,
        })

    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    filename = f"{manager.state.decision_id}_ai_evaluations.json"
    return data, filename


def build_ai_evaluation_csv_download(manager: StudyStateManager) -> tuple[bytes, str]:
    output = io.StringIO()
    output.write(
        "decision_id,round_index,compared_to_round,improved,improvement_score,improvement_magnitude,"
        "faithfulness_score,completeness_score,clarity_score,usefulness_score,non_distortion_score,"
        "faithfulness_change,completeness_change,clarity_change,usefulness_change,non_distortion_change,"
        "new_information_used,key_improvements,remaining_issues,reasoning_summary\n"
    )

    for round_record in manager.state.rounds:
        ai_eval = round_record.ai_evaluation
        if ai_eval is None:
            continue

        row = [
            manager.state.decision_id,
            str(round_record.round_index),
            str(ai_eval.compared_to_round),
            str(ai_eval.improved),
            str(ai_eval.improvement_score),
            ai_eval.improvement_magnitude,
            str(ai_eval.dimension_scores.get("faithfulness", "")),
            str(ai_eval.dimension_scores.get("completeness", "")),
            str(ai_eval.dimension_scores.get("clarity", "")),
            str(ai_eval.dimension_scores.get("usefulness", "")),
            str(ai_eval.dimension_scores.get("non_distortion", "")),
            ai_eval.dimension_changes.get("faithfulness", ""),
            ai_eval.dimension_changes.get("completeness", ""),
            ai_eval.dimension_changes.get("clarity", ""),
            ai_eval.dimension_changes.get("usefulness", ""),
            ai_eval.dimension_changes.get("non_distortion", ""),
            f"\"{' | '.join(ai_eval.new_information_used).replace('\"', '\"\"')}\"",
            f"\"{' | '.join(ai_eval.key_improvements).replace('\"', '\"\"')}\"",
            f"\"{' | '.join(ai_eval.remaining_issues).replace('\"', '\"\"')}\"",
            f"\"{ai_eval.reasoning_summary.replace('\"', '\"\"')}\"",
        ]
        output.write(",".join(row) + "\n")

    filename = f"{manager.state.decision_id}_ai_evaluations.csv"
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
    ai_json_bytes, ai_json_filename = build_ai_evaluation_json_download(manager)
    ai_csv_bytes, ai_csv_filename = build_ai_evaluation_csv_download(manager)

    st.session_state.download_json_bytes = json_bytes
    st.session_state.download_csv_bytes = csv_bytes
    st.session_state.download_json_filename = json_filename
    st.session_state.download_csv_filename = csv_filename

    st.session_state.download_ai_json_bytes = ai_json_bytes
    st.session_state.download_ai_csv_bytes = ai_csv_bytes
    st.session_state.download_ai_json_filename = ai_json_filename
    st.session_state.download_ai_csv_filename = ai_csv_filename

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
        '<div class="app-subtitle">Explore whether iterative LLM questioning helps users better express alternatives, preferences, uncertainties, ethics, and stakeholders.</div>',
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
            placeholder="Describe your situation, goals, constraints, concerns, ethics, and who may be affected.",
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
        st.markdown("## AI Improvement Evaluations")

        ai_rounds = [r for r in manager.state.rounds if r.ai_evaluation is not None]
        if ai_rounds:
            for round_record in ai_rounds:
                with st.expander(f"Round {round_record.round_index} AI Evaluation",
                                 expanded=(round_record.round_index == ai_rounds[-1].round_index)):
                    render_ai_evaluation(round_record.round_index, round_record.ai_evaluation)
        else:
            st.info("No AI improvement evaluations available.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("## Download Results")
        st.success("Study completed. Download your results below.")

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.download_button(
                label="Download JSON Results",
                data=st.session_state.download_json_bytes,
                file_name=st.session_state.download_json_filename,
                mime="application/json",
                use_container_width=True,
            )
        with row1_col2:
            st.download_button(
                label="Download CSV Summary",
                data=st.session_state.download_csv_bytes,
                file_name=st.session_state.download_csv_filename,
                mime="text/csv",
                use_container_width=True,
            )

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.download_button(
                label="Download AI Evaluation JSON",
                data=st.session_state.download_ai_json_bytes,
                file_name=st.session_state.download_ai_json_filename,
                mime="application/json",
                use_container_width=True,
            )
        with row2_col2:
            st.download_button(
                label="Download AI Evaluation CSV",
                data=st.session_state.download_ai_csv_bytes,
                file_name=st.session_state.download_ai_csv_filename,
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()