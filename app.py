from __future__ import annotations

import uuid
from typing import List, Dict

import streamlit as st

from src.config import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MODEL_NAME,
    INITIAL_GENERATION_PROMPT_PATH,
    QUESTION_GENERATION_PROMPT_PATH,
    REFINEMENT_PROMPT_PATH,
    LOGS_DIR,
    PROCESSED_DIR,
)
from src.models.schemas import (
    DecisionInput,
    RoundEvaluation,
    StructuredDecisionOutput,
    ClarificationQuestionSet,
    UserAnswer,
)
from src.models.state import StudyStateManager
from src.pipeline.initial_generator import InitialGenerator
from src.pipeline.question_generator import QuestionGenerator
from src.pipeline.refiner import Refiner
from src.logging.logger import StudyLogger
from src.logging.export import StudyExporter
from src.utils.llm_client import OpenAILLMClient


st.set_page_config(page_title="Decision Articulation Study", layout="wide")


def initialize_services():
    llm_client = OpenAILLMClient(model_name=DEFAULT_MODEL_NAME)
    initial_generator = InitialGenerator(llm_client, INITIAL_GENERATION_PROMPT_PATH)
    question_generator = QuestionGenerator(llm_client, QUESTION_GENERATION_PROMPT_PATH)
    refiner = Refiner(llm_client, REFINEMENT_PROMPT_PATH)
    return initial_generator, question_generator, refiner


def initialize_session_state():
    if "study_started" not in st.session_state:
        st.session_state.study_started = False

    if "study_completed" not in st.session_state:
        st.session_state.study_completed = False

    if "decision_input" not in st.session_state:
        st.session_state.decision_input = None

    if "manager" not in st.session_state:
        st.session_state.manager = None

    if "current_question_set" not in st.session_state:
        st.session_state.current_question_set = None

    if "current_round_index" not in st.session_state:
        st.session_state.current_round_index = 0

    if "rating_submitted_rounds" not in st.session_state:
        st.session_state.rating_submitted_rounds = set()

    if "answers_submitted_rounds" not in st.session_state:
        st.session_state.answers_submitted_rounds = set()

    if "saved_json_path" not in st.session_state:
        st.session_state.saved_json_path = None

    if "saved_csv_path" not in st.session_state:
        st.session_state.saved_csv_path = None


def render_structured_output(round_index: int, structured_output: StructuredDecisionOutput):
    st.subheader(f"Round {round_index} Structured Output")

    st.markdown("**Decision Summary**")
    st.write(structured_output.decision_summary)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Alternatives**")
        for alt in structured_output.alternatives:
            label = f"{alt.id}: {alt.label}"
            if alt.change_type:
                label += f" [{alt.change_type}]"
            with st.expander(label, expanded=True):
                st.write(alt.description)

    with col2:
        st.markdown("**Preferences**")
        for pref in structured_output.preferences:
            label = f"{pref.id}: {pref.label}"
            if pref.source:
                label += f" ({pref.source})"
            if pref.change_type:
                label += f" [{pref.change_type}]"
            with st.expander(label, expanded=True):
                st.write(pref.description)

    with col3:
        st.markdown("**Uncertainties**")
        for unc in structured_output.uncertainties:
            label = f"{unc.id}: {unc.label}"
            if unc.type:
                label += f" ({unc.type})"
            if unc.change_type:
                label += f" [{unc.change_type}]"
            with st.expander(label, expanded=True):
                st.write(unc.description)

    if structured_output.missing_but_relevant_information:
        st.markdown("**Missing but Relevant Information**")
        for item in structured_output.missing_but_relevant_information:
            st.write(f"- {item}")

    if structured_output.refinement_notes:
        st.markdown("**Refinement Notes**")
        for note in structured_output.refinement_notes:
            st.write(f"- {note}")


def render_questions(question_set: ClarificationQuestionSet, round_index: int):
    st.subheader(f"Clarification Questions for Round {round_index}")
    st.caption(question_set.round_goal)

    answers: Dict[str, str] = {}
    for q in question_set.questions:
        with st.expander(f"{q.id}: {q.question}", expanded=True):
            st.write(f"**Target:** {q.target}")
            st.write(f"**Type:** {q.question_type}")
            st.write(f"**Why this is asked:** {q.rationale}")
            answers[q.id] = st.text_area(
                label=f"Answer for {q.id}",
                key=f"answer_round_{round_index}_{q.id}",
                height=100,
            )
    return answers


def render_rating_form(round_index: int):
    st.subheader(f"Evaluation for Round {round_index}")
    st.caption("Please rate each item from 1 to 5.")

    faithfulness = st.slider(
        "Faithfulness to your real situation",
        min_value=1,
        max_value=5,
        value=3,
        key=f"faithfulness_{round_index}",
    )
    completeness = st.slider(
        "Completeness",
        min_value=1,
        max_value=5,
        value=3,
        key=f"completeness_{round_index}",
    )
    clarity = st.slider(
        "Clarity",
        min_value=1,
        max_value=5,
        value=3,
        key=f"clarity_{round_index}",
    )
    usefulness = st.slider(
        "Usefulness for decision-making",
        min_value=1,
        max_value=5,
        value=3,
        key=f"usefulness_{round_index}",
    )
    self_expression_support = st.slider(
        "Helped you express what you meant",
        min_value=1,
        max_value=5,
        value=3,
        key=f"self_expression_support_{round_index}",
    )
    notes = st.text_area(
        "Optional notes",
        key=f"notes_{round_index}",
        height=100,
    )

    submitted = st.button(
        f"Submit Evaluation for Round {round_index}",
        key=f"submit_eval_{round_index}",
    )

    if submitted:
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


def save_results():
    logger = StudyLogger(LOGS_DIR)
    exporter = StudyExporter()

    json_path = logger.save_state(st.session_state.manager)
    csv_path = exporter.export_round_summary_csv(
        st.session_state.manager,
        PROCESSED_DIR / f"{st.session_state.manager.state.decision_id}_round_summary.csv",
    )

    st.session_state.saved_json_path = str(json_path)
    st.session_state.saved_csv_path = str(csv_path)


def start_study(title: str, narrative: str):
    decision_input = DecisionInput(
        decision_id=str(uuid.uuid4()),
        title=title,
        narrative=narrative,
    )

    initial_generator, question_generator, refiner = initialize_services()

    manager = StudyStateManager(
        decision_input=decision_input,
        model_name=DEFAULT_MODEL_NAME,
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
    st.session_state.saved_json_path = None
    st.session_state.saved_csv_path = None

    st.session_state.initial_generator = initial_generator
    st.session_state.question_generator = question_generator
    st.session_state.refiner = refiner


def generate_questions_for_next_round():
    manager = st.session_state.manager
    current_round = manager.get_current_round()

    if current_round is None:
        st.error("No current round found.")
        return

    if current_round.round_index not in st.session_state.rating_submitted_rounds:
        st.warning("Please submit the current round evaluation before generating the next question set.")
        return

    if current_round.round_index >= DEFAULT_MAX_ROUNDS:
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
    st.info(f"Generated clarification questions for Round {current_round.round_index + 1}.")


def submit_answers_and_refine(round_index: int, raw_answers: Dict[str, str]):
    manager = st.session_state.manager
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
    st.success(f"Round {round_index} completed.")


def complete_study():
    latest_round_index = st.session_state.manager.get_current_round().round_index
    if latest_round_index not in st.session_state.rating_submitted_rounds:
        st.warning("Please submit the latest round evaluation before finishing the study.")
        return

    save_results()
    st.session_state.study_completed = True
    st.success("Study completed and results saved.")


def main():
    initialize_session_state()

    st.title("Decision Articulation Study")
    st.write(
        "This app studies whether iterative LLM questioning helps users better express "
        "their decision situation, preferences, and uncertainties."
    )

    with st.sidebar:
        st.header("Study Settings")
        st.write(f"**Model:** {DEFAULT_MODEL_NAME}")
        st.write(f"**Max Rounds:** {DEFAULT_MAX_ROUNDS}")

        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if not st.session_state.study_started:
        st.subheader("Start a New Decision Study")

        title = st.text_input("Decision Title")
        narrative = st.text_area("Decision Narrative", height=180)

        if st.button("Start Study"):
            if not title.strip() or not narrative.strip():
                st.warning("Please provide both a decision title and a decision narrative.")
            else:
                with st.spinner("Generating initial structured decision output..."):
                    start_study(title.strip(), narrative.strip())
                st.rerun()

        return

    manager: StudyStateManager = st.session_state.manager

    st.divider()
    st.markdown("## Decision Input")
    st.write(f"**Title:** {manager.state.title}")
    st.write(f"**Narrative:** {manager.state.narrative}")

    st.divider()
    st.markdown("## Round History")

    for round_record in manager.state.rounds:
        with st.container(border=True):
            render_structured_output(round_record.round_index, round_record.structured_output)

            if round_record.evaluation:
                st.markdown("**Submitted Evaluation**")
                ev = round_record.evaluation
                st.write(
                    {
                        "faithfulness": ev.faithfulness,
                        "completeness": ev.completeness,
                        "clarity": ev.clarity,
                        "usefulness": ev.usefulness,
                        "self_expression_support": ev.self_expression_support,
                        "notes": ev.notes,
                    }
                )
            elif round_record.round_index not in st.session_state.rating_submitted_rounds:
                render_rating_form(round_record.round_index)

    current_round = manager.get_current_round()
    current_round_index = current_round.round_index if current_round else 0

    if current_round_index < DEFAULT_MAX_ROUNDS:
        st.divider()
        st.markdown("## Next Clarification Step")

        if st.session_state.current_question_set is None:
            if current_round_index in st.session_state.rating_submitted_rounds:
                if st.button(f"Generate Questions for Round {current_round_index + 1}"):
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

            if st.button(f"Submit Answers and Generate Round {current_round_index + 1} Output"):
                with st.spinner("Refining structured decision output..."):
                    submit_answers_and_refine(current_round_index + 1, raw_answers)
                st.rerun()

    else:
        st.divider()
        st.markdown("## Finish Study")
        if st.button("Complete Study and Save Results"):
            complete_study()
            st.rerun()

    if st.session_state.study_completed:
        st.divider()
        st.markdown("## Saved Results")
        st.write(f"**JSON Log:** `{st.session_state.saved_json_path}`")
        st.write(f"**CSV Summary:** `{st.session_state.saved_csv_path}`")


if __name__ == "__main__":
    main()