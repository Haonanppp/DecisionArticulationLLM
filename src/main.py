from __future__ import annotations

import uuid
from typing import List

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
    ClarificationQuestion,
    DecisionInput,
    UserAnswer,
)
from src.pipeline.initial_generator import InitialGenerator
from src.pipeline.question_generator import QuestionGenerator
from src.pipeline.refiner import Refiner
from src.pipeline.controller import DecisionStudyController
from src.logging.logger import StudyLogger
from src.logging.export import StudyExporter
from src.evaluation.user_rating import collect_user_rating_cli
from src.utils.llm_client import OpenAILLMClient


def display_structured_output(round_index: int, structured_output) -> None:
    print(f"\n=== Round {round_index} Structured Output ===")
    print(f"\nDecision Summary:\n{structured_output.decision_summary}")

    print("\nAlternatives:")
    for alt in structured_output.alternatives:
        print(f"- {alt.id}: {alt.label}")
        print(f"  {alt.description}")

    print("\nPreferences:")
    for pref in structured_output.preferences:
        print(f"- {pref.id}: {pref.label} ({pref.source})")
        print(f"  {pref.description}")

    print("\nUncertainties:")
    for unc in structured_output.uncertainties:
        print(f"- {unc.id}: {unc.label} ({unc.type})")
        print(f"  {unc.description}")

    print("\nEthics:")
    for item in structured_output.ethics:
        print(f"- {item.id}: {item.label} ({item.category})")
        print(f"  {item.description}")

    print("\nStakeholders:")
    for item in structured_output.stakeholders:
        print(f"- {item.id}: {item.label} ({item.impact_type})")
        print(f"  {item.description}")

    if structured_output.missing_but_relevant_information:
        print("\nMissing but Relevant Information:")
        for item in structured_output.missing_but_relevant_information:
            print(f"- {item}")

    if structured_output.refinement_notes:
        print("\nRefinement Notes:")
        for note in structured_output.refinement_notes:
            print(f"- {note}")


def answer_provider(round_index: int, questions: List[ClarificationQuestion]) -> List[UserAnswer]:
    print(f"\n--- Clarification Round {round_index} ---")
    answers = []
    for q in questions:
        print(f"\n{q.id}: {q.question}")
        user_answer = input("Your answer: ").strip()
        answers.append(
            UserAnswer(
                question_id=q.id,
                question_text=q.question,
                answer=user_answer,
            )
        )
    return answers


def main() -> None:
    print("Decision Articulation Study Demo")
    title = input("Decision title: ").strip()
    narrative = input("Decision narrative: ").strip()

    decision_input = DecisionInput(
        decision_id=str(uuid.uuid4()),
        title=title,
        narrative=narrative,
    )

    llm_client = OpenAILLMClient(model_name=DEFAULT_MODEL_NAME)

    initial_generator = InitialGenerator(llm_client, INITIAL_GENERATION_PROMPT_PATH)
    question_generator = QuestionGenerator(llm_client, QUESTION_GENERATION_PROMPT_PATH)
    refiner = Refiner(llm_client, REFINEMENT_PROMPT_PATH)

    controller = DecisionStudyController(
        model_name=DEFAULT_MODEL_NAME,
        initial_generator=initial_generator,
        question_generator=question_generator,
        refiner=refiner,
        max_rounds=DEFAULT_MAX_ROUNDS,
    )

    manager = controller.run(
        decision_input=decision_input,
        answer_provider=answer_provider,
        evaluation_provider=collect_user_rating_cli,
        round_display_callback=display_structured_output,
    )

    logger = StudyLogger(LOGS_DIR)
    log_path = logger.save_state(manager)

    exporter = StudyExporter()
    csv_path = exporter.export_round_summary_csv(
        manager,
        PROCESSED_DIR / f"{decision_input.decision_id}_round_summary.csv",
    )

    print("\nStudy completed.")
    print(f"Saved JSON log to: {log_path}")
    print(f"Saved CSV summary to: {csv_path}")


if __name__ == "__main__":
    main()