from __future__ import annotations

from typing import Callable, List, Optional

from src.models.schemas import (
    DecisionInput,
    RoundEvaluation,
    UserAnswer,
)
from src.models.state import StudyStateManager
from src.pipeline.improvement_evaluator import ImprovementEvaluator
from src.pipeline.initial_generator import InitialGenerator
from src.pipeline.question_generator import QuestionGenerator
from src.pipeline.refiner import Refiner


class DecisionStudyController:
    def __init__(
        self,
        model_name: str,
        initial_generator: InitialGenerator,
        question_generator: QuestionGenerator,
        refiner: Refiner,
        improvement_evaluator: ImprovementEvaluator,
        max_rounds: int = 2,
    ):
        self.model_name = model_name
        self.initial_generator = initial_generator
        self.question_generator = question_generator
        self.refiner = refiner
        self.improvement_evaluator = improvement_evaluator
        self.max_rounds = max_rounds

    def run(
        self,
        decision_input: DecisionInput,
        answer_provider: Callable[[int, list], List[UserAnswer]],
        evaluation_provider: Callable[[int], RoundEvaluation],
        round_display_callback: Optional[Callable[[int, object], None]] = None,
    ) -> StudyStateManager:
        manager = StudyStateManager(decision_input=decision_input, model_name=self.model_name)

        initial_output = self.initial_generator.run(decision_input)
        manager.add_initial_round(initial_output)

        if round_display_callback:
            round_display_callback(0, initial_output)

        initial_eval = evaluation_provider(0)
        manager.attach_evaluation(0, initial_eval)

        for round_index in range(1, self.max_rounds + 1):
            previous_round = manager.get_current_round()
            if previous_round is None:
                raise RuntimeError("No previous round found.")

            prior_history = manager.get_prior_qa_history_as_text()

            question_set = self.question_generator.run(
                decision_title=manager.state.title,
                decision_narrative=manager.state.narrative,
                current_structured_output=previous_round.structured_output,
                prior_qa_history=prior_history,
            )

            user_answers = answer_provider(round_index, question_set.questions)

            refined_output = self.refiner.run(
                decision_title=manager.state.title,
                decision_narrative=manager.state.narrative,
                previous_structured_output=previous_round.structured_output,
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

            ai_eval = self.improvement_evaluator.run(
                decision_title=manager.state.title,
                decision_narrative=manager.state.narrative,
                previous_round_index=round_index - 1,
                previous_round_output=previous_round.structured_output,
                current_round_output=refined_output,
                clarification_questions=question_set,
                user_answers=user_answers,
            )
            manager.attach_ai_evaluation(round_index, ai_eval)

            if round_display_callback:
                round_display_callback(round_index, refined_output)

            round_eval = evaluation_provider(round_index)
            manager.attach_evaluation(round_index, round_eval)

        return manager