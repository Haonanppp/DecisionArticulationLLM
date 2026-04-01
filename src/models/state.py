from __future__ import annotations

from typing import List, Optional
from src.models.schemas import (
    DecisionInput,
    DecisionStudyState,
    RoundEvaluation,
    RoundRecord,
    StructuredDecisionOutput,
    ClarificationQuestionSet,
    UserAnswer,
)


class StudyStateManager:
    def __init__(self, decision_input: DecisionInput, model_name: str):
        self.state = DecisionStudyState(
            decision_id=decision_input.decision_id,
            title=decision_input.title,
            narrative=decision_input.narrative,
            metadata={"model_name": model_name},
        )

    def add_initial_round(self, structured_output: StructuredDecisionOutput) -> None:
        self.state.rounds.append(
            RoundRecord(
                round_index=0,
                structured_output=structured_output.model_dump(),
            )
        )

    def add_followup_round(
        self,
        round_index: int,
        structured_output: StructuredDecisionOutput,
        question_set: ClarificationQuestionSet,
        user_answers: List[UserAnswer],
    ) -> None:
        self.state.rounds.append(
            RoundRecord(
                round_index=round_index,
                questions=[q.model_dump() for q in question_set.questions],
                user_answers=[a.model_dump() for a in user_answers],
                structured_output=structured_output.model_dump(),
            )
        )

    def attach_evaluation(self, round_index: int, evaluation: RoundEvaluation) -> None:
        for round_record in self.state.rounds:
            if round_record.round_index == round_index:
                round_record.evaluation = evaluation
                return
        raise ValueError(f"Round {round_index} not found.")

    def get_current_round(self) -> Optional[RoundRecord]:
        if not self.state.rounds:
            return None
        return self.state.rounds[-1]

    def get_prior_qa_history_as_text(self) -> str:
        if len(self.state.rounds) <= 1:
            return "No prior clarification history."

        blocks = []
        for round_record in self.state.rounds[1:]:
            blocks.append(f"Round {round_record.round_index}:")
            for q in round_record.questions:
                blocks.append(f"Q ({q.id}): {q.question}")
            for a in round_record.user_answers:
                blocks.append(f"A ({a.question_id}): {a.answer}")
        return "\n".join(blocks)

    def as_dict(self) -> dict:
        return self.state.model_dump()