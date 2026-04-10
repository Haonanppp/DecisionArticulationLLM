from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.models.schemas import (
    ClarificationQuestionSet,
    RoundImprovementEvaluation,
    StructuredDecisionOutput,
    UserAnswer,
)


class ImprovementEvaluator:
    def __init__(self, llm_client: Any, prompt_path: Path):
        self.llm_client = llm_client
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    def build_prompt(
        self,
        decision_title: str,
        decision_narrative: str,
        previous_round_output: StructuredDecisionOutput,
        current_round_output: StructuredDecisionOutput,
        clarification_questions: ClarificationQuestionSet,
        user_answers: List[UserAnswer],
    ) -> str:
        prompt = self.prompt_template
        prompt = prompt.replace("{decision_title}", decision_title)
        prompt = prompt.replace("{decision_narrative}", decision_narrative)
        prompt = prompt.replace(
            "{previous_round_output}",
            previous_round_output.model_dump_json(indent=2),
        )
        prompt = prompt.replace(
            "{current_round_output}",
            current_round_output.model_dump_json(indent=2),
        )
        prompt = prompt.replace(
            "{clarification_questions}",
            clarification_questions.model_dump_json(indent=2),
        )
        prompt = prompt.replace(
            "{user_answers}",
            json.dumps([a.model_dump() for a in user_answers], ensure_ascii=False, indent=2),
        )
        return prompt

    def run(
        self,
        decision_title: str,
        decision_narrative: str,
        previous_round_index: int,
        previous_round_output: StructuredDecisionOutput,
        current_round_output: StructuredDecisionOutput,
        clarification_questions: ClarificationQuestionSet,
        user_answers: List[UserAnswer],
    ) -> RoundImprovementEvaluation:
        prompt = self.build_prompt(
            decision_title=decision_title,
            decision_narrative=decision_narrative,
            previous_round_output=previous_round_output,
            current_round_output=current_round_output,
            clarification_questions=clarification_questions,
            user_answers=user_answers,
        )
        response_json: Dict[str, Any] = self.llm_client.generate_json(prompt)
        response_json["compared_to_round"] = previous_round_index
        return RoundImprovementEvaluation(**response_json)