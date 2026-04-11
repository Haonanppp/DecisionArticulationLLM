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

    @staticmethod
    def _normalize_direction_and_magnitude(response_json: Dict[str, Any]) -> Dict[str, Any]:
        score = response_json.get("improvement_score")

        if "direction" not in response_json:
            if isinstance(score, int):
                if score > 0:
                    response_json["direction"] = "improved"
                elif score < 0:
                    response_json["direction"] = "worsened"
                else:
                    response_json["direction"] = "unchanged"
            else:
                response_json["direction"] = "unchanged"

        if "improvement_magnitude" not in response_json and isinstance(score, int):
            abs_score = abs(score)
            if score == 0:
                response_json["improvement_magnitude"] = "none"
            elif score > 0:
                if abs_score <= 1:
                    response_json["improvement_magnitude"] = "slight_positive"
                elif abs_score <= 3:
                    response_json["improvement_magnitude"] = "moderate_positive"
                else:
                    response_json["improvement_magnitude"] = "strong_positive"
            else:
                if abs_score <= 1:
                    response_json["improvement_magnitude"] = "slight_negative"
                elif abs_score <= 3:
                    response_json["improvement_magnitude"] = "moderate_negative"
                else:
                    response_json["improvement_magnitude"] = "strong_negative"

        return response_json

    @staticmethod
    def _normalize_dimension_scores(response_json: Dict[str, Any]) -> Dict[str, Any]:
        dimension_scores = response_json.get("dimension_scores", {})
        for key in ["faithfulness", "completeness", "clarity", "usefulness", "non_distortion"]:
            if key not in dimension_scores:
                dimension_scores[key] = 0
        response_json["dimension_scores"] = dimension_scores

        dimension_changes = response_json.get("dimension_changes", {})
        for key in ["faithfulness", "completeness", "clarity", "usefulness", "non_distortion"]:
            if key not in dimension_changes:
                dimension_changes[key] = "unchanged"
        response_json["dimension_changes"] = dimension_changes

        return response_json

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
        response_json = self._normalize_direction_and_magnitude(response_json)
        response_json = self._normalize_dimension_scores(response_json)
        return RoundImprovementEvaluation(**response_json)