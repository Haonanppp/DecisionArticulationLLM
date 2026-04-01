from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.models.schemas import (
    ClarificationQuestionSet,
    StructuredDecisionOutput,
)


class QuestionGenerator:
    def __init__(self, llm_client: Any, prompt_path: Path):
        self.llm_client = llm_client
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    def build_prompt(
        self,
        decision_title: str,
        decision_narrative: str,
        current_structured_output: StructuredDecisionOutput,
        prior_qa_history: str,
    ) -> str:
        prompt = self.prompt_template
        prompt = prompt.replace("{decision_title}", decision_title)
        prompt = prompt.replace("{decision_narrative}", decision_narrative)
        prompt = prompt.replace(
            "{current_structured_output}",
            current_structured_output.model_dump_json(indent=2),
        )
        prompt = prompt.replace("{prior_qa_history}", prior_qa_history)
        return prompt

    @staticmethod
    def _normalize_target(value: str) -> str:
        value = value.strip().lower()

        mapping = {
            "alternative": "alternatives",
            "alternatives": "alternatives",
            "option": "alternatives",
            "options": "alternatives",

            "preference": "preferences",
            "preferences": "preferences",
            "priority": "preferences",
            "priorities": "preferences",
            "constraint": "preferences",
            "constraints": "preferences",
            "tradeoff": "preferences",
            "tradeoffs": "preferences",
            "trade-off": "preferences",
            "trade-offs": "preferences",

            "uncertainty": "uncertainties",
            "uncertainties": "uncertainties",
            "risk": "uncertainties",
            "risks": "uncertainties",

            "ethic": "ethics",
            "ethics": "ethics",
            "ethical": "ethics",
            "ethical issue": "ethics",
            "ethical issues": "ethics",
            "morality": "ethics",
            "moral": "ethics",

            "stakeholder": "stakeholders",
            "stakeholders": "stakeholders",
            "affected party": "stakeholders",
            "affected parties": "stakeholders",

            "mixed": "mixed",
        }

        return mapping.get(value, "mixed")

    @staticmethod
    def _normalize_question_type(value: str) -> str:
        value = value.strip().lower()

        allowed = {
            "open-ended",
            "constraint",
            "trade-off",
            "ranking",
            "feasibility",
            "uncertainty",
            "ethics",
            "stakeholder",
        }

        mapping = {
            "open ended": "open-ended",
            "open": "open-ended",
            "constraints": "constraint",
            "tradeoff": "trade-off",
            "tradeoffs": "trade-off",
            "trade off": "trade-off",
            "rank": "ranking",
            "feasible": "feasibility",
            "risk": "uncertainty",
            "ethical": "ethics",
            "moral": "ethics",
            "stakeholders": "stakeholder",
        }

        value = mapping.get(value, value)
        return value if value in allowed else "open-ended"

    def _normalize_response(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        questions = response_json.get("questions", [])
        for q in questions:
            if "target" in q and isinstance(q["target"], str):
                q["target"] = self._normalize_target(q["target"])
            else:
                q["target"] = "mixed"

            if "question_type" in q and isinstance(q["question_type"], str):
                q["question_type"] = self._normalize_question_type(q["question_type"])
            else:
                q["question_type"] = "open-ended"

        return response_json

    def run(
        self,
        decision_title: str,
        decision_narrative: str,
        current_structured_output: StructuredDecisionOutput,
        prior_qa_history: str,
    ) -> ClarificationQuestionSet:
        prompt = self.build_prompt(
            decision_title=decision_title,
            decision_narrative=decision_narrative,
            current_structured_output=current_structured_output,
            prior_qa_history=prior_qa_history,
        )
        response_json: Dict[str, Any] = self.llm_client.generate_json(prompt)
        response_json = self._normalize_response(response_json)
        return ClarificationQuestionSet(**response_json)