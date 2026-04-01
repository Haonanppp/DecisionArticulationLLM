from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.models.schemas import (
    ClarificationQuestionSet,
    StructuredDecisionOutput,
    UserAnswer,
)


class Refiner:
    def __init__(self, llm_client: Any, prompt_path: Path):
        self.llm_client = llm_client
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    def build_prompt(
        self,
        decision_title: str,
        decision_narrative: str,
        previous_structured_output: StructuredDecisionOutput,
        current_questions: ClarificationQuestionSet,
        user_answers: List[UserAnswer],
        prior_qa_history: str,
    ) -> str:
        prompt = self.prompt_template
        prompt = prompt.replace("{decision_title}", decision_title)
        prompt = prompt.replace("{decision_narrative}", decision_narrative)
        prompt = prompt.replace(
            "{previous_structured_output}",
            previous_structured_output.model_dump_json(indent=2),
        )
        prompt = prompt.replace(
            "{current_questions}",
            current_questions.model_dump_json(indent=2),
        )
        prompt = prompt.replace(
            "{user_answers}",
            json.dumps([answer.model_dump() for answer in user_answers], ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace("{prior_qa_history}", prior_qa_history)
        return prompt

    def run(
        self,
        decision_title: str,
        decision_narrative: str,
        previous_structured_output: StructuredDecisionOutput,
        current_questions: ClarificationQuestionSet,
        user_answers: List[UserAnswer],
        prior_qa_history: str,
    ) -> StructuredDecisionOutput:
        prompt = self.build_prompt(
            decision_title=decision_title,
            decision_narrative=decision_narrative,
            previous_structured_output=previous_structured_output,
            current_questions=current_questions,
            user_answers=user_answers,
            prior_qa_history=prior_qa_history,
        )
        response_json: Dict[str, Any] = self.llm_client.generate_json(prompt)
        return StructuredDecisionOutput(**response_json)