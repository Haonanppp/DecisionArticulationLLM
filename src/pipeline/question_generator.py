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
        return ClarificationQuestionSet(**response_json)