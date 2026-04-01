from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.models.schemas import DecisionInput, StructuredDecisionOutput


class InitialGenerator:
    def __init__(self, llm_client: Any, prompt_path: Path):
        self.llm_client = llm_client
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    def build_prompt(self, decision_input: DecisionInput) -> str:
        prompt = self.prompt_template
        prompt = prompt.replace("{decision_title}", decision_input.title)
        prompt = prompt.replace("{decision_narrative}", decision_input.narrative)
        return prompt

    def run(self, decision_input: DecisionInput) -> StructuredDecisionOutput:
        prompt = self.build_prompt(decision_input)
        response_json: Dict[str, Any] = self.llm_client.generate_json(prompt)
        return StructuredDecisionOutput(**response_json)