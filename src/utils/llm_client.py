from __future__ import annotations

import json
from typing import Any, Dict

import streamlit as st
from openai import OpenAI


class OpenAILLMClient:
    def __init__(self, model_name: str):
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing in Streamlit secrets.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )

        output_text = response.output_text
        if not output_text:
            raise ValueError("Model returned empty output.")

        try:
            return json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse model output as JSON: {output_text}") from exc