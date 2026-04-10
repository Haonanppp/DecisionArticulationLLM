from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


PreferenceSource = Literal["explicit", "implicit"]
UncertaintyType = Literal["external", "personal", "informational"]
ChangeType = Literal["unchanged", "revised", "added"]

EthicsCategory = Literal[
    "deception",
    "harm",
    "stealing",
    "fairness",
    "responsibility",
    "none",
    "unclear",
]

ImpactType = Literal["direct", "indirect"]


class Alternative(BaseModel):
    id: str
    label: str
    description: str
    change_type: Optional[ChangeType] = None


class Preference(BaseModel):
    id: str
    label: str
    description: str
    source: PreferenceSource
    change_type: Optional[ChangeType] = None


class Uncertainty(BaseModel):
    id: str
    label: str
    description: str
    type: UncertaintyType
    change_type: Optional[ChangeType] = None


class EthicalIssue(BaseModel):
    id: str
    label: str
    description: str
    category: EthicsCategory
    change_type: Optional[ChangeType] = None


class Stakeholder(BaseModel):
    id: str
    label: str
    description: str
    impact_type: ImpactType
    change_type: Optional[ChangeType] = None


class StructuredDecisionOutput(BaseModel):
    decision_summary: str
    alternatives: List[Alternative]
    preferences: List[Preference]
    uncertainties: List[Uncertainty]
    ethics: List[EthicalIssue] = Field(default_factory=list)
    stakeholders: List[Stakeholder] = Field(default_factory=list)
    missing_but_relevant_information: List[str] = Field(default_factory=list)
    removed_items: Optional[Dict[str, List[str]]] = None
    refinement_notes: Optional[List[str]] = None


class ClarificationQuestion(BaseModel):
    id: str
    question: str
    target: Literal[
        "alternatives",
        "preferences",
        "uncertainties",
        "ethics",
        "stakeholders",
        "mixed",
    ]
    rationale: str
    question_type: Literal[
        "open-ended",
        "constraint",
        "trade-off",
        "ranking",
        "feasibility",
        "uncertainty",
        "ethics",
        "stakeholder",
    ]


class ClarificationQuestionSet(BaseModel):
    round_goal: str
    questions: List[ClarificationQuestion]


class UserAnswer(BaseModel):
    question_id: str
    question_text: str
    answer: str


class RoundEvaluation(BaseModel):
    round_index: int
    faithfulness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    usefulness: int = Field(ge=1, le=5)
    self_expression_support: int = Field(ge=1, le=5)
    notes: Optional[str] = None


class RoundImprovementEvaluation(BaseModel):
    compared_to_round: int
    improved: bool
    improvement_score: int = Field(ge=1, le=5)
    improvement_magnitude: Literal["marginal", "moderate", "substantial", "none", "negative"]
    dimension_scores: Dict[str, int]
    dimension_changes: Dict[str, Literal["improved", "unchanged", "worsened"]]
    new_information_used: List[str] = Field(default_factory=list)
    key_improvements: List[str] = Field(default_factory=list)
    remaining_issues: List[str] = Field(default_factory=list)
    reasoning_summary: str


class DecisionInput(BaseModel):
    decision_id: str
    title: str
    narrative: str


class RoundRecord(BaseModel):
    round_index: int
    questions: List[ClarificationQuestion] = Field(default_factory=list)
    user_answers: List[UserAnswer] = Field(default_factory=list)
    structured_output: StructuredDecisionOutput
    evaluation: Optional[RoundEvaluation] = None
    ai_evaluation: Optional[RoundImprovementEvaluation] = None


class DecisionStudyState(BaseModel):
    decision_id: str
    title: str
    narrative: str
    rounds: List[RoundRecord] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)