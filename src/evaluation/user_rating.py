from __future__ import annotations

from src.models.schemas import RoundEvaluation


def collect_user_rating_cli(round_index: int) -> RoundEvaluation:
    print(f"\n--- Evaluation for Round {round_index} ---")
    print("Please rate each item from 1 to 5.")

    faithfulness = int(input("Faithfulness to your real situation: "))
    completeness = int(input("Completeness: "))
    clarity = int(input("Clarity: "))
    usefulness = int(input("Usefulness for decision-making: "))
    self_expression_support = int(input("Helped you express what you meant: "))
    notes = input("Optional notes: ").strip()

    return RoundEvaluation(
        round_index=round_index,
        faithfulness=faithfulness,
        completeness=completeness,
        clarity=clarity,
        usefulness=usefulness,
        self_expression_support=self_expression_support,
        notes=notes or None,
    )