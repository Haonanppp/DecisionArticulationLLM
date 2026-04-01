from __future__ import annotations

from typing import Dict, List

from src.models.state import StudyStateManager


def average_round_scores(manager: StudyStateManager) -> Dict[int, float]:
    results = {}
    for round_record in manager.state.rounds:
        if not round_record.evaluation:
            continue

        ev = round_record.evaluation
        avg = (
            ev.faithfulness
            + ev.completeness
            + ev.clarity
            + ev.usefulness
            + ev.self_expression_support
        ) / 5.0
        results[round_record.round_index] = avg

    return results


def improvement_from_round0(manager: StudyStateManager) -> Dict[int, float]:
    averages = average_round_scores(manager)
    if 0 not in averages:
        return {}

    base = averages[0]
    return {round_idx: avg - base for round_idx, avg in averages.items()}