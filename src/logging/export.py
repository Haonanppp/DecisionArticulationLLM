from __future__ import annotations

import csv
import json
from pathlib import Path

from src.models.state import StudyStateManager


class StudyExporter:
    def export_round_summary_csv(self, manager: StudyStateManager, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "decision_id",
                "round_index",
                "n_alternatives",
                "n_preferences",
                "n_uncertainties",
                "n_ethics",
                "n_stakeholders",
                "faithfulness",
                "completeness",
                "clarity",
                "usefulness",
                "self_expression_support",
                "notes",
            ])

            for round_record in manager.state.rounds:
                eval_obj = round_record.evaluation
                writer.writerow([
                    manager.state.decision_id,
                    round_record.round_index,
                    len(round_record.structured_output.alternatives),
                    len(round_record.structured_output.preferences),
                    len(round_record.structured_output.uncertainties),
                    len(round_record.structured_output.ethics),
                    len(round_record.structured_output.stakeholders),
                    eval_obj.faithfulness if eval_obj else "",
                    eval_obj.completeness if eval_obj else "",
                    eval_obj.clarity if eval_obj else "",
                    eval_obj.usefulness if eval_obj else "",
                    eval_obj.self_expression_support if eval_obj else "",
                    eval_obj.notes if eval_obj else "",
                ])

        return output_path

    def export_ai_evaluations_json(self, manager: StudyStateManager, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "decision_id": manager.state.decision_id,
            "title": manager.state.title,
            "narrative": manager.state.narrative,
            "ai_evaluations": [],
        }

        for round_record in manager.state.rounds:
            if round_record.ai_evaluation is None:
                continue

            payload["ai_evaluations"].append({
                "round_index": round_record.round_index,
                "compared_to_round": round_record.ai_evaluation.compared_to_round,
                "direction": round_record.ai_evaluation.direction,
                "improvement_score": round_record.ai_evaluation.improvement_score,
                "improvement_magnitude": round_record.ai_evaluation.improvement_magnitude,
                "dimension_scores": round_record.ai_evaluation.dimension_scores,
                "dimension_changes": round_record.ai_evaluation.dimension_changes,
                "new_information_used": round_record.ai_evaluation.new_information_used,
                "key_improvements": round_record.ai_evaluation.key_improvements,
                "remaining_issues": round_record.ai_evaluation.remaining_issues,
                "reasoning_summary": round_record.ai_evaluation.reasoning_summary,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return output_path

    def export_ai_evaluations_csv(self, manager: StudyStateManager, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "decision_id",
                "round_index",
                "compared_to_round",
                "direction",
                "improvement_score",
                "improvement_magnitude",
                "faithfulness_score",
                "completeness_score",
                "clarity_score",
                "usefulness_score",
                "non_distortion_score",
                "faithfulness_change",
                "completeness_change",
                "clarity_change",
                "usefulness_change",
                "non_distortion_change",
                "new_information_used",
                "key_improvements",
                "remaining_issues",
                "reasoning_summary",
            ])

            for round_record in manager.state.rounds:
                ai_eval = round_record.ai_evaluation
                if ai_eval is None:
                    continue

                writer.writerow([
                    manager.state.decision_id,
                    round_record.round_index,
                    ai_eval.compared_to_round,
                    ai_eval.direction,
                    ai_eval.improvement_score,
                    ai_eval.improvement_magnitude,
                    ai_eval.dimension_scores.get("faithfulness", ""),
                    ai_eval.dimension_scores.get("completeness", ""),
                    ai_eval.dimension_scores.get("clarity", ""),
                    ai_eval.dimension_scores.get("usefulness", ""),
                    ai_eval.dimension_scores.get("non_distortion", ""),
                    ai_eval.dimension_changes.get("faithfulness", ""),
                    ai_eval.dimension_changes.get("completeness", ""),
                    ai_eval.dimension_changes.get("clarity", ""),
                    ai_eval.dimension_changes.get("usefulness", ""),
                    ai_eval.dimension_changes.get("non_distortion", ""),
                    " | ".join(ai_eval.new_information_used),
                    " | ".join(ai_eval.key_improvements),
                    " | ".join(ai_eval.remaining_issues),
                    ai_eval.reasoning_summary,
                ])

        return output_path