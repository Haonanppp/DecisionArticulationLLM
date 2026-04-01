from __future__ import annotations

import csv
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