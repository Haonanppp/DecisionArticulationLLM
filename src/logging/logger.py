from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from src.models.state import StudyStateManager


class StudyLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, manager: StudyStateManager) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_path = self.log_dir / f"{manager.state.decision_id}_{timestamp}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(manager.as_dict(), f, ensure_ascii=False, indent=2)

        return file_path