from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
EVALUATIONS_DIR = DATA_DIR / "evaluations"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


OPENAI_API_KEY = "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"
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4-mini")
DEFAULT_MAX_QUESTIONS = 5
DEFAULT_MAX_ROUNDS = 2


INITIAL_GENERATION_PROMPT_PATH = PROMPTS_DIR / "initial_generation.txt"
QUESTION_GENERATION_PROMPT_PATH = PROMPTS_DIR / "question_generation.txt"
REFINEMENT_PROMPT_PATH = PROMPTS_DIR / "refinement.txt"


for directory in [DATA_DIR, LOGS_DIR, EVALUATIONS_DIR, RAW_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)