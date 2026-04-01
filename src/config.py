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