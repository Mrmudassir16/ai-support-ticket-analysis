import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

# Path to the support tickets CSV dataset
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", str(BASE_DIR / "support_tickets.csv"))

# Gemini LLM Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Gemini Model Name (using Gemini 2.5 Flash / 1.5 Flash)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def is_gemini_api_key_configured() -> bool:
    """Check if a valid Gemini API key is configured in the environment."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(key and key != "your_gemini_api_key_here")
