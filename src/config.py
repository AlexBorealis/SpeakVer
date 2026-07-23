import os
from pathlib import Path

from dotenv import load_dotenv

# ============================================================
# Project paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_DIR = BASE_DIR / "config"
ENV_DIR = CONFIG_DIR / "envs"
ENV_PATH = ENV_DIR / ".env"

load_dotenv(ENV_PATH)

# ============================================================
# Environment
# ============================================================
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.15"))

DEFAULT_DEVICE = os.getenv(
    "DEFAULT_DEVICE",
    "cpu",
)

DEFAULT_REPORT_DATASET = os.getenv(
    "DEFAULT_REPORT_DATASET",
    "datasets/test",
)

HF_TOKEN = os.getenv("HF_TOKEN")

ARCHIVES_DIR = BASE_DIR / "archives"
REPORTS_DIR = BASE_DIR / "reports"

RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))

SERVER_NAME = os.getenv(
    "SERVER_NAME",
    "0.0.0.0",
)

SERVER_PORT = int(
    os.getenv(
        "SERVER_PORT",
        "7860",
    )
)
