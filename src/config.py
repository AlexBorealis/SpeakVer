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

# Gradio parameters
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

# Train hyperparameters
LR = os.getenv("LR", 1e-4)
WEIGHT_DECAY = os.getenv("WEIGHT_DECAY", 1e-4)
ENCODER_LR = os.getenv("ENCODER_LR", 1e-4)
CLASSIFIER_LR = os.getenv("CLASSIFIER_LR", 1e-4)

AAM_MARGIN = os.getenv("AAM_MARGIN", 0.3)
AAM_SCALE = os.getenv("AAM_SCALE", 30)

SCHED_MODE = os.getenv("SCHED_MODE", "min")
SCHED_FACTOR = os.getenv("SCHED_FACTOR", 0.5)
SCHED_PATIENCE = os.getenv("SCHED_PATIENCE", 10)
SCHED_THRESHOLD = os.getenv("SCHED_THRESHOLD", 1e-4)
SCHED_THRESHOLD_MODE = os.getenv("SCHED_THRESHOLD_MODE", "rel")
SCHED_COOLDOWN = os.getenv("SCHED_COOLDOWN", 2)
SCHED_MIN_LR = os.getenv("SCHED_MIN_LR", 1e-6)
SCHED_MULT = os.getenv("SCHED_MULT", 2)

WARMUP_EPOCHS = os.getenv("WARMUP_EPOCHS", 3)
EARLY_STOP_PATIENCE = os.getenv("EARLY_STOP_PATIENCE", 20)