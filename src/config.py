import os
from pathlib import Path

from dotenv import load_dotenv

# ============================================================
# Project paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]

ENV_DIR = BASE_DIR / "config" / "envs"

load_dotenv(ENV_DIR / ".env")

# ============================================================
# Environment
# ============================================================
HF_TOKEN = os.getenv("HF_TOKEN")

DATASET_DIR = BASE_DIR / "datasets"
ARCHIVES_DIR = BASE_DIR / "archives"
REPORTS_DIR = BASE_DIR / "reports"
RUNS_DIR = BASE_DIR / "runs"
DEFAULT_REPORT_DATASET = "datasets/test" 

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.15"))

DEFAULT_DEVICE = os.getenv(
    "DEFAULT_DEVICE",
    "cpu",
)

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

TARGET_SAMPLE_RATE = int(
    os.getenv(
        "TARGET_SAMPLE_RATE",
        "8000",
    )
)

# Train hyperparameters
LR = float(os.getenv("LR", 1e-4))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
ENCODER_LR = float(os.getenv("ENCODER_LR", 1e-4))
CLASSIFIER_LR = float(os.getenv("CLASSIFIER_LR", 1e-4))

AAM_MARGIN = float(os.getenv("AAM_MARGIN", 0.3))
AAM_SCALE = float(os.getenv("AAM_SCALE", 30))

SCHED_MODE = os.getenv("SCHED_MODE", "min")
SCHED_FACTOR = float(os.getenv("SCHED_FACTOR", 0.5))
SCHED_PATIENCE = int(os.getenv("SCHED_PATIENCE", 10))
SCHED_THRESHOLD = float(os.getenv("SCHED_THRESHOLD", 1e-4))
SCHED_THRESHOLD_MODE = os.getenv("SCHED_THRESHOLD_MODE", "rel")
SCHED_COOLDOWN = int(os.getenv("SCHED_COOLDOWN", 2))
SCHED_MIN_LR = float(os.getenv("SCHED_MIN_LR", 1e-6))
SCHED_MULT = int(os.getenv("SCHED_MULT", 2))

WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", 3))
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", 20))