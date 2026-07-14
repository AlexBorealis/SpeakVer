import argparse
import os

import torch

from src.data.audio_preprocessor import AudioPreprocessor
from src.data.baseline_report import BaselineReport
from src.data.metrics import Metrics
from src.data.pair_builder import PairBuilder
from src.data.plotter import Plotter
from src.model.aamsoftmax import AAMSoftmax
from src.model.embedding_extractor import EmbeddingExtractor
from src.train.speaker_dataset import SpeakerDataset
from src.train.trainer import Trainer
from src.utils.utils import load_checkpoint

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description="Generate speaker verification report")

parser.add_argument(
    "--dataset_path",
    type=str,
    default="speaker_dataset/test",
    help="Path to test dataset",
)

parser.add_argument(
    "--experiment",
    type=str,
    default=None,
    required=False,
    help="Training experiment name",
)

parser.add_argument(
    "--output_dir", type=str, default=None, help="Directory for saving report"
)

args = parser.parse_args()

# ============================================================
# Output directory
# ============================================================
if args.output_dir is None or args.output_dir.strip() == "":
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    REPORT_DIR = os.path.join("reports", f"report_{timestamp}")
else:
    REPORT_DIR = args.output_dir

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("Experiment:", args.experiment)
print("Report directory:", REPORT_DIR)
print("=" * 60)

# ============================================================
# Paths
# ============================================================
checkpoint = os.path.join(
    "runs", "speaker_train", args.experiment, "weights", "best.pt"
)
print("Checkpoint:", checkpoint)

# ============================================================
# Dataset
# ============================================================
test_dataset = SpeakerDataset(args.dataset_path, return_audio=False)

# ============================================================
# Preprocessing
# ============================================================
train_preprocessor = AudioPreprocessor(device="cpu", augment=True, target_sr=8000)
val_preprocessor = AudioPreprocessor(device="cpu")

# ============================================================
# Utils
# ============================================================
builder = PairBuilder()
metrics = Metrics()
plotter = Plotter()

# ============================================================
# Model
# ============================================================
extractor = EmbeddingExtractor()
# Default model
if args.experiment is None or args.experiment.strip() == "":
    print("Using default model")
# Load trained checkpoint
else:
    checkpoint = os.path.join(
        "runs", "speaker_train", args.experiment, "weights", "best.pt"
    )
    print("Loading checkpoint:", checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    extractor = load_checkpoint(checkpoint, extractor)

# ============================================================
# Criterion
# ============================================================
criterion = AAMSoftmax(embedding_dim=192, num_classes=test_dataset.get_num_speakers())
optimizer = torch.optim.AdamW(
    list(extractor.parameters()) + list(criterion.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)

# ============================================================
# Report
# ============================================================
report = BaselineReport(REPORT_DIR)
trainer = Trainer(
    train_preprocessor=train_preprocessor,
    val_preprocessor=val_preprocessor,
    metrics=metrics,
    builder=builder,
    encoder=extractor,
    criterion=criterion,
    optimizer=optimizer,
)

# ============================================================
# Validation
# ============================================================
result = trainer.validate(test_dataset)

print("\nOptimal threshold:")
print(result["metrics"]["threshold"])

print("\nMetrics:")
print(result["metrics"])

# ============================================================
# Save report
# ============================================================
report.save(
    metrics=result["metrics"],
    pairs=result["pairs"],
    scores=result["scores"],
    labels=result["labels"],
    plotter=plotter,
)

print("\nBaseline report successfully saved")
print("Directory:", REPORT_DIR)
