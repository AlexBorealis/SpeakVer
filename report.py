import argparse
import os
from datetime import datetime

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

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description="Generate speaker verification report")

parser.add_argument(
    "--dataset_path",
    type=str,
    default="datasets/test",
    help="Path to test dataset",
)

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Path to checkpoint",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for saving report",
)

parser.add_argument(
    "--no-balance",
    action="store_true",
)

parser.add_argument(
    "--threshold",
    type=float,
    default=None,
    help="Fixed cosine similarity threshold",
)

parser.add_argument(
    "--device",
    type=str,
    default="cpu",
)

args = parser.parse_args()

# ============================================================
# Report directory
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

DEFAULT_DIR = os.path.join(
    "reports",
    f"report_{timestamp}",
)

REPORT_DIR = DEFAULT_DIR

if args.output_dir is not None:
    REPORT_DIR = f"{DEFAULT_DIR}_{args.output_dir}"

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("Model :", args.model_path if args.model_path else "SpeechBrain pretrained")
print("Dataset:", args.dataset_path)
print("Device :", args.device)
print("Report :", REPORT_DIR)
print("=" * 60)

# ============================================================
# Dataset
# ============================================================
test_dataset = SpeakerDataset(
    args.dataset_path,
    return_audio=False,
)

# ============================================================
# Preprocessing
# ============================================================
train_preprocessor = AudioPreprocessor(
    augment=True,
    target_sr=8000,
)

val_preprocessor = AudioPreprocessor(
    target_sr=8000,
)

# ============================================================
# Utils
# ============================================================
builder = PairBuilder(
    balance=not args.no_balance,
)
metrics = Metrics()
plotter = Plotter()

# ============================================================
# Model
# ============================================================
extractor = EmbeddingExtractor(
    device=args.device,
)

if args.model_path is not None:
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    extractor.load_encoder_weights(args.model_path)

else:
    print("Using pretrained SpeechBrain encoder")

# ============================================================
# Criterion
# ============================================================
criterion = AAMSoftmax(
    num_classes=test_dataset.get_num_speakers(),
)

optimizer = torch.optim.AdamW(
    list(extractor.parameters()) + list(criterion.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)

# ============================================================
# Trainer
# ============================================================
trainer = Trainer(
    builder=builder,
    train_preprocessor=train_preprocessor,
    val_preprocessor=val_preprocessor,
    metrics=metrics,
    encoder=extractor,
    criterion=criterion,
    optimizer=optimizer,
    threshold=args.threshold,
)

# ============================================================
# Report
# ============================================================
report = BaselineReport(REPORT_DIR)

# ============================================================
# Validation
# ============================================================
result = trainer.validate(test_dataset)

metrics = result["metrics"]

print()
print("=" * 60)
print("Speaker Verification Metrics")
print("=" * 60)

print(f"{'Accuracy':20}: {metrics['accuracy']:.4f}")
print(f"{'Precision':20}: {metrics['precision']:.4f}")
print(f"{'Recall':20}: {metrics['recall']:.4f}")
print(f"{'F1-score':20}: {metrics['f1']:.4f}")
print(f"{'ROC AUC':20}: {metrics['roc_auc']:.4f}")
print(f"{'EER':20}: {metrics['eer']:.4f}")
print(f"{'Threshold':20}: {metrics['threshold']:.4f}")

print()
print("Confusion Matrix")
print("-" * 60)

cm = metrics["confusion_matrix"]

print(f"{'':18}Predicted")
print(f"{'':18}Different      Same")
print(f"{'Actual Different':18}{cm[0][0]:10d}{cm[0][1]:10d}")
print(f"{'Actual Same':18}{cm[1][0]:10d}{cm[1][1]:10d}")

print()
print("Normalized Confusion Matrix")
print("-" * 60)

cm = metrics["confusion_matrix_normalized"]

print(f"{'':18}Predicted")
print(f"{'':18}Different      Same")
print(f"{'Actual Different':18}{cm[0][0]:10.4f}{cm[0][1]:10.4f}")
print(f"{'Actual Same':18}{cm[1][0]:10.4f}{cm[1][1]:10.4f}")

print("=" * 60)

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

print()
print("Baseline report successfully saved")
print("Directory:", REPORT_DIR)
