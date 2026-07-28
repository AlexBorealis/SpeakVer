import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.config import REPORTS_DIR
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
def parse_args():
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
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--microphone",
        default="throat_microphone",
        choices=[
            "headset_microphone",
            "forehead_accelerometer",
            "soft_in_ear_microphone",
            "rigid_in_ear_microphone",
            "temple_vibration_pickup",
            "throat_microphone",
        ],
        help="Microphone type fo training.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed cosine similarity threshold",
    )

    parser.add_argument(
        "--balance",
        action="store_true",
        help="Enable balancing of validation pairs",
    )

    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable progress bar",
    )

    return parser.parse_args()


# ============================================================
# Report directory
# ============================================================
def prepare_output_dir(output_dir: str | None) -> Path:
    report_dir = Path(f"report_{datetime.now():%Y%m%d_%H%M%S}")

    if output_dir:
        report_dir = Path(f"{report_dir}_{output_dir}")

    report_dir = (REPORTS_DIR / report_dir).resolve()

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return report_dir


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    report_dir = prepare_output_dir(args.output_dir)

    print("=" * 60)
    print(
        "Model :",
        args.model_path if args.model_path else "SpeechBrain pretrained",
    )
    print("Dataset:", args.dataset_path)
    print("Device :", args.device)
    print("Report :", report_dir)
    print("=" * 60)
    print()

    # Dataset
    test_dataset = SpeakerDataset(
        args.dataset_path,
        return_audio=False,
        microphone=args.microphone
    )
    dataset_stats = test_dataset.statistics()

    # Preprocessing
    train_preprocessor = AudioPreprocessor(augment=True)
    val_preprocessor = AudioPreprocessor()

    # Utils
    builder = PairBuilder(
        balance=args.balance,
        disable=args.disable,
    )
    metrics = Metrics()
    plotter = Plotter()

    # Model
    extractor = EmbeddingExtractor(device=args.device)

    if args.model_path:
        checkpoint = Path(args.model_path)

        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        extractor.load_encoder_weights(str(checkpoint))

        print(
            "Loaded checkpoint:",
            checkpoint,
        )
    else:
        print("Using pretrained SpeechBrain encoder")

    # classifier
    classifier = AAMSoftmax(
        num_classes=test_dataset.get_num_speakers(),
    )

    optimizer = torch.optim.AdamW(
        list(extractor.parameters()) + list(classifier.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Trainer
    trainer = Trainer(
        builder=builder,
        train_preprocessor=train_preprocessor,
        val_preprocessor=val_preprocessor,
        metrics=metrics,
        encoder=extractor,
        classifier=classifier,
        optimizer=optimizer,
        threshold=args.threshold,
        disable=args.disable,
    )

    # Validation
    result = trainer.validate(test_dataset)
    metrics_result = result["metrics"]

    print("=" * 60)
    print("Speaker Dataset Statistics")
    print("=" * 60)

    print(f"Count speakers                  : {dataset_stats['num_speakers']:.4f}")
    print(f"Count audio files               : {dataset_stats['num_samples']:.4f}")
    print(f"Min count recordings            : {dataset_stats['min_qty']:.4f}")
    print(f"Max count recordings            : {dataset_stats['max_qty']:.4f}")
    print(f"Mean count recordings           : {dataset_stats['mean_qty']:.4f}")
    print(f"Median count recordings         : {dataset_stats['median_qty']:.4f}")
    print(f"Std count recordings            : {dataset_stats['std_qty']:.4f}")
    print()

    print("=" * 60)
    print(f"Min duration recordings         : {dataset_stats['min_duration']:.4f}")
    print(f"Max duration recordings         : {dataset_stats['max_duration']:.4f}")
    print(f"Mean duration recordings        : {dataset_stats['mean_duration']:.4f}")
    print(f"Median duration recordings      : {dataset_stats['median_duration']:.4f}")
    print(f"Std duration recordings         : {dataset_stats['std_duration']:.4f}")
    print("=" * 60)
    print()

    print("Top-10 speakers")
    print("=" * 60)
    for speaker, n in dataset_stats["speaker_distribution"].most_common(10):
        print(f"{speaker:15s}                 : {n}")
    print("=" * 60)
    print()

    print("Speaker Verification Metrics")
    print("=" * 60)
    print(f"Accuracy                        : {metrics_result['accuracy']:.4f}")
    print(f"Precision                       : {metrics_result['precision']:.4f}")
    print(f"Recall                          : {metrics_result['recall']:.4f}")
    print(f"F1-score                        : {metrics_result['f1']:.4f}")
    print(f"ROC AUC                         : {metrics_result['roc_auc']:.4f}")
    print(f"EER                             : {metrics_result['eer']:.4f}")
    print(f"MinDCF                          : {metrics_result['min_dcf']:.4f}")
    print(f"Threshold                       : {metrics_result['threshold']:.4f}")
    print("=" * 60)
    print()

    # Save report
    report = BaselineReport(report_dir)

    report.save(
        metrics=metrics_result,
        pairs=result["pairs"],
        scores=result["scores"],
        labels=result["labels"],
        dataset_stats=dataset_stats,
        plotter=plotter,
    )

    print("Baseline report successfully saved")
    print("Directory:", report_dir)


if __name__ == "__main__":
    main()
