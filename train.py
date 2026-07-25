import argparse
import os
import sys
import traceback
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.audio_preprocessor import AudioPreprocessor
from src.data.metrics import Metrics
from src.data.pair_builder import PairBuilder
from src.model.aamsoftmax import AAMSoftmax
from src.model.embedding_extractor import EmbeddingExtractor
from src.train.speaker_dataset import SpeakerDataset
from src.train.trainer import Trainer
from src.utils.utils import split_dataset


# ============================================================
# Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ECAPA-TDNN for speaker verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            Train model:
                python -m train

            Train on GPU:
                python -m train --device cuda:0

            Resume training:
                python -m train --model_path runs/checkpoints/best.pt

            Train in background:
                nohup bash -c 'echo "PID: $$"; exec python -u -m train \
                 --device cuda:0 --batch_size 64 --trainable_blocks 3 --disable \
                --cache_path cache/val_pairs.pkl \
                --save_dir runs/exp1_blocks3_batch64_epochs50_no_balance_plateausched' \
                > train.log 2>&1 &

            Monitor progress:
                tail -f train.log
        """,
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="speaker_dataset",
        help="Dataset directory.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to checkpoint.",
    )

    parser.add_argument(
        "--cache_filename",
        type=str,
        default=None,
        help="Cache filename for validation parts of dataset.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="runs",
        help="Directory for checkpoints.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of epochs.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device.",
    )

    parser.add_argument(
        "--target_sr",
        type=int,
        default=8000,
        help="Target sampling rate.",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--trainable_blocks",
        type=int,
        default=2,
        help="Number of unfrozen ECAPA blocks.",
    )

    parser.add_argument(
        "--neg_ratio",
        type=int,
        default=None,
        help="In that times negative pairs bigger than positive pairs.",
    )

    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable progress bar.",
    )

    parser.add_argument(
        "--balance",
        action="store_true",
        help="Enable balancing of validation pairs.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    print("=" * 80)
    print(f"Started          : {datetime.now()}")
    print(f"PID              : {os.getpid()}")
    print(f"Dataset          : {args.dataset_path}")
    print(f"Device           : {args.device}")
    print(f"Epochs           : {args.epochs}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Trainable blocks : {args.trainable_blocks}")
    print("=" * 80)

    # Model
    extractor = EmbeddingExtractor(
        trainable_blocks=args.trainable_blocks,
        device=args.device,
    )

    criterion = AAMSoftmax(
        num_classes=0,  # temporary
    )

    # Utils
    builder = PairBuilder(
        balance=args.balance,
        negative_ratio=args.neg_ratio,
        cache_filename=f"{args.dataset_path}/{args.cache_filename}",
        disable=args.disable,
    )

    metrics = Metrics()

    train_preprocessor = AudioPreprocessor(
        augment=True,
        random_crop=True,
        target_sr=args.target_sr,
    )

    val_preprocessor = AudioPreprocessor(
        target_sr=args.target_sr,
    )

    # Dataset split
    split_dataset(
        SpeakerDataset(
            args.dataset_path,
            return_audio=False,
        ),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_key="speaker_id",
        persistent=True,
        output_dir=args.dataset_path,
    )

    # Datasets
    train_dataset = SpeakerDataset(
        f"{args.dataset_path}/train",
        preprocessor=train_preprocessor,
        return_audio=True,
    )

    val_dataset = SpeakerDataset(
        f"{args.dataset_path}/val",
        preprocessor=val_preprocessor,
        return_audio=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=args.device.startswith("cuda"),
    )

    # Training components
    criterion = AAMSoftmax(
        num_classes=train_dataset.get_num_speakers(),
    )

    optimizer = torch.optim.AdamW(
        list(extractor.parameters()) + list(criterion.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        threshold=1e-3,
        threshold_mode="rel",
        cooldown=2,
        min_lr=1e-6,
    )

    # Trainer
    trainer = Trainer(
        builder=builder,
        train_preprocessor=train_preprocessor,
        val_preprocessor=val_preprocessor,
        metrics=metrics,
        encoder=extractor,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=args.save_dir,
        early_stop_patience=12,
        disable=args.disable,
    )

    # Resume
    start_epoch = 0

    if args.model_path is not None:
        start_epoch = trainer.load_checkpoint(args.model_path)

    # Training
    for epoch in tqdm(
        range(start_epoch, args.epochs),
        desc="Training",
        total=args.epochs - start_epoch,
    ):
        train_loss = trainer.train(train_loader)
        val_metrics = trainer.validate(val_dataset)

        eer = val_metrics["metrics"]["eer"]
        roc_auc = val_metrics["metrics"]["roc_auc"]

        checkpoint = trainer.save_checkpoint(
            epoch + 1,
            train_loss,
            val_metrics,
        )

        improved = trainer.update_best(checkpoint)

        trainer.step_scheduler(eer)

        lr = trainer.get_lr()

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"eer={eer:.4f} | "
            f"auc={roc_auc:.4f} | "
            f"lr={lr:.2e}" + (" (*) BEST" if improved else "")
        )

        if trainer.should_stop():
            print()
            print("=" * 80)
            print(
                f"Early stopping after {epoch + 1} epochs\n"
                f"Best EER : {trainer.best_eer:.4f}"
            )
            print("=" * 80)
            break

    print("=" * 80)
    print("SUCCESS")
    print(f"Best EER : {trainer.best_eer:.4f}")
    print(f"Finished : {datetime.now()}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)

    except Exception:
        print("=" * 80)
        print("ERROR")
        print(traceback.format_exc())
        print("=" * 80)
        sys.exit(1)
