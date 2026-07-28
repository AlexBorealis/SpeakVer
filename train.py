import argparse
import math
import os
import pickle
import sys
import traceback
from datetime import datetime
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    AAM_MARGIN,
    AAM_SCALE,
    CLASSIFIER_LR,
    EARLY_STOP_PATIENCE,
    ENCODER_LR,
    SCHED_MIN_LR,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from src.data.audio_preprocessor import AudioPreprocessor
from src.data.metrics import Metrics
from src.data.pair_builder import PairBuilder
from src.model.aamsoftmax import AAMSoftmax
from src.model.embedding_extractor import EmbeddingExtractor
from src.train.dynamic_batch_sampler import DynamicBatchSampler
from src.train.speaker_dataset import SpeakerDataset
from src.train.trainer import Trainer
from src.utils.utils import collate_fn, split_dataset


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
                --device cuda:0 --batch_size 64 --trainable_blocks 3 \
                --cache_filename val_pairs.pkl --disable  \
                --dataset_path speaker_dataset \
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
        "--neg_ratio",
        type=int,
        default=None,
        help="In that times negative pairs bigger than positive pairs.",
    )

    parser.add_argument(
        "--trainable_modules",
        nargs="+",
        default=[
            "classifier",
        ],
        choices=[
            "tdnn",
            "seres2netblock1",
            "seres2netblock2",
            "seres2netblock3",
            "mfa",
            "asp",
            "asp_bn",
            "fc",
            "classifier",
        ],
        help="Modules from model for training.",
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

    unfreeze_schedule = {
        5: [
            "classifier",
        ],
        10: [
            "asp",
            "asp_bn",
            "fc",
            "classifier",
        ],
        30: [
            "mfa",
            "asp",
            "asp_bn",
            "fc",
            "classifier",
        ],
    }
    trainable_modules = unfreeze_schedule[list(unfreeze_schedule.keys())[-1]]

    print("=" * 80)
    print(f"Started          : {datetime.now()}")
    print(f"PID              : {os.getpid()}")
    print(f"Dataset          : {args.dataset_path}")
    print(f"Device           : {args.device}")
    print(f"Epochs           : {args.epochs}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Trainable modules : {trainable_modules}")
    print("=" * 80)

    # Utils
    builder = PairBuilder(
        balance=args.balance,
        negative_ratio=args.neg_ratio,
        cache_filename=f"{args.dataset_path}/{args.cache_filename}",
        disable=args.disable,
    )

    metrics = Metrics()

    train_preprocessor = AudioPreprocessor(
        augment=True, target_sr=args.target_sr, fix_length=False, max_duration=7
    )

    val_preprocessor = AudioPreprocessor(
        target_sr=args.target_sr, fix_length=False, max_duration=7
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
    # Train
    train_dataset = SpeakerDataset(
        f"{args.dataset_path}/train",
        preprocessor=train_preprocessor,
        microphone=args.microphone,
        return_audio=True,
    )

    val_dataset = SpeakerDataset(
        f"{args.dataset_path}/val",
        preprocessor=val_preprocessor,
        microphone=args.microphone,
        return_audio=False,
    )

    if not os.path.exists(f"{args.dataset_path}/lengths_cache.pkl"):
        lengths = [
            item["length"] for item in tqdm(train_dataset, total=len(train_dataset))
        ]

        with open(f"{args.dataset_path}/lengths_cache.pkl", "wb") as file:
            pickle.dump(lengths, file)
    else:
        with open(f"{args.dataset_path}/lengths_cache.pkl", "rb") as file:
            lengths = pickle.load(file)

    batch_sampler = DynamicBatchSampler(
        lengths=lengths,
        max_batch_length=2500000,
        bucket_size=256,
        max_batch_size=args.batch_size,
        min_batch_size=4,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        # batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler,
        pin_memory=True,
    )

    # Validation
    val_dataset = SpeakerDataset(
        f"{args.dataset_path}/val",
        preprocessor=val_preprocessor,
        return_audio=False,
    )

    # Model
    extractor = EmbeddingExtractor(
        device=args.device,
    )

    # Training components
    classifier = AAMSoftmax(
        num_classes=train_dataset.get_num_speakers(),
        margin=float(AAM_MARGIN),
        scale=int(AAM_SCALE),
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": extractor.parameters(),
                "lr": float(ENCODER_LR),
            },
            {
                "params": classifier.parameters(),
                "lr": float(CLASSIFIER_LR),
            },
        ],
        weight_decay=float(WEIGHT_DECAY),
    )

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode=SCHED_MODE,
    #     factor=float(SCHED_FACTOR),
    #     patience=int(SCHED_PATIENCE),
    #     threshold=float(SCHED_THRESHOLD),
    #     threshold_mode=SCHED_THRESHOLD_MODE,
    #     cooldown=int(SCHED_COOLDOWN),
    #     min_lr=float(SCHED_MIN_LR),
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=float(SCHED_MIN_LR),
    )

    # Trainer
    extractor.set_trainable_modules(
        ["classifier"],
        classifier,
    )

    trainer = Trainer(
        builder=builder,
        train_preprocessor=train_preprocessor,
        val_preprocessor=val_preprocessor,
        metrics=metrics,
        encoder=extractor,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=args.save_dir,
        early_stop_patience=int(EARLY_STOP_PATIENCE),
        warmup_epochs=int(WARMUP_EPOCHS),
        disable=args.disable,
    )

    # Save experiment configuration
    trainer.save_config(
        {
            "dataset": args.dataset_path,
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "trainable_modules": trainable_modules,
            "target_sr": args.target_sr,
            # classifier
            "classifier": classifier.__class__.__name__,
            "margin": classifier.margin,
            "scale": classifier.scale,
            # Optimizer
            "optimizer": optimizer.__class__.__name__,
            # "learning_rate": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0]["weight_decay"],
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "classifier_lr": optimizer.param_groups[1]["lr"],
            # Scheduler
            "scheduler": scheduler.__class__.__name__,
            # "scheduler_factor": scheduler.factor,
            # "scheduler_patience": scheduler.patience,
            # "scheduler_threshold": scheduler.threshold,
            # "scheduler_threshold_mode": scheduler.threshold_mode,
            # "scheduler_cooldown": scheduler.cooldown,
            # "scheduler_mode": scheduler.mode,
            # "scheduler_min_lr": scheduler.min_lrs[0],
            "T_0": scheduler.T_0,
            "T_mult": scheduler.T_mult,
            "eta_min": scheduler.eta_min,
            # Trainer
            "warmup_epochs": trainer.warmup_epochs,
            "early_stop_patience": trainer.early_stop_patience,
            # Dataset
            "balance_validation": args.balance,
            "negative_ratio": args.neg_ratio,
            "max_duration": train_preprocessor.max_duration
            if train_preprocessor.fix_length
            else None,
            "max_batch_length": batch_sampler.max_batch_length,
            "max_batch_size": batch_sampler.max_batch_size,
            "min_batch_size": batch_sampler.min_batch_size,
        }
    )

    # Resume
    start_epoch = 1

    if args.model_path is not None:
        start_epoch = trainer.load_checkpoint(args.model_path)

    last_epoch = start_epoch - 1

    # Training
    for epoch in tqdm(
        range(start_epoch, args.epochs + 1),
        desc="Training",
        total=args.epochs + 1 - start_epoch,
    ):
        last_epoch = epoch

        # ------------------------------------------
        # Progressive unfreezing
        # ------------------------------------------
        if epoch in unfreeze_schedule:
            print()
            print(f"Epoch {epoch}: updating trainable blocks")

            extractor.set_trainable_modules(
                unfreeze_schedule[epoch],
                classifier,
            )

        epoch_start = perf_counter()

        train_loss = trainer.train(train_loader)

        val_metrics = trainer.validate(val_dataset)
        metrics = val_metrics["metrics"]

        eer = metrics["eer"]
        roc_auc = metrics["roc_auc"]
        min_dcf = metrics["min_dcf"]

        combined_score = math.sqrt(eer * min_dcf)

        metrics["combined_score"] = combined_score

        checkpoint = trainer.save_checkpoint(
            epoch=epoch,
            train_loss=train_loss,
            metrics=val_metrics,
        )

        improved = trainer.update_best(
            checkpoint=checkpoint,
            current_epoch=epoch,
        )

        trainer.step_scheduler(
            metric=combined_score,
            current_epoch=epoch,
        )

        epoch_time = perf_counter() - epoch_start

        trainer.append_history(
            epoch=epoch,
            train_loss=train_loss,
            metrics=metrics,
            improved=improved,
            epoch_time=epoch_time,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss={train_loss:.4f} | "
            f"EER={eer:.4f} | "
            f"minDCF={min_dcf:.4f} | "
            f"AUC={roc_auc:.4f} | "
            f"Combined score={combined_score:.4f} | "
            f"lr={trainer.get_lr():.2e}" + (" (*) BEST" if improved else "")
        )

        if trainer.should_stop():
            print()
            print("=" * 80)
            print("EARLY STOPPING")
            print("-" * 80)
            print(f"Stopped epoch : {epoch}")
            print(f"Best epoch    : {trainer.best_epoch}")
            print(f"Best EER      : {trainer.best_eer:.4f}")
            print(f"Best minDCF   : {trainer.best_min_dcf:.4f}")
            print(f"Best ROC-AUC  : {trainer.best_auc:.4f}")
            print("=" * 80)
            break

    # Save experiment summary
    trainer.save_summary(last_epoch)

    # Finish
    print("=" * 80)
    print("SUCCESS")
    print(f"Best epoch     : {trainer.best_epoch}")
    print(f"Best EER       : {trainer.best_eer:.4f}")
    print(f"Best minDCF    : {trainer.best_min_dcf:.4f}")
    print(f"Best ROC-AUC   : {trainer.best_auc:.4f}")
    print(f"Finished       : {datetime.now()}")
    print("=" * 80)


# ==========================================================
# Result
# ==========================================================
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
