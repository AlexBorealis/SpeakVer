import csv
import itertools
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import torch.optim.optimizer as opt
from tqdm import tqdm

from src.data.audio_preprocessor import AudioPreprocessor
from src.data.metrics import Metrics
from src.data.pair_builder import PairBuilder
from src.model.aamsoftmax import AAMSoftmax
from src.model.embedding_extractor import EmbeddingExtractor
from src.train.speaker_dataset import SpeakerDataset


class Trainer:
    def __init__(
        self,
        builder: PairBuilder,
        train_preprocessor: AudioPreprocessor,
        val_preprocessor: AudioPreprocessor,
        metrics: Metrics,
        encoder: EmbeddingExtractor,
        classifier: AAMSoftmax,
        optimizer: opt.Optimizer,
        scheduler: lrs.LRScheduler = None,
        threshold: float = None,
        save_dir: str | Path = "runs",
        early_stop_patience: int = 12,
        warmup_epochs: int = 3,
        device: str = "cuda:0",
        disable: bool = True,
    ):
        self.builder = builder
        self.train_preprocessor = train_preprocessor
        self.val_preprocessor = val_preprocessor
        self.metrics = metrics
        self.encoder = encoder
        self.classifier = classifier
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.threshold = threshold

        self.device = device
        self.encoder.to(self.device)
        self.classifier.to(self.device)

        self.save_dir = Path(save_dir)

        self.best_eer = float("inf")
        self.best_score = float("inf")
        self.best_min_dcf = float("inf")
        self.best_auc = 0.0
        self.best_epoch = 0

        self.weights = self.save_dir / "weights"
        self.history_file = self.save_dir / "history.csv"
        self.config_file = self.save_dir / "config.json"
        self.summary_file = self.save_dir / "summary.json"

        self.started = datetime.now()

        self.epochs_without_improvement = 0
        self.early_stop_patience = early_stop_patience

        # Параметры разогрева (Warmup)
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

        # Validation cache
        self.val_pairs = None
        self.val_unique_samples = None
        self.val_dataset_id = None

        # AMP
        self.scaler = torch.amp.GradScaler("cuda:0")
        self.max_grad_norm = 5.0

        self.disable = disable

    # ==========================================================
    # Train
    # ==========================================================
    def train(self, loader):
        self.encoder.train()
        self.classifier.train()

        total_loss = 0.0

        iterator = tqdm(loader, disable=self.disable)

        for batch in iterator:
            waveforms = batch["waveforms"].to(
                self.device,
                non_blocking=True,
            )

            lengths = batch["lengths"].to(
                self.device,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda:0"):
                embeddings = self.encoder(
                    waveforms,
                    lengths,
                )

                loss, _ = self.classifier(
                    embeddings,
                    labels,
                )

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                itertools.chain(
                    self.encoder.parameters(),
                    self.classifier.parameters(),
                ),
                max_norm=self.max_grad_norm,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
            )

        return total_loss / len(loader)

    # ==========================================================
    # Prepare validation cache
    # ==========================================================
    def _prepare_validation_cache(self, dataset: SpeakerDataset):
        dataset_id = id(dataset)

        # уже подготовлено
        if self.val_pairs is not None and self.val_dataset_id == dataset_id:
            return

        print("Preparing validation pairs cache...")

        # создание или загрузка через PairBuilder
        self.val_pairs = self.builder.build(dataset)

        self.val_unique_samples = {}

        for pair in tqdm(
            self.val_pairs,
            desc="Collect validation samples",
            disable=self.disable,
        ):
            for sample in (pair["sample1"], pair["sample2"]):
                key = (
                    sample["speaker_id"],
                    sample["recording"],
                )

                if not dataset.shuffle and dataset.microphone is not None:
                    domain = next(iter(sample["domains"].values()))

                    sample = {
                        **sample,
                        "audio_path": domain[dataset.microphone],
                    }

                self.val_unique_samples[key] = sample

        self.val_dataset_id = dataset_id

        print(
            f"Validation cache ready: "
            f"{len(self.val_pairs)} pairs, "
            f"{len(self.val_unique_samples)} samples"
        )

    # ==========================================================
    # Validation
    # ==========================================================
    @torch.inference_mode()
    def validate(self, dataset):
        self.encoder.eval()

        # Prepare cache
        self._prepare_validation_cache(dataset)
        pairs = self.val_pairs

        # Extract embeddings
        embeddings = []
        sample_to_idx = {}

        for idx, (key, sample) in enumerate(
            tqdm(
                self.val_unique_samples.items(),
                desc="Extract embeddings",
                disable=self.disable,
            )
        ):
            if not dataset.shuffle and dataset.microphone is not None:
                audio_path = sample["audio_path"]
            else:
                config = random.choice(list(sample["domains"]))
                microphones = sample["domains"][config]
                microphone = random.choice(list(microphones))
                audio_path = microphones[microphone]

            waveform, _ = self.val_preprocessor(audio_path)

            emb = self.encoder.extract(waveform)
            emb = emb.squeeze(0).cpu()
            embeddings.append(emb)

            sample_to_idx[key] = idx

        # [num_samples, embedding_dim]
        embeddings = torch.stack(embeddings)

        # Convert pairs -> indexes
        idx1 = []
        idx2 = []
        labels = []

        for pair in pairs:
            key1 = (
                pair["sample1"]["speaker_id"],
                pair["sample1"]["recording"],
            )

            key2 = (
                pair["sample2"]["speaker_id"],
                pair["sample2"]["recording"],
            )

            idx1.append(sample_to_idx[key1])
            idx2.append(sample_to_idx[key2])
            labels.append(pair["label"])

        idx1 = torch.tensor(idx1)
        idx2 = torch.tensor(idx2)

        labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

        emb1 = embeddings[idx1]
        emb2 = embeddings[idx2]
        scores = F.cosine_similarity(
            emb1,
            emb2,
            dim=1,
        )

        # для metrics CPU
        threshold = (
            self.metrics.find_best_threshold(
                scores.numpy(),
                labels.numpy(),
            )
            if self.threshold is None
            else self.threshold
        )

        result = self.metrics.evaluate(
            scores.numpy(),
            labels.numpy(),
            threshold,
        )

        return {
            "metrics": result,
            "pairs": pairs,
            "labels": labels,
            "scores": scores,
        }

    # ==========================================================
    # Scheduler
    # ==========================================================
    def step_scheduler(
        self,
        metric: float,
        current_epoch: int,
    ):
        """
        Warmup:
            линейно увеличивает LR каждой группы
            с 10% до 100%.

        Далее:
            управление передается scheduler.
        """

        if current_epoch <= self.warmup_epochs:
            alpha = current_epoch / self.warmup_epochs
            scale = 0.1 + 0.9 * alpha

            for base_lr, group in zip(
                self.base_lrs,
                self.optimizer.param_groups,
            ):
                group["lr"] = base_lr * scale

            return

        if isinstance(
            self.scheduler,
            lrs.ReduceLROnPlateau,
        ):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    # ==========================================================
    # Checkpoints
    # ==========================================================
    def save_checkpoint(
        self,
        epoch,
        train_loss,
        metrics,
    ):
        self.weights.mkdir(
            parents=True,
            exist_ok=True,
        )

        m = metrics["metrics"]

        checkpoint = {
            # Training state
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            # Model
            "encoder": self.encoder.state_dict(),
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            # Current validation metrics
            "eer": float(m["eer"]),
            "min_dcf": float(m["min_dcf"]),
            "roc_auc": float(m["roc_auc"]),
            "accuracy": float(m["accuracy"]),
            "threshold": float(m["threshold"]),
            "combined_score": float(m.get("combined_score", 0.0)),
            # Best model information
            "best_epoch": int(self.best_epoch),
            "best_score": float(self.best_score),
            "best_eer": float(self.best_eer),
            "best_min_dcf": float(self.best_min_dcf),
            "best_auc": float(self.best_auc),
            # Optimizer
            "lr": float(self.get_lr()),
        }

        torch.save(
            checkpoint,
            self.weights / "last.pt",
        )

        return checkpoint

    def update_best(self, checkpoint, current_epoch):
        """
        Лексикографическое сравнение моделей.
        Приоритет:
            1. Минимальный EER
            2. Минимальный minDCF
            3. Максимальный ROC-AUC
        """

        EER_EPS = 1e-3
        DCF_EPS = 1e-3
        AUC_EPS = 1e-4

        eer = checkpoint["eer"]
        min_dcf = checkpoint["min_dcf"]
        roc_auc = checkpoint["roc_auc"]

        improved = False

        # 1. Основной критерий — EER
        if eer < self.best_eer - EER_EPS:
            improved = True

        # 2. Если EER практически одинаковый
        elif abs(eer - self.best_eer) <= EER_EPS:
            if min_dcf < self.best_min_dcf - DCF_EPS:
                improved = True
            # 3. Если и minDCF одинаковый
            elif abs(min_dcf - self.best_min_dcf) <= DCF_EPS:
                if roc_auc > self.best_auc + AUC_EPS:
                    improved = True

        # ----------------------------------------------------------
        if improved:
            self.best_eer = eer
            self.best_min_dcf = min_dcf
            self.best_auc = roc_auc
            self.best_epoch = current_epoch
            self.best_score = checkpoint.get(
                "combined_score",
                eer,
            )
            self.epochs_without_improvement = 0

            checkpoint["best_eer"] = self.best_eer
            checkpoint["best_min_dcf"] = self.best_min_dcf
            checkpoint["best_auc"] = self.best_auc
            checkpoint["best_epoch"] = self.best_epoch
            checkpoint["best_score"] = self.best_score

            torch.save(checkpoint, self.weights / "best.pt")

        else:
            if current_epoch > self.warmup_epochs:
                self.epochs_without_improvement += 1
            else:
                self.epochs_without_improvement = 0

        return improved

    def should_stop(self):
        return self.epochs_without_improvement >= self.early_stop_patience

    # ==========================================================
    # Load
    # ==========================================================
    def load_checkpoint(self, path):
        print("=" * 60)
        print(f"Loading checkpoint: {path}")
        print("=" * 60)

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.encoder.load_state_dict(checkpoint["encoder"])

        if checkpoint.get("classifier") is not None:
            self.classifier.load_state_dict(checkpoint["classifier"])

        if checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        # Restore best metrics
        self.best_epoch = checkpoint.get(
            "best_epoch",
            checkpoint.get("epoch", 0),
        )

        self.best_score = checkpoint.get(
            "best_score",
            float("inf"),
        )

        self.best_eer = checkpoint.get(
            "best_eer",
            checkpoint.get("eer", float("inf")),
        )

        self.best_min_dcf = checkpoint.get(
            "best_min_dcf",
            checkpoint.get("min_dcf", float("inf")),
        )

        self.best_auc = checkpoint.get(
            "best_auc",
            checkpoint.get("roc_auc", 0.0),
        )

        epoch = checkpoint.get("epoch", 0)

        # Information
        print(f"Epoch            : {epoch}")
        print(f"Best epoch       : {self.best_epoch}")
        print()

        print("Current metrics")
        print(f"  Train loss     : {checkpoint.get('train_loss', 0):.4f}")
        print(f"  EER            : {checkpoint.get('eer', 0):.4f}")
        print(f"  minDCF         : {checkpoint.get('min_dcf', 0):.4f}")
        print(f"  ROC-AUC        : {checkpoint.get('roc_auc', 0):.4f}")
        print(f"  Accuracy       : {checkpoint.get('accuracy', 0):.4f}")
        print(f"  Threshold      : {checkpoint.get('threshold', 0):.4f}")
        print(f"  Combined score : {checkpoint.get('combined_score', 0):.4f}")
        print()

        print("Best metrics")
        print(f"  Best EER       : {self.best_eer:.4f}")
        print(f"  Best minDCF    : {self.best_min_dcf:.4f}")
        print(f"  Best ROC-AUC   : {self.best_auc:.4f}")
        print(f"  Best score     : {self.best_score:.4f}")
        print()

        print(f"Learning rate    : {self.get_lr():.2e}")
        print("=" * 60)

        return epoch + 1

    # ==========================================================
    # Utils
    # ==========================================================
    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def save_config(self, config: dict):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        config["started"] = self.started.isoformat()

        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def append_history(
        self,
        epoch,
        train_loss,
        metrics,
        improved,
        epoch_time=None,
    ):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        file_exists = self.history_file.exists()

        with open(
            self.history_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(
                    [
                        "epoch",
                        "best_epoch",
                        "train_loss",
                        "combined_score",
                        "best_score",
                        "eer",
                        "best_eer",
                        "roc_auc",
                        "best_auc",
                        "min_dcf",
                        "best_min_dcf",
                        "accuracy",
                        "threshold",
                        "lr",
                        "improved",
                        "epoch_time",
                    ]
                )

            writer.writerow(
                [
                    epoch,
                    self.best_epoch,
                    train_loss,
                    metrics["combined_score"],
                    self.best_score,
                    metrics["eer"],
                    self.best_eer,
                    metrics["roc_auc"],
                    self.best_auc,
                    metrics["min_dcf"],
                    self.best_min_dcf,
                    metrics["accuracy"],
                    metrics["threshold"],
                    self.get_lr(),
                    int(improved),
                    epoch_time,
                ]
            )

    def save_summary(self, last_epoch):
        summary = {
            "finished": datetime.now().isoformat(),
            "best_epoch": self.best_epoch,
            "best_eer": self.best_eer,
            "best_score": self.best_score,
            "epochs_completed": last_epoch,
            "early_stopping": self.should_stop(),
        }

        with open(
            self.summary_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=4)
