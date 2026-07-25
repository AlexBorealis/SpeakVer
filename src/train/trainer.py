from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
from tqdm import tqdm

from src.utils.utils import get_audio_input, get_sample_key


class Trainer:
    def __init__(
        self,
        builder,
        train_preprocessor,
        val_preprocessor,
        metrics,
        encoder,
        criterion,
        optimizer,
        scheduler=None,
        threshold=None,
        save_dir="runs",
        early_stop_patience=12,
        disable=True,
    ):
        self.builder = builder

        self.train_preprocessor = train_preprocessor
        self.val_preprocessor = val_preprocessor

        self.metrics = metrics

        self.encoder = encoder
        self.criterion = criterion

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.threshold = threshold

        self.device = next(self.encoder.parameters()).device

        self.encoder.to(self.device)
        self.criterion.to(self.device)

        self.save_dir = Path(save_dir)
        self.weights = self.save_dir / "weights"

        self.best_eer = float("inf")
        self.epochs_without_improvement = 0
        self.early_stop_patience = early_stop_patience

        self.disable = disable

        # Validation cache
        self.val_pairs = None

        # уникальные аудиозаписи для extraction
        self.val_unique_samples = None

        # чтобы понимать, тот ли dataset используется
        self.val_dataset_id = None

    # ==========================================================
    # Train
    # ==========================================================
    def train(self, loader):
        self.encoder.train()
        self.criterion.train()

        total_loss = 0.0

        iterator = tqdm(loader, disable=self.disable)

        for waveforms, labels in iterator:
            waveforms = waveforms.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            embeddings = self.encoder(waveforms)

            loss, _ = self.criterion(
                embeddings,
                labels,
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            iterator.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    # ==========================================================
    # Prepare validation cache
    # ==========================================================
    def _prepare_validation_cache(self, dataset):
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
            self.val_unique_samples[get_sample_key(pair["sample1"])] = pair["sample1"]
            self.val_unique_samples[get_sample_key(pair["sample2"])] = pair["sample2"]

        self.val_dataset_id = dataset_id

        print(
            f"Validation cache ready: "
            f"{len(self.val_pairs)} pairs, "
            f"{len(self.val_unique_samples)} samples"
        )

    # ==========================================================
    # Validation
    # ==========================================================
    def validate(self, dataset):
        self.encoder.eval()

        # Prepare cache
        self._prepare_validation_cache(dataset)
        pairs = self.val_pairs

        # Extract embeddings
        embeddings = []
        sample_to_idx = {}

        with torch.no_grad():
            for idx, (key, sample) in enumerate(
                tqdm(
                    self.val_unique_samples.items(),
                    desc="Extract embeddings",
                    disable=self.disable,
                )
            ):
                waveform = self.val_preprocessor.load_audio(get_audio_input(sample))
                emb = self.encoder.extract(waveform)

                # гарантируем shape [192]
                emb = emb.squeeze(0)
                embeddings.append(emb)
                sample_to_idx[key] = idx

        # [num_samples, embedding_dim]
        embeddings = torch.stack(embeddings).to(self.device)

        # Convert pairs -> indexes
        idx1 = []
        idx2 = []
        labels = []

        for pair in pairs:
            idx1.append(sample_to_idx[get_sample_key(pair["sample1"])])
            idx2.append(sample_to_idx[get_sample_key(pair["sample2"])])
            labels.append(pair["label"])

        idx1 = torch.tensor(
            idx1,
            device=self.device,
        )
        idx2 = torch.tensor(
            idx2,
            device=self.device,
        )
        labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

        # Fast cosine similarity
        with torch.no_grad():
            emb1 = embeddings[idx1]
            emb2 = embeddings[idx2]
            scores = F.cosine_similarity(
                emb1,
                emb2,
                dim=1,
            )

        # для metrics CPU
        scores_cpu = scores.cpu()
        labels_cpu = labels.cpu()

        threshold = (
            self.metrics.find_best_threshold(
                scores_cpu.numpy(),
                labels_cpu.numpy(),
            )
            if self.threshold is None
            else self.threshold
        )

        result = self.metrics.evaluate(
            labels_cpu.numpy(),
            scores_cpu.numpy(),
            threshold,
        )

        return {
            "metrics": result,
            "pairs": pairs,
            "labels": labels_cpu,
            "scores": scores_cpu,
        }

    # ==========================================================
    # Scheduler
    # ==========================================================
    def step_scheduler(self, metric=None):
        if self.scheduler is None:
            return

        if isinstance(
            self.scheduler,
            lrs.ReduceLROnPlateau,
        ):
            if metric is None:
                raise ValueError("ReduceLROnPlateau requires validation metric.")

            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    # ==========================================================
    # Utils
    # ==========================================================
    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

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

        checkpoint = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "encoder": self.encoder.state_dict(),
            "criterion": self.criterion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "eer": float(metrics["metrics"]["eer"]),
            "roc_auc": float(metrics["metrics"]["roc_auc"]),
            "accuracy": float(metrics["metrics"]["accuracy"]),
            "threshold": float(metrics["metrics"]["threshold"]),
            "best_eer": float(self.best_eer),
            "lr": float(self.get_lr()),
        }

        torch.save(
            checkpoint,
            self.weights / "last.pt",
        )

        return checkpoint

    def update_best(self, checkpoint):
        improved = checkpoint["eer"] < self.best_eer

        if improved:
            self.best_eer = checkpoint["eer"]
            self.epochs_without_improvement = 0

            checkpoint["best_eer"] = self.best_eer

            torch.save(
                checkpoint,
                self.weights / "best.pt",
            )

        else:
            self.epochs_without_improvement += 1

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

        if checkpoint.get("criterion") is not None:
            self.criterion.load_state_dict(checkpoint["criterion"])

        if checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.best_eer = checkpoint.get(
            "best_eer",
            checkpoint.get(
                "eer",
                float("inf"),
            ),
        )

        epoch = checkpoint.get("epoch", 0)

        print(f"Epoch      : {epoch}")
        print(f"Train loss : {checkpoint.get('train_loss', 0):.4f}")
        print(f"EER        : {checkpoint.get('eer', 0):.4f}")
        print(f"ROC-AUC    : {checkpoint.get('roc_auc', 0):.4f}")
        print(f"Accuracy   : {checkpoint.get('accuracy', 0):.4f}")
        print(f"Threshold  : {checkpoint.get('threshold', 0):.4f}")
        print(f"Best EER   : {self.best_eer:.4f}")
        print(f"LR         : {self.get_lr():.2e}")
        print("=" * 60)

        return epoch + 1
