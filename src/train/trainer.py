from pathlib import Path

import torch
import torch.nn.functional as F
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
    ):
        self.builder = builder
        # TODO: поменять препроцессоры
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

    def train(self, loader):
        self.encoder.train()
        self.criterion.train()

        total_loss = 0

        pbar = tqdm(loader)
        for waveforms, labels in pbar:
            waveforms = waveforms.to(
                self.device,
                non_blocking=True,
            )
            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            embeddings = self.encoder(waveforms)

            loss, _ = self.criterion(embeddings, labels)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    def validate(self, dataset):
        self.encoder.eval()

        pairs = self.builder.build(dataset)

        embedding_cache = {}
        unique_samples = {}

        for pair in tqdm(pairs):
            sample1 = pair["sample1"]
            sample2 = pair["sample2"]

            unique_samples[get_sample_key(sample1)] = sample1
            unique_samples[get_sample_key(sample2)] = sample2

        with torch.no_grad():
            for key, sample in tqdm(unique_samples.items(), desc="Extract embeddings"):
                waveform = self.val_preprocessor.load_audio(get_audio_input(sample))

                embedding_cache[key] = self.encoder.extract(waveform)

        # Calculate cosine similarity
        scores = []
        labels = []

        for pair in tqdm(pairs, desc="Cosine similarity"):
            sample1 = pair["sample1"]
            sample2 = pair["sample2"]

            emb1 = embedding_cache[get_sample_key(sample1)]
            emb2 = embedding_cache[get_sample_key(sample2)]

            score = F.cosine_similarity(emb1, emb2, dim=0)

            scores.append(score.item())
            labels.append(pair["label"])

        scores = torch.tensor(scores)
        labels = torch.tensor(labels)

        threshold = (
            self.metrics.find_best_threshold(scores, labels)
            if self.threshold is None
            else self.threshold
        )

        result = self.metrics.evaluate(labels, scores, threshold)

        return {"metrics": result, "pairs": pairs, "labels": labels, "scores": scores}

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def save_checkpoint(self, epoch, train_loss, metrics):
        self.weights.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            # Training state
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            # Model
            "encoder": self.encoder.state_dict(),
            "criterion": self.criterion.state_dict(),
            # Optimization
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            # Metrics
            "eer": float(metrics["metrics"]["eer"]),
            "best_eer": float(self.best_eer),
            "roc_auc": float(metrics["metrics"]["roc_auc"]),
            "accuracy": float(metrics["metrics"]["accuracy"]),
            "threshold": float(metrics["metrics"]["threshold"]),
            # Learning rate
            "lr": float(self.get_lr()),
        }

        torch.save(checkpoint, self.weights / "last.pt")

        if checkpoint["eer"] < self.best_eer:
            self.best_eer = checkpoint["eer"]

            torch.save(checkpoint, self.weights / "best.pt")

    def load_checkpoint(self, path):
        """
        Полностью восстанавливает состояние обучения.

        Возвращает epoch, с которого необходимо продолжить обучение.
        """
        print("=" * 60)
        print(f"Loading checkpoint: {path}")
        print("=" * 60)

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        # Model
        self.encoder.load_state_dict(checkpoint["encoder"])
        # Criterion
        if checkpoint.get("criterion") is not None:
            self.criterion.load_state_dict(checkpoint["criterion"])
        # Optimizer
        if checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        # Scheduler
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        # Best metric
        self.best_eer = checkpoint.get(
            "eer",
            float("inf"),
        )
        epoch = checkpoint.get(
            "epoch",
            0,
        )
        train_loss = checkpoint.get(
            "train_loss",
            None,
        )
        eer = checkpoint.get(
            "eer",
            None,
        )
        roc_auc = checkpoint.get(
            "roc_auc",
            None,
        )
        accuracy = checkpoint.get(
            "accuracy",
            None,
        )
        threshold = checkpoint.get(
            "threshold",
            None,
        )
        lr = checkpoint.get(
            "lr",
            None,
        )

        # Information
        print(f"Epoch      : {epoch}")
        if train_loss is not None:
            print(f"Train loss : {train_loss:.4f}")
        if eer is not None:
            print(f"EER        : {eer:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC    : {roc_auc:.4f}")
        if accuracy is not None:
            print(f"Accuracy   : {accuracy:.4f}")
        if threshold is not None:
            print(f"Threshold  : {threshold:.4f}")
        if lr is not None:
            print(f"LR         : {lr:.2e}")
        else:
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"LR         : {current_lr:.2e} (current)")
        print("=" * 60)

        return epoch + 1
