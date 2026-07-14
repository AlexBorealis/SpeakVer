from pathlib import Path
from sklearn.metrics import roc_curve
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
        save_dir="runs/speaker_train/exp",
    ):
        self.builder = builder
        self.train_preprocessor = train_preprocessor
        self.val_preprocessor = val_preprocessor
        self.metrics = metrics
        self.encoder = encoder
        self.criterion = criterion
        self.optimizer = optimizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder.to(self.device)
        self.criterion.to(self.device)

        self.save_dir = Path(save_dir)
        self.weights = self.save_dir / "weights"

        self.weights.mkdir(parents=True, exist_ok=True)

        self.best_eer = float("inf")

    def train(self, loader):
        self.encoder.train()
        self.criterion.train()

        total_loss = 0

        pbar = tqdm(loader)
        for waveforms, labels in pbar:
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

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

        threshold = self.metrics.find_best_threshold(scores, labels)

        result = self.metrics.evaluate(labels, scores, threshold)

        return {"metrics": result, "pairs": pairs, "labels": labels, "scores": scores}

    def save_checkpoint(self, epoch, train_loss, metrics):
        checkpoint = {
            "epoch": epoch,
            "encoder": self.encoder.state_dict(),
            "criterion": self.criterion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "eer": metrics["metrics"]["eer"],
            "roc_auc": metrics["metrics"]["roc_auc"],
            "accuracy": metrics["metrics"]["accuracy"],
            "threshold": metrics["metrics"]["threshold"],
        }

        torch.save(checkpoint, self.weights / "last.pt")

        if metrics["metrics"]["eer"] < self.best_eer:
            self.best_eer = metrics["metrics"]["eer"]

            torch.save(checkpoint, self.weights / "best.pt")
