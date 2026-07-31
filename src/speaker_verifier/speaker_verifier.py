import torch
import torch.nn.functional as F

from src.data.audio_preprocessor import AudioPreprocessor
from src.model.embedding_extractor import EmbeddingExtractor


class SpeakerVerifier:
    def __init__(
        self,
        checkpoint: str | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.device = device
        self.threshold = threshold

        self.preprocessor = AudioPreprocessor()
        self.model = EmbeddingExtractor()

        if checkpoint is not None:
            print(f"Loading checkpoint: {checkpoint}")
            self.model.load_encoder_weights(
                checkpoint,
            )
        else:
            print("Using default EmbeddingExtractor.")

        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def get_embedding(self, audio_path: str):
        result = self.preprocessor(audio_path)
        embedding = self.model.extract(result["waveform"])

        return embedding

    @staticmethod
    def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        score = F.cosine_similarity(emb1, emb2, dim=0)

        return float(score.item())

    def compare_embeddings(self, emb1, emb2):
        score = self.cosine_similarity(emb1, emb2)

        confidence = (score + 1.0) / 2.0
        confidence = max(0.0, min(confidence, 1.0))

        return {
            "same": score >= self.threshold,
            "score": score,
            "confidence": confidence,
        }

    def compare_audio(self, audio1: str, audio2: str):
        emb1 = self.get_embedding(audio1)
        emb2 = self.get_embedding(audio2)

        return self.compare_embeddings(emb1, emb2)
