from src.config import (
    DEFAULT_DEVICE,
    DEFAULT_THRESHOLD,
)
from src.gradio.state import state
from src.gradio.utils.model_utils import resolve_checkpoint
from src.speaker_verifier.speaker_verifier import SpeakerVerifier


class ModelService:
    """
    Service responsible for loading models and running speaker verification.
    """

    @staticmethod
    def load_model(
        experiment: str,
        checkpoint: str,
    ) -> str:
        """
        Load selected checkpoint.
        Returns model status string.
        """
        checkpoint_path = resolve_checkpoint(
            experiment,
            checkpoint,
        )

        state.verifier = SpeakerVerifier(
            checkpoint=checkpoint_path,
            threshold=DEFAULT_THRESHOLD,
            device=DEFAULT_DEVICE,
        )

        state.current_checkpoint = checkpoint_path

        if checkpoint_path is None:
            return "Using default SpeechBrain ECAPA-TDNN"

        return f"Loaded checkpoint:\n{checkpoint_path}"

    @staticmethod
    def compare(
        audio1: str,
        audio2: str,
        threshold: float,
    ) -> dict:
        """
        Compare two audio files.
        """
        if audio1 is None or audio2 is None:
            raise ValueError("Please upload two audio files.")

        state.verifier.threshold = threshold

        return state.verifier.compare_audio(
            audio1,
            audio2,
        )
