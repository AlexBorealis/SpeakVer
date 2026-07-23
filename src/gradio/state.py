from src.config import (
    DEFAULT_DEVICE,
    DEFAULT_THRESHOLD,
)
from src.speaker_verifier.speaker_verifier import SpeakerVerifier


class AppState:
    """
    Global application state.
    """

    def __init__(self):
        self.current_checkpoint = None
        self.reset()

    def reset(self):
        """
        Load the default pretrained model.
        """

        self.verifier = SpeakerVerifier(
            checkpoint=None,
            threshold=DEFAULT_THRESHOLD,
            device=DEFAULT_DEVICE,
        )

        self.current_checkpoint = None


state = AppState()