from pathlib import Path

from src.config import RUNS_DIR

DEFAULT_MODEL_NAME = "Default model"


def get_models() -> list[str]:
    """
    Return available experiment names.
    """
    if not RUNS_DIR.exists():
        return []

    return sorted(
        directory.name for directory in RUNS_DIR.iterdir() if directory.is_dir()
    )


def get_checkpoints(
    experiment: str,
) -> list[str]:
    """
    Return available checkpoints for an experiment.
    """
    if not experiment or experiment == DEFAULT_MODEL_NAME:
        return []

    weights_dir = RUNS_DIR / experiment / "weights"

    if not weights_dir.exists():
        return []

    return sorted(checkpoint.name for checkpoint in weights_dir.glob("*.pt"))


def resolve_checkpoint(
    experiment: str,
    checkpoint: str | None = None,
) -> Path | None:
    """
    Resolve checkpoint path.

    Returns
    -------
    Path | None
        None:
            Use pretrained SpeechBrain model.
        Path:
            Use specified checkpoint.
    """
    if not experiment or experiment == DEFAULT_MODEL_NAME:
        return None

    if checkpoint and checkpoint != DEFAULT_MODEL_NAME:
        checkpoint_path = RUNS_DIR / experiment / "weights" / checkpoint
    else:
        checkpoint_path = RUNS_DIR / experiment / "weights" / "best.pt"

    if checkpoint_path.exists():
        return checkpoint_path

    return None
