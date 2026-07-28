import random
import shutil
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset

from src.train.speaker_dataset import SpeakerDataset


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function для speaker verification.

    Выполняет:
    - padding до максимальной длины внутри batch;
    - вычисление относительных длин;
    - формирование labels.

    Parameters
    ----------
    batch
        Список элементов Dataset.

        Каждый элемент:

        {
            "waveform": Tensor [1, T],
            "length": int,
            "speaker": int,
        }

    Returns
    -------
    dict

        {
            "waveforms": Tensor [B, 1, Tmax],
            "lengths": Tensor [B],
            "labels": Tensor [B],
        }
    """

    waveforms = [sample["waveform"].squeeze(0) for sample in batch]

    lengths = torch.tensor(
        [sample["length"] for sample in batch],
        dtype=torch.long,
    )

    labels = torch.tensor(
        [sample["speaker"] for sample in batch],
        dtype=torch.long,
    )

    waveforms = pad_sequence(
        waveforms,
        batch_first=True,
    )

    relative_lengths = lengths.float() / lengths.max()

    return {
        "waveforms": waveforms,
        "lengths": relative_lengths,
        "labels": labels,
    }


def has_speaker_dirs(path):
    path = Path(path)

    if not path.exists():
        return False

    return any(item.is_dir() for item in path.iterdir())


def split_dataset(
    dataset,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
    split_key="speaker_id",
    persistent=False,
    output_dir="speaker_dataset",
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    output_dir = Path(output_dir)

    if persistent:
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        test_dir = output_dir / "test"

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        split_exists = (
            has_speaker_dirs(train_dir)
            and has_speaker_dirs(val_dir)
            and has_speaker_dirs(test_dir)
        )

        if split_exists:
            print("Found existing dataset split. Loading...")

            return (
                SpeakerDataset(str(train_dir), return_audio=False),
                SpeakerDataset(str(val_dir), return_audio=False),
                SpeakerDataset(str(test_dir), return_audio=False),
            )

    random.seed(seed)

    # Используем внутренний список samples, а не __getitem__()
    samples = dataset.samples if hasattr(dataset, "samples") else dataset

    speakers = list({item[split_key] for item in samples})
    random.shuffle(speakers)

    n = len(speakers)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": set(speakers[:train_end]),
        "val": set(speakers[train_end:val_end]),
        "test": set(speakers[val_end:]),
    }

    if not persistent:
        indices = {"train": [], "val": [], "test": []}

        for i, item in enumerate(samples):
            for name, spks in splits.items():
                if item[split_key] in spks:
                    indices[name].append(i)

        return (
            Subset(dataset, indices["train"]),
            Subset(dataset, indices["val"]),
            Subset(dataset, indices["test"]),
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    source_root = dataset.root_dir

    for split_name, spks in splits.items():
        split_path = output_dir / split_name
        split_path.mkdir(exist_ok=True)

        for speaker in spks:
            src = source_root / speaker
            dst = split_path / speaker

            if src.exists() and not dst.exists():
                shutil.move(src, dst)

    return (
        SpeakerDataset(str(output_dir / "train")),
        SpeakerDataset(str(output_dir / "val")),
        SpeakerDataset(str(output_dir / "test")),
    )
