from pathlib import Path
import shutil

import torch
from torch.utils.data import Subset
import random

from src.train.speaker_dataset import SpeakerDataset

def has_speaker_dirs(path):
    path = Path(path)

    if not path.exists():
        return False

    return any(
        item.is_dir()
        for item in path.iterdir()
    )


def load_checkpoint(
    path,
    extractor,
    criterion=None
):
    ckpt = torch.load(
        path,
        map_location="cpu"
    )

    extractor.load_state_dict(
        ckpt["encoder"]
    )

    if criterion:
        criterion.load_state_dict(
            ckpt["criterion"]
        )

    return extractor


def split_dataset(
    dataset,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
    split_key="speaker_id",
    persistent=False,
    output_dir="speaker_dataset"
):
    assert abs(
        train_ratio + val_ratio + test_ratio - 1.0
    ) < 1e-6

    output_dir = Path(output_dir)

    if persistent:
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        test_dir = output_dir / "test"

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
                SpeakerDataset(str(test_dir), return_audio=False)
            )

    random.seed(seed)

    speakers = list(
        set(
            item[split_key]
            for item in dataset
        )
    )

    random.shuffle(speakers)

    n = len(speakers)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": set(speakers[:train_end]),
        "val": set(speakers[train_end:val_end]),
        "test": set(speakers[val_end:])
    }

    if not persistent:
        indices = {
            "train": [],
            "val": [],
            "test": []
        }

        for i, item in enumerate(dataset):
            for name, spks in splits.items():
                if item[split_key] in spks:
                    indices[name].append(i)

        return (
            Subset(dataset, indices["train"]),
            Subset(dataset, indices["val"]),
            Subset(dataset, indices["test"])
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    source_root = dataset.root_dir

    for split_name, spks in splits.items():
        split_path = output_dir / split_name
        split_path.mkdir(exist_ok=True)

        for speaker in spks:
            src = source_root / speaker
            dst = split_path / speaker

            if src.exists():
                shutil.move(src, dst)

    return (
        SpeakerDataset(str(output_dir / "train")),
        SpeakerDataset(str(output_dir / "val")),
        SpeakerDataset(str(output_dir / "test"))
    )


def get_audio_input(item):
    keys = [
        "audio.throat_microphone",
        "path",
        "audio"
    ]

    for key in keys:
        if key in item:
            return item[key]

    raise KeyError(
        f"Audio field not found. Available keys: {item.keys()}"
    )

def get_sample_key(sample):
    if "path" in sample:
        return sample["path"]

    if "id" in sample:
        return str(sample["id"])

    return str(id(sample))