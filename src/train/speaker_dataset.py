import json
from collections import Counter
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class SpeakerDataset(Dataset):
    """
    Универсальный датасет Speaker Verification.

    Структура каталога:

    dataset/
        speaker_001/
            audio1.wav
            audio2.wav

        speaker_002/
            audio1.wav
            audio2.wav
    """

    AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

    def __init__(self, root_dir, preprocessor=None, return_audio=True):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.preprocessor = preprocessor
        self.return_audio = return_audio

        self.samples = []
        self.speakers = {}

        self._scan()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # режим для PairBuilder / validation
        if not self.return_audio:
            return sample

        # режим обучения
        waveform = self.preprocessor.load_audio(sample["path"])

        return (waveform.squeeze(0), sample["speaker_index"])

    def get_speakers(self):
        return self.speakers

    def get_num_speakers(self):
        return len(self.speakers)

    def get_num_samples(self):
        return len(self.samples)

    def _scan(self):
        self.samples.clear()
        self.speakers.clear()

        speaker_index = 0

        for speaker_dir in sorted(self.root_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_name = speaker_dir.name
            self.speakers[speaker_name] = speaker_index

            for file in speaker_dir.rglob("*"):
                if not file.is_file():
                    continue

                if file.suffix.lower() not in self.AUDIO_EXTENSIONS:
                    continue

                duration = None

                metadata_file = file.with_suffix(".json")

                if metadata_file.exists():
                    try:
                        with open(metadata_file, encoding="utf-8") as f:
                            metadata = json.load(f)

                        duration = metadata.get("duration")

                    except Exception:
                        duration = None

                self.samples.append(
                    {
                        "speaker_id": speaker_name,
                        "speaker_index": speaker_index,
                        "path": str(file),
                        "duration": duration,
                    }
                )

            speaker_index += 1

    def statistics(self):
        """
        Statistics of Dataset
        """
        counter = Counter()

        durations = []

        for sample in self.samples:
            counter[sample["speaker_id"]] += 1

            if sample["duration"] is not None:
                durations.append(sample["duration"])

        counts = np.array(list(counter.values()))
        durations = np.array(durations)

        return {
            "num_speakers": len(counter),
            "num_samples": len(self.samples),
            "min_qty": int(counts.min()),
            "max_qty": int(counts.max()),
            "mean_qty": float(counts.mean()),
            "median_qty": float(np.median(counts)),
            "std_qty": float(counts.std()),
            "min_duration": float(durations.min()),
            "max_duration": float(durations.max()),
            "mean_duration": float(durations.mean()),
            "median_duration": float(np.nanmedian(durations)),
            "std_duration": float(durations.std()),
            "distribution": counter,
        }

    def summary(self):
        print("=" * 60)
        print("Speaker Dataset")
        print("=" * 60)
        print(f"Dataset path : {self.root_dir}")
        print(f"Speakers     : {self.get_num_speakers()}")
        print(f"Audio files  : {self.get_num_samples()}")
        print("=" * 60)
