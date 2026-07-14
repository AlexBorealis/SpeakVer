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

                self.samples.append(
                    {
                        "speaker_id": speaker_name,
                        "speaker_index": speaker_index,
                        "path": str(file),
                    }
                )

            speaker_index += 1

    def statistics(self):
        """
        Вывод статистики по датасету.
        """
        counter = Counter()

        for sample in self.samples:
            counter[sample["speaker_id"]] += 1

        counts = np.array(list(counter.values()))

        print("=" * 60)
        print("Speaker Dataset Statistics")
        print("=" * 60)

        print(f"Speakers              : {len(counter)}")
        print(f"Audio files           : {len(self.samples)}")
        print(f"Min recordings        : {counts.min()}")
        print(f"Max recordings        : {counts.max()}")
        print(f"Mean recordings       : {counts.mean():.2f}")
        print(f"Median recordings     : {np.median(counts):.2f}")
        print(f"Std recordings        : {counts.std():.2f}")

        print("=" * 60)
        print("Top-10 speakers")
        print("=" * 60)

        for speaker, n in counter.most_common(10):
            print(f"{speaker:15s} {n}")

        print("=" * 60)

        return {
            "num_speakers": len(counter),
            "num_samples": len(self.samples),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "std": float(counts.std()),
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
