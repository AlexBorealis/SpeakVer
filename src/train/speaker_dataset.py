import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from src.data.audio_preprocessor import AudioPreprocessor


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

    def __init__(
        self,
        root_dir,
        preprocessor: AudioPreprocessor = None,
        return_audio: bool = True,
        microphone: str | None = None,
        shuffle: bool = False,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.preprocessor = preprocessor
        self.return_audio = return_audio
        self.microphone = microphone
        self.shuffle = shuffle

        self.samples = []
        self.speakers = {}

        self._scan()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        if not self.return_audio:
            return sample

        config = random.choice(list(sample["domains"]))

        microphones = sample["domains"][config]

        microphone = self.microphone
        if self.shuffle or self.microphone is None:
            microphone = random.choice(list(microphones))

        path = microphones[microphone]

        waveform, length = self.preprocessor(path)

        return {
            "waveform": waveform.squeeze(0),
            "speaker": sample["speaker_index"],
            "length": length,
            "config": config,
            "microphone": microphone,
            "path": path,
        }

    def get_speakers(self):
        return self.speakers

    def get_num_speakers(self):
        return len(self.speakers)

    def get_num_samples(self):
        return len(self.samples)

    def get_mic_type(self):
        config = random.choice(list(self.samples[0]["domains"]))

        microphones = self.samples[0]["domains"][config]
        return list(microphones.keys())

    def _scan(self):
        self.samples.clear()
        self.speakers.clear()

        recordings = {}

        speaker_index = 0

        # speech_clean, speech_noise, ...
        for config_dir in sorted(self.root_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            config = config_dir.name

            # speaker_001
            for speaker_dir in sorted(config_dir.iterdir()):
                if not speaker_dir.is_dir():
                    continue

                speaker_id = speaker_dir.name

                if speaker_id not in self.speakers:
                    self.speakers[speaker_id] = speaker_index
                    speaker_index += 1

                speaker_idx = self.speakers[speaker_id]

                for microphone_dir in speaker_dir.iterdir():
                    if not microphone_dir.is_dir():
                        continue

                    microphone = microphone_dir.name

                    for wav_path in microphone_dir.glob("*.wav"):
                        recording = wav_path.stem

                        key = (
                            speaker_id,
                            recording,
                        )

                        if key not in recordings:
                            recordings[key] = {
                                "speaker_id": speaker_id,
                                "speaker_index": speaker_idx,
                                "recording": recording,
                                "domains": {},
                            }

                        recordings[key]["domains"].setdefault(config, {})[
                            microphone
                        ] = str(wav_path)

        self.samples = list(recordings.values())

    def statistics(self):
        """
        Statistics of Dataset.
        """

        speaker_counter = Counter()
        domain_counter = Counter()
        microphone_counter = Counter()

        durations = []

        for sample in self.samples:
            speaker_counter[sample["speaker_id"]] += 1

            for domain, microphones in sample["domains"].items():
                domain_counter[domain] += 1

                for microphone, _ in microphones.items():
                    microphone_counter[microphone] += 1

                # Берем длительность только один раз для данного домена
                first_path = next(iter(microphones.values()))
                info = sf.info(first_path)
                durations.append(info.duration)

        counts = np.array(list(speaker_counter.values()))
        durations = np.array(durations)

        return {
            "num_speakers": len(speaker_counter),
            "num_samples": len(self.samples),
            "min_qty": int(counts.min()),
            "max_qty": int(counts.max()),
            "mean_qty": float(counts.mean()),
            "median_qty": float(np.median(counts)),
            "std_qty": float(counts.std()),
            "min_duration": float(durations.min()),
            "max_duration": float(durations.max()),
            "mean_duration": float(durations.mean()),
            "median_duration": float(np.median(durations)),
            "std_duration": float(durations.std()),
            "domain_distribution": dict(domain_counter),
            "microphone_distribution": dict(microphone_counter),
            "speaker_distribution": speaker_counter,
            "durations_distribution": durations.tolist(),
        }

    def save_dataset_stats(
        self,
        data: dict[str, Any],
        save_dir: str | Path,
        file_path: str = "dataset_stats.json",
    ) -> None:
        """Преобразует статистику датасета в JSON и сохраняет в файл.

        :param data: Словарь с данными (может содержать объекты Counter).
        :param file_path: Путь к файлу для сохранения.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        full_path = save_dir / file_path

        processed_data = data.copy()

        if "speaker_distribution" in processed_data:
            if isinstance(processed_data["speaker_distribution"], Counter):
                processed_data["speaker_distribution"] = dict(
                    processed_data["speaker_distribution"]
                )

        # Записываем данные в JSON-файл
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)
            print(f"Saving complete: {file_path}")
        except IOError as e:
            print(f"Error when saving file {file_path}: {e}")

    def summary(self):
        print("=" * 60)
        print("Speaker Dataset")
        print("=" * 60)
        print(f"Dataset path : {self.root_dir}")
        print(f"Speakers     : {self.get_num_speakers()}")
        print(f"Audio files  : {self.get_num_samples()}")
        print("=" * 60)
