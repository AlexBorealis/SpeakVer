import json
import random
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT
from tqdm import tqdm


class AudioPreprocessor:
    def __init__(
        self,
        target_sr: int = 8000,
        mono: bool = True,
        resample: bool = True,
        augment: bool = False,
        max_duration: float = 5.0,
        fix_length: bool = False,
        remove_dc: bool = True,
        peak_normalize: bool = True,
        target_peak: float = 0.8,
        vad: bool = True,
        vad_trigger_level: float = 15.0,
        vad_search_time: float = 0.03,
    ):
        self.target_sr = target_sr
        self.mono = mono
        self.resample = resample

        self.augment = augment

        self.max_duration = max_duration
        self.fix_length = fix_length
        self.max_samples = int(target_sr * max_duration)

        self.remove_dc = remove_dc
        self.peak_normalize = peak_normalize
        self.target_peak = target_peak

        self.vad = vad
        self.vad_trigger_level = vad_trigger_level
        self.vad_search_time = vad_search_time

        self.resamplers = {}

    def __call__(
        self,
        audio_input: str | Path | dict | Any,
    ) -> tuple[torch.Tensor, int]:
        waveform, sample_rate = self.load_audio(audio_input)

        waveform, length = self.process_audio(waveform, sample_rate)

        return {
            "waveform": waveform,
            "init_rate": sample_rate,
            "target_rate": self.target_sr,
            "length": length,
        }

    def _get_resampler(self, sample_rate: int):
        if sample_rate not in self.resamplers:
            self.resamplers[sample_rate] = AT.Resample(sample_rate, self.target_sr)

        return self.resamplers[sample_rate]

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.shape[-1]

        if length >= self.max_samples:
            waveform = waveform[:, 0 : 0 + self.max_samples]

        elif length < self.max_samples:
            waveform = F.pad(waveform, (0, self.max_samples - length))

        return waveform

    def _augment(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        # Additive Gaussian noise
        if random.random() <= 0.5:
            noise_std = random.uniform(0.001, 0.007)

            waveform = waveform + (torch.randn_like(waveform) * noise_std)

        # Random gain
        if random.random() <= 0.5:
            gain = random.uniform(0.9, 1.1)

            waveform = waveform * gain

        # Random polarity inversion
        if random.random() <= 0.5:
            waveform = -waveform

        return waveform.clamp_(-1.0, 1.0)

    def load_audio(
        self,
        audio_input: str | Path | dict | Any,
    ) -> tuple[torch.Tensor, int]:
        """
        Загружает аудио без какой-либо обработки.

        Parameters
        ----------
        audio_input
            Путь к файлу, HuggingFace Audio, словарь
            {"array", "sampling_rate"} или другой поддерживаемый объект.

        Returns
        -------
        waveform
            Tensor [channels, samples]

        sample_rate
            Частота дискретизации.
        """

        if hasattr(audio_input, "get_all_samples"):
            audio = audio_input.get_all_samples()

            return audio.data, audio.sample_rate

        if isinstance(audio_input, dict):
            return (
                torch.tensor(
                    audio_input["array"],
                    dtype=torch.float32,
                ).unsqueeze(0),
                audio_input["sampling_rate"],
            )

        if isinstance(audio_input, (str, Path)):
            return torchaudio.load(audio_input)

        raise TypeError(f"Unsupported audio type: {type(audio_input)}")

    def process_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> tuple[torch.Tensor, int]:
        """
        Выполняет всю предобработку аудио.

        Returns
        -------
        waveform
            Обработанный сигнал.

        length
            Итоговая длина после обработки
            (в количестве сэмплов).
        """
        if self.fix_length:
            waveform = self._fix_length(waveform)

        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if self.resample and sample_rate != self.target_sr:
            waveform = self._get_resampler(sample_rate)(waveform)
            sample_rate = self.target_sr

        if self.remove_dc:
            waveform = waveform - waveform.mean()

        if self.vad:
            try:
                voiced = AF.vad(
                    waveform,
                    sample_rate=sample_rate,
                    trigger_level=self.vad_trigger_level,
                    search_time=self.vad_search_time,
                )

                if voiced.numel():
                    waveform = voiced

            except Exception:
                pass

        if self.peak_normalize:
            peak = waveform.abs().amax()

            if peak > 0:
                waveform = waveform / peak
                waveform *= self.target_peak

        if self.augment:
            waveform = self._augment(waveform)

        return waveform, waveform.shape[-1]

    def save_samples(
        self,
        samples: list,
        output_dir: str = "speaker_dataset",
    ) -> tuple[int, int]:

        if isinstance(samples, dict):
            samples = [samples]

        output_dir = Path(output_dir)

        microphones = [
            "audio.headset_microphone",
            "audio.forehead_accelerometer",
            "audio.soft_in_ear_microphone",
            "audio.rigid_in_ear_microphone",
            "audio.temple_vibration_pickup",
            "audio.throat_microphone",
        ]

        saved = 0
        failed = 0

        for sample in tqdm(samples, leave=False):
            try:
                speaker_id = sample["speaker_id"]
                sentence_id = str(sample["sentence_id"])

                speaker_dir = output_dir / speaker_id
                speaker_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                metadata = {}

                for key, value in sample.items():
                    if key.startswith("audio."):
                        continue

                    metadata[key] = value

                metadata["filename"] = f"{sentence_id}.wav"

                with open(
                    speaker_dir / f"{sentence_id}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        metadata,
                        f,
                        ensure_ascii=False,
                        indent=4,
                    )

                for microphone in microphones:
                    if microphone not in sample:
                        continue

                    mic_name = microphone.removeprefix("audio.")

                    mic_dir = speaker_dir / mic_name
                    mic_dir.mkdir(
                        exist_ok=True,
                    )

                    audio = sample[microphone].get_all_samples()

                    torchaudio.save(
                        str(mic_dir / f"{sentence_id}.wav"),
                        audio.data.cpu(),
                        audio.sample_rate,
                    )

                saved += 1

            except Exception:
                failed += 1

                print("=" * 80)
                print(f"Failed sample #{saved + failed}")
                print(f"Speaker : {sample.get('speaker_id')}")
                print(f"Sentence: {sample.get('sentence_id')}")
                print(traceback.format_exc())
                print("=" * 80)

        print(f"Batch finished. Saved: {saved}, Failed: {failed}")

        return saved, failed
