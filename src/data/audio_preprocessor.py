import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT
from tqdm import tqdm


class AudioPreprocessor:
    def __init__(
        self,
        target_sr=8000,
        mono=True,
        resample=True,
        remove_dc=True,
        peak_normalize=True,
        vad=True,
        augment=False,
        random_crop=False,
        max_duration=5.0,
        target_peak=0.8,
        vad_trigger_level=15.0,
        vad_search_time=0.03,
    ):
        self.target_sr = target_sr

        self.mono = mono
        self.resample = resample
        self.remove_dc = remove_dc
        self.peak_normalize = peak_normalize
        self.vad = vad

        self.augment = augment

        self.max_duration = max_duration
        self.max_samples = int(target_sr * max_duration)

        self.random_crop = random_crop

        self.target_peak = target_peak
        self.vad_trigger_level = vad_trigger_level
        self.vad_search_time = vad_search_time

        self.resamplers = {}

    def _get_resampler(self, sample_rate):
        if sample_rate not in self.resamplers:
            self.resamplers[sample_rate] = AT.Resample(sample_rate, self.target_sr)

        return self.resamplers[sample_rate]

    def _fix_length(self, waveform):
        length = waveform.shape[-1]

        if length > self.max_samples:
            if self.random_crop:
                start = random.randint(0, length - self.max_samples)
            else:
                start = 0

            waveform = waveform[:, start : start + self.max_samples]

        elif length < self.max_samples:
            waveform = F.pad(waveform, (0, self.max_samples - length))

        return waveform

    def _augment(self, waveform):
        if random.random() < 0.5:
            noise_level = random.uniform(0.002, 0.01)

            waveform = waveform + (torch.randn_like(waveform) * noise_level)

        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)

            waveform = waveform * gain

        return torch.clamp(waveform, -1, 1)

    def load_audio(self, audio_input):
        if hasattr(audio_input, "get_all_samples"):
            audio = audio_input.get_all_samples()
            waveform = audio.data

            sample_rate = audio.sample_rate

        elif isinstance(audio_input, dict):
            waveform = torch.tensor(
                audio_input["array"], dtype=torch.float32
            ).unsqueeze(0)

            sample_rate = audio_input["sampling_rate"]

        elif isinstance(audio_input, (str, None)):
            waveform, sample_rate = torchaudio.load(audio_input)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio_input)}")

        if self.mono:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

        if self.resample:
            if sample_rate != self.target_sr:
                waveform = self._get_resampler(sample_rate)(waveform)

        if self.remove_dc:
            waveform = waveform - waveform.mean()

        if self.peak_normalize:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak
                waveform = waveform * self.target_peak

        if self.vad:
            try:
                voiced = AF.vad(
                    waveform,
                    sample_rate=self.target_sr,
                    trigger_level=self.vad_trigger_level,
                    search_time=self.vad_search_time,
                )

                if voiced.numel() > 0:
                    waveform = voiced
            except Exception:
                pass

        if self.augment:
            waveform = self._augment(waveform)

        waveform = self._fix_length(waveform)

        return waveform

    def save_audio(self, audio_input, output_dir="debug_audio", filename="sample.wav"):
        os.makedirs(output_dir, exist_ok=True)

        waveform = self.load_audio(audio_input)

        if not filename.lower().endswith(".wav"):
            filename += ".wav"

        path = os.path.join(output_dir, filename)

        torchaudio.save(path, waveform.cpu(), self.target_sr)

        return path

    def save_samples(
        self,
        samples,
        output_dir="speaker_dataset",
        microphone="audio.throat_microphone",
    ):
        if isinstance(samples, dict):
            samples = [samples]

        output_dir = Path(output_dir)

        counters = {}

        for sample in tqdm(samples):
            speaker_id = sample["speaker_id"]

            speaker_dir = output_dir / speaker_id

            speaker_dir.mkdir(parents=True, exist_ok=True)

            if speaker_id not in counters:
                existing = list(speaker_dir.glob("*.wav"))

                counters[speaker_id] = len(existing)

            counters[speaker_id] += 1

            filename = f"{counters[speaker_id]:06d}.wav"

            waveform = self.load_audio(sample[microphone])

            torchaudio.save(str(speaker_dir / filename), waveform.cpu(), self.target_sr)

            metadata = {
                "speaker_id": sample["speaker_id"],
                "gender": sample.get("gender"),
                "sentence_id": sample.get("sentence_id"),
                "duration": sample.get("duration"),
                "raw_text": sample.get("raw_text"),
                "normalized_text": sample.get("normalized_text"),
                "phonemized_text": sample.get("phonemized_text"),
                "filename": filename,
            }

            metadata_path = speaker_dir / f"{Path(filename).stem}.json"

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
