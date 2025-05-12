# audio/make_mel.py
import torch
import torchaudio

class MelExtractor:
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        f_min=0,
        f_max=8000,
        norm="slaney",
        mel_scale="slaney",
        power=1.0,
        mean=None,
        std=None,
    ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            norm=norm,
            mel_scale=mel_scale,
            power=power,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.mean = mean
        self.std = std

    def __call__(self, wav: torch.Tensor):
        if wav.dim() == 2:
            wav = wav.squeeze(0)
        mel = self.mel_transform(wav)
        mel_db = self.db_transform(mel)
        if self.mean is not None and self.std is not None:
            mel_db = (mel_db - self.mean[:, None]) / (self.std[:, None] + 1e-9)
        return mel_db

    def denormalize(self, mel_db):
        if self.mean is not None and self.std is not None:
            return mel_db * self.std[:, None] + self.mean[:, None]
        return mel_db