# audio/make_mel.py
import torch
import torchaudio


class MelExtractor:
    """
    通用 Mel 及 dB/归一化处理器。
    返回及期望的梅尔频谱维度始终是 (n_mels, T)。
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: int = 0,
        f_max: int = 8000,
        norm: str = "slaney",
        mel_scale: str = "slaney",
        power: float = 1.0,
        mean: torch.Tensor  = None,
        std: torch.Tensor  = None,
    ):
        # ---------- 频谱/梅尔变换 ----------
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
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

        # ---------- 均值/方差 ----------
        # 始终存成 float32 Tensor；设备在使用时动态对齐
        if mean is not None:
            mean = torch.as_tensor(mean, dtype=torch.float32)
        if std is not None:
            std = torch.as_tensor(std, dtype=torch.float32)

        self.mean = mean
        self.std = std
        self.n_mels = n_mels

    # -------------------------------------------------- #
    #                     正向提取                       #
    # -------------------------------------------------- #
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        wav : 1-D 或 (1,T) Tensor
        返回
        ----
        mel_db_norm : (n_mels, T)
        """
        if wav.dim() == 2:
            wav = wav.squeeze(0)

        mel = self.mel_transform(wav)     # (n_mels, T)
        mel_db = self.db_transform(mel)   # dB

        if self.mean is not None and self.std is not None:
            mean, std = self._match_device(mel_db.device)
            mel_db = (mel_db - mean[:, None]) / (std[:, None] + 1e-9)
        return mel_db

    # -------------------------------------------------- #
    #                    反归一化                        #
    # -------------------------------------------------- #
    def denormalize(self, mel_db: torch.Tensor) -> torch.Tensor:
        """
        支持输入形状：
            1. (B, n_mels,   T)
            2. (B,      T, n_mels)   —— 会自动转置
            3. (n_mels, T) / (T, n_mels)
        """
        if mel_db.dim() == 2:  # 无 batch
            if mel_db.shape[0] != self.n_mels and mel_db.shape[1] == self.n_mels:
                mel_db = mel_db.t()
        elif mel_db.dim() == 3:  # 有 batch
            if mel_db.shape[1] != self.n_mels and mel_db.shape[2] == self.n_mels:
                mel_db = mel_db.transpose(1, 2)
        else:
            raise ValueError("mel_db 维度必须为 2 或 3")

        if self.mean is not None and self.std is not None:
            mean, std = self._match_device(mel_db.device)
            mel_db = mel_db * std[:, None] + mean[:, None]
        return mel_db

    # -------------------------------------------------- #
    #               内部工具：对齐设备                    #
    # -------------------------------------------------- #
    def _match_device(self, device: torch.device):
        """
        把 mean/std 搬到目标设备（若尚未在该设备上）。
        返回搬好后的 (mean, std) 引用，避免重复 .to()
        """
        if self.mean.device != device:
            self.mean = self.mean.to(device)
        if self.std.device != device:
            self.std = self.std.to(device)
        return self.mean, self.std