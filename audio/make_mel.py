# audio/make_mel.py
import torch
from speechbrain.lobes.models.HifiGAN import mel_spectogram

class MelExtractor:
    """
    SpeechBrain 官方 mel_spectogram 包装器。
    - 返回张量形状： (n_mels, T)
    - 内部直接产生对 DiffWave 友好的 ln 压缩梅尔，不再做全局均值/方差归一化。
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 256,
        win_length: int = 1024,
        n_fft: int = 1024,
        n_mels: int = 80,
        f_min: int = 0,
        f_max: int = 8000,
        power: float = 1.0,
        normalized: bool = False,
        norm: str = "slaney",
        mel_scale: str = "slaney",
    ):
        self.cfg = dict(
            sample_rate=sample_rate,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
            normalized=normalized,
            norm=norm,
            mel_scale=mel_scale,
            compression=True,      # ★ 与 SpeechBrain 示例保持一致
        )
        self.n_mels = n_mels

    # ---------------- 正向：wav → mel ---------------- #
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: 1-D 或 (1,T) Tensor, 采样率必须等于 cfg.sample_rate。
        return: (n_mels, T)  — ln 压缩梅尔
        """
        if wav.dim() == 2:
            wav = wav.squeeze(0)
        mel = mel_spectogram(audio=wav, **self.cfg)       # (1,n_mels,T)
        return mel.squeeze(0)                             # (n_mels,T)

    # ---------------- 反压缩：mel → 线性域 ----------- #
    def decompress(self, mel_db: torch.Tensor) -> torch.Tensor:
        """
        把 ln 压缩的梅尔转换回线性功率域（DiffWave 需要）。
        输入可带 batch：
            (B,n_mels,T) 或 (B,1,n_mels,T) 或 (n_mels,T)
        输出形状与输入保持一致。
        """
        need_unsqueeze = False
        if mel_db.dim() == 2:          # (n_mels,T)
            mel_lin = torch.exp(mel_db)
        elif mel_db.dim() == 3:        # (B,n_mels,T)
            mel_lin = torch.exp(mel_db)
        elif mel_db.dim() == 4:        # (B,1,n_mels,T)
            need_unsqueeze = True
            mel_lin = torch.exp(mel_db.squeeze(1))
        else:
            raise RuntimeError(f"Unexpected mel shape {mel_db.shape}")
        if need_unsqueeze:
            mel_lin = mel_lin.unsqueeze(1)
        return mel_lin