# vocoder/diffwave_vocoder.py
from functools import lru_cache
import torch
from typing import Union

from speechbrain.inference.vocoders import DiffWaveVocoder as SB_DiffWave  # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_diffwave():
    print("⏬  Loading pretrained DiffWave (SpeechBrain)…")
    model = SB_DiffWave.from_hparams(
        source="speechbrain/tts-diffwave-ljspeech",
        savedir="pretrained/diffwave-ljspeech",
        run_opts={"device": DEVICE},
    )
    return model


@torch.no_grad()
def mel2wav_diffwave(mel_log: torch.Tensor) -> torch.Tensor:
    """
    mel_log 支持三种形状：
        (B, 1, 80, T)
        (B, 80, T)
        (80, T)            (会自动补 batch 维)
    输出：
        (B, N)  float32  [-1, 1]
    """
    # -------- reshape 到 (B, 80, T) -------- #
    if mel_log.dim() == 4:           # (B,1,80,T)
        mel = mel_log.squeeze(1)     # -> (B,80,T)
    elif mel_log.dim() == 3:         # (B,80,T)
        mel = mel_log
    elif mel_log.dim() == 2:         # (80,T)
        mel = mel_log.unsqueeze(0)   # -> (1,80,T)
    else:
        raise ValueError("mel_log must have 2/3/4 dims.")

    # -------- DiffWave 推理 -------- #
    model = load_diffwave()
    wav = model.decode_batch(mel, mel_lens=None, hop_len=256)  # [-1,1]
    return wav.to("cpu")