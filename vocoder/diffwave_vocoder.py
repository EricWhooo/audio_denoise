# vocoder/diffwave_vocoder.py
from functools import lru_cache
import torch

DiffWave = None

from speechbrain.inference.vocoders import DiffWaveVocoder as SB_DiffWave  # type: ignore
DiffWave = SB_DiffWave

# ------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def load_diffwave():
    print("⏬  Loading pretrained DiffWave (SpeechBrain)…")
    model = DiffWave.from_hparams(
        source="speechbrain/tts-diffwave-ljspeech",
        savedir="pretrained/diffwave-ljspeech",
        run_opts={"device": DEVICE},
    )
    return model

@torch.no_grad()
def mel2wav_diffwave(mel_log: torch.Tensor) -> torch.Tensor:
    """
    mel_log: (B, 1, 80, T)  ——  对数梅尔谱 (log-mel **dB**)
    return : (B, N) float32, 取值 ±1
    """
    model = load_diffwave()
    mel = mel_log.squeeze(1).permute(0, 2, 1)  # (B, T, 80)
    wav = model.decode_batch(mel, mel_lens=None, hop_len=256)  # [-1,1] float32
    return wav