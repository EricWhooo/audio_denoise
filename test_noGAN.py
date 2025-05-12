# test_noGAN.py —— 预训练 DiffWave (SpeechBrain) + pysepm 评测
"""
示例：
  python test_noGAN.py \
      --root /vast/lb4434/datasets/voicebank-demand \
      --exp vae_run80mel_56_0.1 \
      --subset 56spk \
      --save_dir ./samples/vae_run80mel_56_0.1_2
"""

from __future__ import annotations
import argparse, random
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch, soundfile as sf
from tqdm import tqdm

from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio as SISDR,
    ShortTimeObjectiveIntelligibility   as STOI,
    PerceptualEvaluationSpeechQuality   as PESQ,
)

from pysepm import composite as sepm_composite        # CSIG/CBAK/COVL
from vocoder.diffwave_vocoder import mel2wav_diffwave # ← 新增封装

# ---------- 项目内部 ----------------------------------------------
from training.dataset                 import create_dataloaders
from variational_autoencoder.autoencoder import AutoencoderKL
from audio.stft                       import TacotronSTFT
from audio.audio_processing           import dynamic_range_decompression

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", DEVICE)

# ---------- helper: resample to 16 kHz for PESQ --------------------
import torchaudio

def _to_16k(tensor_22k: torch.Tensor) -> torch.Tensor:
    """tensor: (B, N) • float32 [-1,1]"""
    return torchaudio.functional.resample(
        tensor_22k, orig_freq=22050, new_freq=16000)

# ---------- helper: 22 k → 16 k ----------
def resample_22k_to_16k(wav_tensor: torch.Tensor) -> torch.Tensor:
    # wav_tensor: (B, N_22k) float32 [-1, 1]
    return torchaudio.functional.resample(wav_tensor, 22050, 16000)

# ---------- VAE config --------------------------------------------
def ddconfig() -> Dict:
    return dict(
        ch=128, out_ch=1, ch_mult=(1,2,4), num_res_blocks=2, attn_resolutions=[],
        dropout=0.0, resamp_with_conv=True, in_channels=1, resolution=1024,
        z_channels=32)

# ---------- log-mel → waveform (DiffWave) --------------------------
@torch.no_grad()
def mel_to_waveform(mel: torch.Tensor) -> np.ndarray:
    """
    mel: (B,1,80,T)  —— ln 压缩后的 log-mel（与 VAE 输出一致）
    返回  int16 numpy (B,N)
    """
    mel_lin   = dynamic_range_decompression(mel.squeeze(1))          # (B,T,80)
    mel_log10 = torch.log10(mel_lin.clamp(min=1e-5)).unsqueeze(1)    # (B,1,80,T)
    wav = mel2wav_diffwave(mel_log10)                                # (B,N) ±1
    return (wav.cpu().numpy() * 32767).astype(np.int16)

# ---------- evaluate ------------------------------------------------
@torch.no_grad()
def evaluate(model, loader,
             sisdr_m, stoi_m, pesq_m,
             save_dir: Optional[Path], n_examples=3):

    sums = dict(sisdr=0.0, stoi=0.0, pesq=0.0,
                csig=0.0, cbak=0.0, covl=0.0)
    save_ids = set(random.sample(range(len(loader)), min(n_examples, len(loader))))
    pbar = tqdm(loader, desc="Testing")

    for idx, (noisy_mel, clean_mel) in enumerate(pbar):
        noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)
        den_mel, _ = model(noisy_mel, sample_posterior=False)

        den_wav   = mel_to_waveform(den_mel)/32768.0
        clean_wav = mel_to_waveform(clean_mel)/32768.0
        noisy_wav = mel_to_waveform(noisy_mel)/32768.0

        den_t, clean_t = map(lambda x: torch.from_numpy(x).to(DEVICE),
                             (den_wav, clean_wav))
        den_16k   = resample_22k_to_16k(den_t)
        clean_16k = resample_22k_to_16k(clean_t)
        sums["sisdr"] += sisdr_m(den_t, clean_t).cpu()
        sums["stoi"]  += stoi_m (den_t, clean_t).cpu()
        sums["pesq"] += pesq_m(den_16k, clean_16k).cpu()

        csig, cbak, covl = sepm_composite(
            clean_16k[0].cpu().numpy().astype("float32"),
            den_16k  [0].cpu().numpy().astype("float32"),
            fs=16000)
        sums["csig"] += csig; sums["cbak"] += cbak; sums["covl"] += covl

        if save_dir and idx in save_ids:
            sf.write(save_dir/f"sample{idx}_noisy.wav",   noisy_wav[0],22050)
            sf.write(save_dir/f"sample{idx}_denoised.wav",den_wav [0],22050)
            sf.write(save_dir/f"sample{idx}_clean.wav",   clean_wav[0],22050)

    n = len(loader)
    for k in sums: sums[k] /= n
    return sums

# ---------- main ----------------------------------------------------
def main(cfg):
    out_dir = Path(cfg.save_dir) if cfg.save_dir else None
    if out_dir: out_dir.mkdir(parents=True, exist_ok=True)

    stft_cpu = TacotronSTFT()   # 参数已改为 22k/80/256
    _,_,test_loader = create_dataloaders(
        cfg.root, dict(target_length=512, fn_STFT=stft_cpu),
        batch_size=1, subset_type=cfg.subset)

    model = AutoencoderKL(ddconfig=ddconfig(), embed_dim=32,
                          image_key="fbank", time_shuffle=1).to(DEVICE)
    model.load_state_dict(torch.load(Path("runs")/cfg.exp/"best.ckpt",
                                      map_location=DEVICE))

    sisdr = SISDR().to(DEVICE)
    stoi  = STOI(22050, False).to(DEVICE)
    pesq  = PESQ(16000, "wb").to(DEVICE)

    res = evaluate(model, test_loader,
                   sisdr, stoi, pesq, out_dir)

    print(f"[TEST] SI-SDR={res['sisdr']:.2f} dB | STOI={res['stoi']:.3f} | "
          f"PESQ={res['pesq']:.3f} | CSIG={res['csig']:.3f} | "
          f"CBAK={res['cbak']:.3f} | COVL={res['covl']:.3f}")

# ---------- CLI -----------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", required=True)
    pa.add_argument("--subset", default="28spk")
    pa.add_argument("--exp",    default="vae_denoise")
    pa.add_argument("--save_dir", default="")
    main(pa.parse_args())