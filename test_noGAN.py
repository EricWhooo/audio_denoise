# test.py  ── 用 Griffin-Lim 重建波形
"""
Usage:
  python test_noGAN.py \
      --root /vast/lb4434/datasets/voicebank-demand \
      --exp vae_run4_56_beta1 \
      --subset 56spk \
      --save_dir ./samples/vae_run4_56_beta1
"""

# test_noGAN.py  ── GPU Griffin-Lim + pysepm (CSIG/CBAK/COVL)
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

# ---------- pysepm  (CSIG / CBAK / COVL) ---------------------------
from pysepm import composite as sepm_composite      # pip install pysepm

# ---------- 项目内部 ----------------------------------------------
from training.dataset                 import create_dataloaders
from variational_autoencoder.autoencoder import AutoencoderKL
from audio.stft                       import TacotronSTFT
from audio.audio_processing           import dynamic_range_decompression

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", DEVICE)

# ---------- GPU Griffin-Lim ----------------------------------------
def griffin_lim_gpu(mag: torch.Tensor, n_fft: int, hop: int, n_iters: int = 32):
    B, F, T = mag.shape
    win = torch.hann_window(n_fft, device=mag.device)
    phase = torch.rand(B, F, T, device=mag.device) * 2 * np.pi
    for _ in range(n_iters):
        spec  = torch.polar(mag, phase)
        wav   = torch.istft(spec, n_fft, hop, n_fft, win, center=True)
        spec  = torch.stft (wav , n_fft, hop, n_fft, win,
                            center=True, return_complex=True)
        phase = spec.angle()
    wav = torch.istft(torch.polar(mag, phase), n_fft, hop, n_fft, win, center=True)
    wav = wav / wav.abs().amax(1, keepdim=True).clamp(min=1e-5)
    return wav                                      # (B, N), float32

# ---------- Tacotron STFT (CPU-side) -------------------------------
def build_stft():
    return TacotronSTFT(
        filter_length=1024, hop_length=160, win_length=1024,
        n_mel_channels=64, sampling_rate=16000, mel_fmin=0, mel_fmax=8000)

# ---------- VAE config --------------------------------------------
def ddconfig() -> Dict:
    return dict(
        ch=128, out_ch=1, ch_mult=(1,2,4), num_res_blocks=2, attn_resolutions=[],
        dropout=0.0, resamp_with_conv=True, in_channels=1, resolution=1024,
        z_channels=32)

# ---------- log-mel → waveform ------------------------------------
@torch.no_grad()
def mel_to_waveform(mel: torch.Tensor, stft_cpu: TacotronSTFT, n_iter=32):
    stft_gpu = stft_cpu.to(DEVICE)
    if not hasattr(stft_gpu, "_mel_inv"):
        stft_gpu._mel_inv = torch.pinverse(stft_gpu.mel_basis).to(DEVICE)  # (513,64)

    n_fft, hop = stft_gpu.stft_fn.filter_length, stft_gpu.stft_fn.hop_length
    mel = mel.squeeze(1).permute(0,2,1)                # (B,64,512)
    mel_lin = dynamic_range_decompression(mel)
    mag = (stft_gpu._mel_inv @ mel_lin).clamp_min(1e-5)  # (B,513,512)

    wav = griffin_lim_gpu(mag, n_fft, hop, n_iters=n_iter)
    return (wav.cpu().numpy() * 32767).astype(np.int16)   # (B,N) int16

# ---------- evaluate ------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, stft_cpu,
             sisdr_m, stoi_m, pesq_m,
             save_dir: Optional[Path], n_examples=3):

    sums = dict(sisdr=0.0, stoi=0.0, pesq=0.0,
                csig=0.0, cbak=0.0, covl=0.0)
    save_ids = set(random.sample(range(len(loader)), min(n_examples, len(loader))))
    pbar = tqdm(loader, desc="Testing")

    for idx, (noisy_mel, clean_mel) in enumerate(pbar):
        noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)
        den_mel, _ = model(noisy_mel, sample_posterior=False)

        den_wav   = mel_to_waveform(den_mel,   stft_cpu)/32768.0
        clean_wav = mel_to_waveform(clean_mel, stft_cpu)/32768.0
        noisy_wav = mel_to_waveform(noisy_mel, stft_cpu)/32768.0

        den_t, clean_t = map(lambda x: torch.from_numpy(x).to(DEVICE),
                             (den_wav, clean_wav))
        sums["sisdr"] += sisdr_m(den_t, clean_t).cpu()
        sums["stoi"]  += stoi_m (den_t, clean_t).cpu()
        sums["pesq"]  += pesq_m (den_t, clean_t).cpu()

        # pysepm composite 返回 csig, cbak, covl, segSNR
        csig, cbak, covl = sepm_composite(
            clean_wav[0].astype(np.float32),
            den_wav  [0].astype(np.float32),
            fs=16000)

        sums["csig"] += csig
        sums["cbak"] += cbak
        sums["covl"] += covl

        if save_dir and idx in save_ids:
            sf.write(save_dir/f"sample{idx}_noisy.wav",   noisy_wav[0],16000)
            sf.write(save_dir/f"sample{idx}_denoised.wav",den_wav [0],16000)
            sf.write(save_dir/f"sample{idx}_clean.wav",   clean_wav[0],16000)

    n = len(loader)
    for k in sums: sums[k] /= n
    return sums

# ---------- main ----------------------------------------------------
def main(cfg):
    out_dir = Path(cfg.save_dir) if cfg.save_dir else None
    if out_dir: out_dir.mkdir(parents=True, exist_ok=True)

    stft_cpu = build_stft()
    _,_,test_loader = create_dataloaders(
        cfg.root, dict(target_length=512, fn_STFT=stft_cpu),
        batch_size=1, subset_type=cfg.subset)

    model = AutoencoderKL(ddconfig=ddconfig(), embed_dim=32,
                          image_key="fbank", time_shuffle=1).to(DEVICE)
    model.load_state_dict(torch.load(Path("runs")/cfg.exp/"best.ckpt",
                                      map_location=DEVICE))

    sisdr = SISDR().to(DEVICE)
    stoi  = STOI(16000, False).to(DEVICE)
    pesq  = PESQ(16000, "wb").to(DEVICE)

    res = evaluate(model, test_loader, stft_cpu,
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