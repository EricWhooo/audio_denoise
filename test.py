#test.py
"""
Usage example:
  python test.py \
      --root /vast/lb4434/datasets/voicebank-demand \
      --exp vae_run4_28_beta1 \
      --subset 28spk \
      --save_dir ./samples/vae_run4_28_beta1
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio as SISDR,
    ShortTimeObjectiveIntelligibility as STOI,
    PerceptualEvaluationSpeechQuality as PESQ,
)
from tqdm import tqdm
import soundfile as sf
import numpy as np

from speechmetrics import load as _load_speechmetrics

speechmetrics_fn = _load_speechmetrics(["csig", "cbak", "covl"], window="full")

# ---------- project imports ----------
from training.dataset import create_dataloaders
from variational_autoencoder.autoencoder import AutoencoderKL
from audio.stft import TacotronSTFT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

def build_stft() -> TacotronSTFT:
    return TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=64,
        sampling_rate=16000,
        mel_fmin=0,
        mel_fmax=8000,
    )

def ddconfig() -> Dict:
    return dict(
        ch=128,
        out_ch=1,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=1,
        resolution=1024,
        z_channels=32,
    )

@torch.no_grad()
def evaluate(
    model,
    loader,
    sisdr_metric,
    stoi_metric,
    pesq_metric,
    save_dir: Optional[Path] = None,
    n_examples: int = 3,
):
    model.eval()
    sisdr_total = stoi_total = pesq_total = 0.0
    csig_total = cbak_total = covl_total = 0.0
    # 随机挑选要保存的 batch 索引
    save_indices = set(random.sample(range(len(loader)), min(n_examples, len(loader))))
    saved_cnt = 0

    def float_to_int16(wav: np.ndarray) -> np.ndarray:
        wav = np.clip(wav, -1.0, 1.0)
        return (wav * 32767).astype(np.int16)

    for batch_idx, (noisy_mel, clean_mel) in enumerate(tqdm(loader, desc="Testing", disable=True)):
        noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)

        rec_mel, _ = model(noisy_mel, sample_posterior=False)

        # ----------- 波形 -----------
        den_wav = model.decode_to_waveform(rec_mel).astype("float32")
        clean_wav = model.decode_to_waveform(clean_mel).astype("float32")
        noisy_wav = model.decode_to_waveform(noisy_mel).astype("float32")
        #print("den_wav  max, min, mean:", den_wav.max(), den_wav.min(), den_wav.mean())

        den_wav_t = torch.from_numpy(den_wav).to(DEVICE)
        clean_wav_t = torch.from_numpy(clean_wav).to(DEVICE)

        # ----------- 指标 -----------
        sisdr_total += sisdr_metric(den_wav_t, clean_wav_t).cpu()
        stoi_total += stoi_metric(den_wav_t, clean_wav_t).cpu()
        #print("den_wav  max, min, mean:", den_wav_t.max(), den_wav_t.min(), den_wav_t.mean())
        #pesq_total += pesq_metric(den_wav_t, clean_wav_t).cpu()

        
        comp = speechmetrics_fn(clean_wav, den_wav, 16000)
        csig_total += comp.get("csig", 0)
        cbak_total += comp.get("cbak", 0)
        covl_total += comp.get("covl", 0)

        # ----------- 保存样本 -----------
        if save_dir is not None and batch_idx in save_indices:
            sf.write(save_dir / f"sample{saved_cnt}_noisy.wav", noisy_wav[0], 16000)
            sf.write(save_dir / f"sample{saved_cnt}_denoised.wav", den_wav[0], 16000)
            sf.write(save_dir / f"sample{saved_cnt}_clean.wav", clean_wav[0], 16000)
            saved_cnt += 1

    n = len(loader)
    return (
        sisdr_total / n,
        stoi_total / n,
        pesq_total / n,
        csig_total / n,
        cbak_total / n,
        covl_total / n,
    )

def main(cfg: argparse.Namespace):
    log_root = Path("runs") / cfg.exp
    sample_save_dir: Optional[Path] = None
    if cfg.save_dir:
        sample_save_dir = Path(cfg.save_dir)
        sample_save_dir.mkdir(parents=True, exist_ok=True)
    
    stft_fn = build_stft()
    mel_params = dict(target_length=512, fn_STFT=stft_fn) #1024 -> 10 sec
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=cfg.root, mel_params=mel_params, batch_size=cfg.batch_size, subset_type=cfg.subset
    )

    model = AutoencoderKL(
            ddconfig=ddconfig(),
            embed_dim=32,
            image_key="fbank",
            time_shuffle=1,
        ).to(DEVICE)

    sisdr_metric = SISDR().to(DEVICE)
    stoi_metric = STOI(16000, False).to(DEVICE)
    pesq_metric = PESQ(16000, "wb").to(DEVICE)

    state_dict = torch.load(log_root / "best.ckpt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    (
        test_sisdr,
        test_stoi,
        test_pesq,
        test_csig,
        test_cbak,
        test_covl,
    ) = evaluate(model, test_loader, sisdr_metric, stoi_metric, pesq_metric, sample_save_dir)

    print(
        f"[TEST] SI-SDR={test_sisdr:.2f} dB | STOI={test_stoi:.3f} | "
        f"PESQ={test_pesq:.3f} | CSIG={test_csig:.3f} | "
        f"CBAK={test_cbak:.3f} | COVL={test_covl:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="VoiceBank-DEMAND root")
    parser.add_argument("--subset", type=str, default="28spk")
    parser.add_argument("--exp", type=str, default="vae_denoise")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
    )
    cfg = parser.parse_args()
    main(cfg)