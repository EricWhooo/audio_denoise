#!/usr/bin/env python3
# denoise/training/train_vae.py
"""
Training script for AutoencoderKL (VAE) speech denoiser with optional Weights & Biases
logging or stdout progress bar.

Usage example:
  python train_vae.py \
      --root /vast/lb4434/datasets/voicebank-demand \
      --epochs 5 \
      --exp vae_run1 \
      --subset 28spk \
      --beta 1\
      --save_dir ./samples \
      --use_wandb
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


def beta_anneal(step: int, total_steps: int, beta_max: float = 1e-3) -> float:
    return beta_max * min(1.0, step / (0.3 * total_steps))


# -------------------------- logging helpers ------------------------ #
class Logger:
    """统一 WandB / print / tensorboard（可扩展）"""

    def __init__(self, use_wandb: bool, exp_name: str, config: argparse.Namespace):
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb

            wandb.init(project="vae_speech_denoise", name=exp_name, config=vars(config))
            self.backend = wandb
        else:
            self.backend = None

    def log(self, metrics: dict, step: Optional[int] = None):
        if self.use_wandb:
            self.backend.log(metrics, step=step)
        else:
            msg = " | ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
            if step is not None:
                print(f"[step {step}] {msg}")
            else:
                print(msg)


# --------------------------  main  --------------------------------- #
def main(cfg: argparse.Namespace):
    log_root = Path("runs") / cfg.exp
    log_root.mkdir(parents=True, exist_ok=True)

    # 保存样本的目录
    sample_save_dir: Optional[Path] = None
    if cfg.save_dir:
        sample_save_dir = Path(cfg.save_dir)
        sample_save_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(cfg.use_wandb, cfg.exp, cfg)

    # 数据
    stft_fn = build_stft()
    mel_params = dict(target_length=512, fn_STFT=stft_fn) #1024 -> 10 sec
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=cfg.root, mel_params=mel_params, batch_size=cfg.batch_size, subset_type=cfg.subset
    )

    # 模型
    model = AutoencoderKL(
        ddconfig=ddconfig(),
        embed_dim=32,
        image_key="fbank",
        time_shuffle=1,
    ).to(DEVICE)

    # 优化器 & 调度器
    optimizer = Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * cfg.epochs)

    total_steps = len(train_loader) * cfg.epochs
    global_step = 0
    best_val = float("inf")
    epoch_no_improve = 0

    # 评估指标
    sisdr_metric = SISDR().to(DEVICE)
    stoi_metric = STOI(16000, False).to(DEVICE)
    pesq_metric = PESQ(16000, "wb").to(DEVICE)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, disable=cfg.use_wandb, desc=f"Epoch {epoch}")

        for noisy_mel, clean_mel in pbar:
            noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)

            rec_mel, posterior = model(noisy_mel)

            recon_loss = F.l1_loss(rec_mel, clean_mel)
            kl_loss = torch.mean(posterior.kl())
            beta = beta_anneal(global_step, total_steps,beta_max=cfg.beta)
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            if global_step % cfg.log_step == 0:
                logger.log(
                    {
                        "train/loss": loss.item(),
                        "train/recon_l1": recon_loss.item(),
                        "train/kl": kl_loss.item(),
                        "train/beta": beta,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", recon=f"{recon_loss.item():.4f}", kl=f"{kl_loss.item():.4f}"
            )
            global_step += 1

        # ---------- validation ----------
        val_recon, val_kl = validate(model, val_loader)
        logger.log({"val/recon": val_recon, "val/kl": val_kl}, step=global_step)
        val_loss = val_recon + val_kl
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), log_root / "best.ckpt")
            epoch_no_improve = 0
            print(f"✓ epoch {epoch}: best model saved (val_loss={best_val:.4f})")
        else:
            epoch_no_improve += 1
        if epoch_no_improve == 20:
            print(f"no improve for {epoch_no_improve}epochs,early stopping!")
            break

    # ---------------- test ----------------
    model.load_state_dict(torch.load(log_root / "best.ckpt", map_location=DEVICE))
    (
        test_sisdr,
        test_stoi,
        test_pesq,
        test_csig,
        test_cbak,
        test_covl,
    ) = evaluate(model, test_loader, sisdr_metric, stoi_metric, pesq_metric, sample_save_dir)

    logger.log(
        {
            "test/SI-SDR": test_sisdr,
            "test/STOI": test_stoi,
            "test/PESQ": test_pesq,
            "test/CSIG": test_csig,
            "test/CBAK": test_cbak,
            "test/COVL": test_covl,
        }
    )
    print(
        f"[TEST] SI-SDR={test_sisdr:.2f} dB | STOI={test_stoi:.3f} | "
        f"PESQ={test_pesq:.3f} | CSIG={test_csig:.3f} | "
        f"CBAK={test_cbak:.3f} | COVL={test_covl:.3f}"
    )


# ------------------ helper functions ------------------ #
@torch.no_grad()
def validate(model, loader):
    model.eval()
    recon_total, kl_total = 0.0, 0.0
    for noisy_mel, clean_mel in loader:
        noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)
        rec_mel, posterior = model(noisy_mel, sample_posterior=False)
        recon = F.l1_loss(rec_mel, clean_mel)
        kl = torch.mean(posterior.kl())
        recon_total += recon.item()
        kl_total += kl.item()
    return recon_total / len(loader.dataset), kl_total / len(loader.dataset)


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

    for batch_idx, (noisy_mel, clean_mel) in enumerate(tqdm(loader, desc="Testing", disable=True)):
        noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)

        rec_mel, _ = model(noisy_mel, sample_posterior=False)

        # ----------- 波形 -----------
        den_wav = model.decode_to_waveform(rec_mel).astype("float32") / 32768.0
        clean_wav = model.decode_to_waveform(clean_mel).astype("float32") / 32768.0
        noisy_wav = model.decode_to_waveform(noisy_mel).astype("float32") / 32768.0

        den_wav_t = torch.from_numpy(den_wav).to(DEVICE)
        clean_wav_t = torch.from_numpy(clean_wav).to(DEVICE)

        # ----------- 指标 -----------
        sisdr_total += sisdr_metric(den_wav_t, clean_wav_t).cpu()
        stoi_total += stoi_metric(den_wav_t, clean_wav_t).cpu()
        pesq_total += pesq_metric(den_wav_t, clean_wav_t).cpu()

        if speechmetrics_fn is not None:
            comp = speechmetrics_fn(clean_wav, den_wav, fs=16000)
            csig_total += comp.get("csig", 0)
            cbak_total += comp.get("cbak", 0)
            covl_total += comp.get("covl", 0)
        else:
            csig_total += float("nan")
            cbak_total += float("nan")
            covl_total += float("nan")

        # ----------- 保存样本 -----------
        if save_dir is not None and batch_idx in save_indices:
            sf.write(save_dir / f"sample{saved_cnt}_noisy.wav", noisy_wav, 16000)
            sf.write(save_dir / f"sample{saved_cnt}_denoised.wav", den_wav, 16000)
            sf.write(save_dir / f"sample{saved_cnt}_clean.wav", clean_wav, 16000)
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


# ---------------- argparse ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="VoiceBank-DEMAND root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_step", type=int, default=50)
    parser.add_argument("--exp", type=str, default="vae_denoise")
    parser.add_argument("--subset", type=str, default="28spk")
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging; if not set, prints progress bar",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="保存测试样本（noisy/denoised/clean wav）的目录；为空则不保存",
    )
    cfg = parser.parse_args()
    main(cfg)