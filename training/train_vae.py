#!/usr/bin/env python3
# denoise/training/train_vae.py
"""
Training script for AutoencoderKL (VAE) speech denoiser with optional Weights & Biases
logging or stdout progress bar.

Usage example:
  python -m denoise.training.train_vae \
      --root /data/VoiceBank-DEMAND \
      --epochs 50 \
      --exp vae_run1 \
      --use_wandb
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio as SISDR,
    ShortTimeObjectiveIntelligibility as STOI,
)
from tqdm import tqdm

# ---------- project imports ----------
from denoise.training.dataset import create_dataloaders
from denoise.variational_autoencoder.autoencoder import AutoencoderKL
from denoise.audio.stft import TacotronSTFT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
            # 打印到 stdout
            msg = " | ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
            if step is not None:
                print(f"[step {step}] {msg}")
            else:
                print(msg)


# --------------------------  main  --------------------------------- #
def main(cfg: argparse.Namespace):
    log_root = Path("runs") / cfg.exp
    log_root.mkdir(parents=True, exist_ok=True)

    logger = Logger(cfg.use_wandb, cfg.exp, cfg)

    # 数据
    stft_fn = build_stft()
    mel_params = dict(target_length=1024, fn_STFT=stft_fn)
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg.root, mel_params, cfg.batch_size
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

    # 评估指标
    sisdr_metric = SISDR().to(DEVICE)
    stoi_metric = STOI(16000, False).to(DEVICE)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, disable=cfg.use_wandb, desc=f"Epoch {epoch}")

        for noisy_mel, clean_mel in pbar:
            noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)

            # 前向
            rec_mel, posterior = model(noisy_mel)

            # 损失
            recon_loss = F.l1_loss(rec_mel, clean_mel)
            kl_loss = torch.mean(posterior.kl())
            beta = beta_anneal(global_step, total_steps)
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            # log
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
        val_loss = validate(model, val_loader)
        logger.log({"val/loss": val_loss}, step=global_step)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), log_root / "best.ckpt")
            print(f"✓ epoch {epoch}: best model saved (val_loss={best_val:.4f})")

    # ---------------- test ----------------
    model.load_state_dict(torch.load(log_root / "best.ckpt", map_location=DEVICE))
    test_sisdr, test_stoi = evaluate(model, test_loader, sisdr_metric, stoi_metric)
    logger.log({"test/SI-SDR": test_sisdr, "test/STOI": test_stoi})
    print(f"[TEST] SI-SDR={test_sisdr:.2f} dB | STOI={test_stoi:.3f}")


# ------------------ helper functions ------------------ #
@torch.no_grad()
def validate(model, loader):
    model.eval()
    total = 0.0
    for noisy_mel, clean_mel in loader:
        noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)
        rec_mel, posterior = model(noisy_mel, sample_posterior=False)
        recon = F.l1_loss(rec_mel, clean_mel, reduction="sum")
        kl = torch.sum(posterior.kl())
        total += (recon + kl).item()
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, sisdr_metric, stoi_metric):
    model.eval()
    sisdr_total, stoi_total = 0.0, 0.0
    for noisy_mel, clean_mel in tqdm(loader, desc="Testing", disable=True):
        noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)
        rec_mel, _ = model(noisy_mel, sample_posterior=False)

        denoised_wav = model.decode_to_waveform(rec_mel).astype("float32") / 32768.0
        clean_wav = model.decode_to_waveform(clean_mel).astype("float32") / 32768.0

        denoised_wav = torch.from_numpy(denoised_wav).to(DEVICE)
        clean_wav = torch.from_numpy(clean_wav).to(DEVICE)

        sisdr_total += sisdr_metric(denoised_wav, clean_wav).cpu()
        stoi_total += stoi_metric(denoised_wav, clean_wav).cpu()
    n = len(loader)
    return sisdr_total / n, stoi_total / n


# ---------------- argparse ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="VoiceBank-DEMAND root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_step", type=int, default=50)
    parser.add_argument("--exp", type=str, default="vae_denoise")
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging; if not set, prints progress bar",
    )
    cfg = parser.parse_args()
    main(cfg)