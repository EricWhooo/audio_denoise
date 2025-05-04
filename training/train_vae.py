#!/usr/bin/env python3
"""
Training script for AutoencoderKL (VAE) speech denoiser.

运行方式（示例）:
    python -m denoise.training.train_vae \
        --root /path/to/VoiceBank-DEMAND \
        --epochs 50 \
        --exp exp1

❗ 请确认已安装 torchmetrics==0.11 以便计算 SI-SDR 与 STOI
"""

import argparse
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI

from denoise.training.dataset import create_dataloaders
from denoise.variational_autoencoder.autoencoder import AutoencoderKL
from denoise.variational_autoencoder.distributions import DiagonalGaussianDistribution
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
    """配置与 modules.Encoder / Decoder 对应"""
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


def beta_anneal(step, total_steps, beta_max=0.001):
    """线性 β 退火"""
    return beta_max * min(1.0, step / (0.3 * total_steps))


def main(cfg):
    log_root = Path("runs") / cfg.exp
    log_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_root.as_posix())

    # ---------------- Dataset ----------------------------
    stft_fn = build_stft()
    mel_params = dict(target_length=1024, fn_STFT=stft_fn)
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg.root, mel_params, cfg.batch_size
    )

    # ---------------- Model ------------------------------
    model = AutoencoderKL(
        ddconfig=ddconfig(),
        embed_dim=32,
        image_key="fbank",
        time_shuffle=1,
    ).to(DEVICE)
    print(model)
    print("Total parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    # metrics --------------------------------------------
    sisdr = SISDR().to(DEVICE)
    stoi = STOI(16000, False).to(DEVICE)

    # ---------------- Optim & Sched ----------------------
    optimizer = Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * cfg.epochs)

    global_step = 0
    best_val_loss = float("inf")

    # ---------------- Training Loop ----------------------
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for noisy_mel, clean_mel in train_loader:
            noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)

            rec, posterior = model(noisy_mel)

            # 1) reconstruction loss (L1)
            recon_loss = F.l1_loss(rec, clean_mel)

            # 2) KL loss
            kl_loss = torch.mean(posterior.kl())

            beta = beta_anneal(global_step, cfg.epochs * len(train_loader))
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            if global_step % cfg.log_step == 0:
                writer.add_scalar("train/recon_l1", recon_loss, global_step)
                writer.add_scalar("train/kl", kl_loss, global_step)
                writer.add_scalar("train/beta", beta, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        # ------------ Validation ------------------------
        if val_loader is not None:
            val_loss = validate(model, val_loader, epoch, writer)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = log_root / "best.ckpt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"✓ Saved best model to {ckpt_path}")

    # ---------------- Testing ---------------------------
    ckpt = torch.load(log_root / "best.ckpt", map_location=DEVICE)
    model.load_state_dict(ckpt)
    evaluate(model, test_loader, writer, tag="test")
    writer.close()
    print("Training complete. Logs =>", log_root)


@torch.no_grad()
def validate(model, loader, epoch, writer):
    model.eval()
    total_loss = 0.0
    for noisy_mel, clean_mel in loader:
        noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)
        rec, posterior = model(noisy_mel, sample_posterior=False)
        recon_l1 = F.l1_loss(rec, clean_mel, reduction="sum")
        kl = torch.sum(posterior.kl())
        total_loss += recon_l1 + kl
    total_loss /= len(loader.dataset)
    writer.add_scalar("val/total_loss", total_loss, epoch)
    print(f"[Val] Epoch {epoch}: loss={total_loss:.4f}")
    return total_loss.item()


@torch.no_grad()
def evaluate(model, loader, writer, tag="test"):
    model.eval()
    sisdr = SISDR().to(DEVICE)
    stoi = STOI(16000, False).to(DEVICE)
    total_sisdr, total_stoi = 0.0, 0.0
    for noisy_mel, clean_mel in loader:
        noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)
        rec_mel, _ = model(noisy_mel, sample_posterior=False)

        # 转为 waveform 以计算音频级指标
        denoised_wav = model.decode_to_waveform(rec_mel).astype("float32") / 32768.0
        clean_wav = model.decode_to_waveform(clean_mel).astype("float32") / 32768.0

        denoised_wav = torch.from_numpy(denoised_wav).to(DEVICE)
        clean_wav = torch.from_numpy(clean_wav).to(DEVICE)

        total_sisdr += sisdr(denoised_wav, clean_wav).cpu()
        total_stoi += stoi(denoised_wav, clean_wav).cpu()

    n = len(loader)
    sisdr_score = total_sisdr / n
    stoi_score = total_stoi / n
    writer.add_scalar(f"{tag}/SI-SDR", sisdr_score)
    writer.add_scalar(f"{tag}/STOI", stoi_score)
    print(f"[{tag.upper()}] SI-SDR: {sisdr_score:.2f} dB | STOI: {stoi_score:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="VoiceBank-DEMAND root directory")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log_step", type=int, default=50)
    p.add_argument("--exp", type=str, default="vae_denoise")
    cfg = p.parse_args()
    main(cfg)