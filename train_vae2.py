#!/usr/bin/env python3
# training/train_vae2.py
"""
python train_vae2.py \
    --root /vast/lb4434/datasets/voicebank-demand \
    --epochs 1 \
    --batch_size 16 \
    --lr 1e-4 \
    --exp vae_rundiff_t \
    --subset 28spk \
    --beta 1e-3 \
    --save_dir ./samples/vae_rundiff_t --mel_stats mel_stats_28spk.pth
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio as SISDR,
    ShortTimeObjectiveIntelligibility   as STOI,
    PerceptualEvaluationSpeechQuality   as PESQ,
)
from tqdm import tqdm
import soundfile as sf

from audio.make_mel import MelExtractor
from training.dataset_new import create_dataloaders
from variational_autoencoder.autoencoder import AutoencoderKL
from test_vae2 import evaluate        # 复用评测逻辑

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

# ---------- 模型配置 ----------
def ddconfig() -> Dict:
    return dict(
        ch=128, out_ch=1, ch_mult=(1, 2, 4),
        num_res_blocks=2, attn_resolutions=[],
        dropout=0.0, resamp_with_conv=True,
        in_channels=1, resolution=1024,
        z_channels=32,
    )

def beta_anneal(step: int, total_steps: int, beta_max: float = 1e-3) -> float:
    return beta_max * min(1.0, step / (0.3 * total_steps))

# ---------- 简易日志 ----------
class Logger:
    def __init__(self, use_wandb: bool, exp_name: str, cfg: argparse.Namespace):
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(project="vae_speech_denoise", name=exp_name, config=vars(cfg))
            self.backend = wandb
        else:
            self.backend = None

    def log(self, metrics: dict, step: Optional[int] = None):
        if self.use_wandb:
            self.backend.log(metrics, step=step)
        else:
            msg = " | ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
            print(f"[{step if step is not None else 'log'}] {msg}")

# ---------- 验证 ----------
@torch.no_grad()
def validate(model: AutoencoderKL, loader):
    model.eval()
    recon_tot = kl_tot = n_sample = 0

    for noisy_mel, clean_mel, mask in loader:
        noisy_mel = noisy_mel.to(DEVICE).unsqueeze(1)       # (B,1,80,T)
        clean_mel = clean_mel.to(DEVICE)                    # (B,80,T)
        mask      = mask.to(DEVICE)

        rec_mel, posterior = model(noisy_mel)               # rec_mel:(B,1,80,T)
        rec_mel   = rec_mel.squeeze(1)                      # -> (B,80,T)
        min_T     = min(rec_mel.shape[-1], clean_mel.shape[-1], mask.shape[-1])
        rec_mel   = rec_mel[..., :min_T]
        clean_mel = clean_mel[..., :min_T]
        mask      = mask[..., :min_T]

        loss_map  = F.l1_loss(rec_mel, clean_mel, reduction="none")
        loss_sum  = (loss_map * mask[:, None, :]).sum()
        valid_cnt = mask.sum() * rec_mel.shape[1]           # n_valid*T
        recon_l1  = loss_sum / (valid_cnt + 1e-8)
        kl        = torch.mean(posterior.kl())

        recon_tot += recon_l1.item() * noisy_mel.size(0)
        kl_tot    += kl.item()        * noisy_mel.size(0)
        n_sample  += noisy_mel.size(0)

    return recon_tot / n_sample, kl_tot / n_sample

# ---------- 主程序 ----------
def main(cfg: argparse.Namespace):
    run_dir = Path("runs") / cfg.exp
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(cfg.save_dir) if cfg.save_dir else None
    if sample_dir: sample_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(cfg.use_wandb, cfg.exp, cfg)

    # MelExtractor
    mel_stats = torch.load(cfg.mel_stats, map_location="cpu")
    mel_extractor = MelExtractor(mean=mel_stats["mean"], std=mel_stats["std"])

    train_loader, val_loader, test_loader = create_dataloaders(
        cfg.root, mel_extractor, batch_size=cfg.batch_size, subset_type=cfg.subset
    )

    model = AutoencoderKL(ddconfig=ddconfig(), embed_dim=32,
                          image_key="fbank", time_shuffle=1).to(DEVICE)
    optim = Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    sched = CosineAnnealingLR(optim, T_max=len(train_loader) * cfg.epochs)

    total_steps = len(train_loader) * cfg.epochs
    global_step = 0
    best_val, patience = float("inf"), 0

    # ------------- 训练 -------------
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for noisy_mel, clean_mel, mask in pbar:
            noisy_mel = noisy_mel.to(DEVICE)      # (B,80,T)
            clean_mel = clean_mel.to(DEVICE)
            mask      = mask.to(DEVICE)

            rec_mel, posterior = model(noisy_mel.unsqueeze(1))  # -> (B,1,80,T)
            rec_mel   = rec_mel.squeeze(1)                      # (B,80,T)

            min_T     = min(rec_mel.shape[-1], clean_mel.shape[-1], mask.shape[-1])
            rec_mel   = rec_mel  [..., :min_T]
            clean_mel = clean_mel[..., :min_T]
            mask      = mask     [..., :min_T]

            loss_raw  = F.l1_loss(rec_mel, clean_mel, reduction="none")
            loss_sum  = (loss_raw * mask[:, None, :]).sum()
            loss_den  = mask.sum() * rec_mel.shape[1]
            recon_l1  = loss_sum / (loss_den + 1e-8)
            kl        = torch.mean(posterior.kl())

            beta = beta_anneal(global_step, total_steps, cfg.beta)
            loss = recon_l1 + beta * kl

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step(); sched.step()

            if global_step % cfg.log_step == 0:
                logger.log({"train/loss": loss.item(),
                            "train/recon": recon_l1.item(),
                            "train/kl": kl.item(),
                            "train/beta": beta,
                            "lr": optim.param_groups[0]["lr"]},
                           step=global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        # ---------- 验证 ----------
        val_recon, val_kl = validate(model, val_loader)
        val_loss = val_recon + val_kl
        logger.log({"val/recon": val_recon, "val/kl": val_kl,
                    "val/loss": val_loss}, step=global_step)

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            torch.save(model.state_dict(), run_dir / "best.ckpt")
            print(f"✓ epoch {epoch}: best model saved (val_loss={best_val:.4f})")
        else:
            patience += 1
        if patience == 20:
            print("Early stop (no improve 20 epochs)"); break

    # ------------- TEST -------------
    model.load_state_dict(torch.load(run_dir / "best.ckpt", map_location=DEVICE))

    sisdr_m = SISDR().to(DEVICE)
    stoi_m  = STOI(22050, False).to(DEVICE)
    pesq_m  = PESQ(16000, "wb").to(DEVICE)

    test_metrics = evaluate(model, test_loader,
                            sisdr_m, stoi_m, pesq_m,
                            sample_dir, mel_extractor=mel_extractor)

    logger.log({f"test/{k}": v for k, v in test_metrics.items()})
    print("TEST metrics:", ", ".join(f"{k}:{v:.3f}" for k, v in test_metrics.items()))

# ---------- CLI ----------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", required=True)
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--batch_size", type=int, default=8)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--log_step", type=int, default=50)
    pa.add_argument("--exp", type=str, default="vae_denoise")
    pa.add_argument("--subset", type=str, default="28spk")
    pa.add_argument("--beta", type=float, default=1e-3)
    pa.add_argument("--use_wandb", action="store_true")
    pa.add_argument("--save_dir", type=str, default="")
    pa.add_argument("--mel_stats", type=str, default="mel_stats.pth",
                    help="路径: mel mean/std")
    cfg = pa.parse_args()
    main(cfg)