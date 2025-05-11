#!/usr/bin/env python3
# training/train_vae.py
"""
AutoencoderKL 训练脚本（DiffWave 22 kHz 版本）

示例：
  python train_vae.py \
      --root /vast/lb4434/datasets/voicebank-demand \
      --epochs 5 \
      --exp vae_run80mel_t \
      --subset 28spk \
      --beta 1e-3 \
      --save_dir ./samples/vae_run80me_t \
      --use_wandb
"""

from __future__ import annotations
import argparse, os, random
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

# ---------- project imports ----------
from audio.stft                       import TacotronSTFT
from training.dataset                 import create_dataloaders
from variational_autoencoder.autoencoder import AutoencoderKL
from test_noGAN                        import evaluate        # ✅ 复用评测逻辑

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

# ---------------- models & config ---------------- #
def build_stft() -> TacotronSTFT:
    """22 050 Hz / 80-bin / hop 256"""
    return TacotronSTFT()          # 全部默认参数已在 stft.py 中修改

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

# ---------------- logging ---------------- #
class Logger:
    """统一 WandB / print"""

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

# ---------------- helper ---------------- #
@torch.no_grad()
def validate(model: AutoencoderKL, loader):
    model.eval()
    recon_tot = kl_tot = 0.0
    for noisy_mel, clean_mel in loader:
        noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)
        rec_mel, posterior  = model(noisy_mel, sample_posterior=False)
        recon_tot += F.l1_loss(rec_mel, clean_mel).item()
        kl_tot    += torch.mean(posterior.kl()).item()
    n = len(loader)
    return recon_tot / n, kl_tot / n

# ---------------- train ---------------- #
def main(cfg: argparse.Namespace):
    run_dir = Path("runs")/cfg.exp
    run_dir.mkdir(parents=True, exist_ok=True)

    sample_dir: Optional[Path] = None
    if cfg.save_dir:
        sample_dir = Path(cfg.save_dir); sample_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(cfg.use_wandb, cfg.exp, cfg)

    # data
    stft_fn = build_stft()
    mel_params = dict(target_length=512, fn_STFT=stft_fn)
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg.root, mel_params, batch_size=cfg.batch_size, subset_type=cfg.subset
    )

    # model
    model = AutoencoderKL(ddconfig=ddconfig(), embed_dim=32,
                          image_key="fbank", time_shuffle=1).to(DEVICE)

    # optim
    optim = Adam(model.parameters(), lr=cfg.lr, betas=(0.9,0.999))
    sched = CosineAnnealingLR(optim, T_max=len(train_loader)*cfg.epochs)

    total_steps = len(train_loader)*cfg.epochs
    global_step = 0
    best_val = float("inf"); patience = 0

    for epoch in range(1, cfg.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for noisy_mel, clean_mel in pbar:
            noisy_mel, clean_mel = noisy_mel.to(DEVICE), clean_mel.to(DEVICE)

            rec_mel, posterior = model(noisy_mel)
            recon_l1 = F.l1_loss(rec_mel, clean_mel)
            kl       = torch.mean(posterior.kl())
            beta     = beta_anneal(global_step, total_steps, cfg.beta)
            loss     = recon_l1 + beta*kl

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step(); sched.step()

            if global_step % cfg.log_step == 0:
                logger.log({"train/loss":loss.item(),
                            "train/recon":recon_l1.item(),
                            "train/kl":kl.item(),
                            "train/beta":beta,
                            "lr":optim.param_groups[0]["lr"]},
                           step=global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        # ------ validation ------
        val_recon,val_kl = validate(model, val_loader)
        val_loss = val_recon + val_kl
        logger.log({"val/recon":val_recon,"val/kl":val_kl,
                    "val/loss":val_loss}, step=global_step)

        if val_loss < best_val:
            best_val = val_loss; patience=0
            torch.save(model.state_dict(), run_dir/"best.ckpt")
            print(f"✓ epoch {epoch}: best model saved (val_loss={best_val:.4f})")
        else:
            patience += 1
        if patience==20:
            print("Early stop (no improve 20 epochs)")
            break

    # -------------- TEST --------------
    model.load_state_dict(torch.load(run_dir/"best.ckpt", map_location=DEVICE))

    sisdr_m = SISDR().to(DEVICE)
    stoi_m  = STOI(22050, False).to(DEVICE)
    pesq_m  = PESQ(16000, "wb").to(DEVICE)

    test_metrics = evaluate(model, test_loader,
                            sisdr_m, stoi_m, pesq_m,
                            sample_dir)

    logger.log({f"test/{k}":v for k,v in test_metrics.items()})
    print("TEST metrics:", ", ".join(f"{k}:{v:.3f}" for k,v in test_metrics.items()))

# ---------------- argparse ---------------- #
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", required=True, help="VoiceBank-DEMAND root")
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--batch_size", type=int, default=8)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--log_step", type=int, default=50)
    pa.add_argument("--exp", type=str, default="vae_denoise")
    pa.add_argument("--subset", type=str, default="28spk")
    pa.add_argument("--beta", type=float, default=1e-3)
    pa.add_argument("--use_wandb", action="store_true")
    pa.add_argument("--save_dir", type=str, default="")
    cfg = pa.parse_args()
    main(cfg)