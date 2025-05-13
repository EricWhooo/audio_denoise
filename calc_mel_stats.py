# scripts/calc_mel_stats.py
import torch
from training.dataset_new import create_dataloaders
from audio.make_mel import MelExtractor

SUBSET = "28spk"

# ---------------- 1. dataloader ---------------- #
mel_extractor = MelExtractor(mean=None, std=None)
train_loader, _, _ = create_dataloaders(
    root_dir="/vast/lb4434/datasets/voicebank-demand",
    mel_extractor=mel_extractor,
    batch_size=1,
    subset_type=SUBSET,
)

# ---------------- 2. 累加求均值/方差 ---------------- #
mel_sum     = torch.zeros(80)
mel_sq_sum  = torch.zeros(80)
total_frames = 0

for noisy_mel, clean_mel, _ in train_loader:          # 记得接收 mask
    for mel in (noisy_mel, clean_mel):
        # mel : [B, 1, 80, T]  —— batch_size=1
        mel = mel.squeeze(0).squeeze(0)               # -> [80, T]
        mel_sum    += mel.sum(dim=1)
        mel_sq_sum += (mel ** 2).sum(dim=1)
        total_frames += mel.shape[1]

mean = mel_sum / total_frames
std  = (mel_sq_sum / total_frames - mean ** 2).sqrt()

torch.save({"mean": mean, "std": std}, f"mel_stats_{SUBSET}.pth")
print("mean:", mean)
print("std :", std)