# scripts/calc_mel_stats.py
import torch
from training.dataset_new import create_dataloaders
from audio.make_mel import MelExtractor
SUBSET = '28spk'
mel_extractor = MelExtractor(mean=None, std=None)
train_loader, _, _ = create_dataloaders(
    root_dir="/vast/lb4434/datasets/voicebank-demand",
    mel_extractor=mel_extractor,
    batch_size=1,
    subset_type = SUBSET,
)

mel_sum = 0.0
mel_sq_sum = 0.0
mel_count = 0

for noisy_mel, clean_mel in train_loader:
    for mel in [noisy_mel, clean_mel]:
        mel = mel.squeeze(0)  # [n_mels, T]
        mel_sum += mel.sum(dim=1)
        mel_sq_sum += (mel ** 2).sum(dim=1)
        mel_count += mel.shape[1]

mean = mel_sum / mel_count
std = (mel_sq_sum / mel_count - mean ** 2).sqrt()
torch.save({'mean': mean, 'std': std}, f"mel_stats_{SUBSET}.pth")
print("mean:", mean)
print("std:", std)