# VAE-Speech-Denoiser

This repository implements a Variational Auto-Encoder (VAE) speech denoising pipeline, featuring:

- **Dataset loading** (supports train / val / test)
- **Model training** & TensorBoard monitoring
- **Post-training evaluation** (L1, KL, SI-SDR, STOI)
- **Inference script**: input a noisy wav, output the denoised result
- **Code tested on** PyTorch ≥ 1.12 / CUDA ≥ 11.3
- **GPU recommended**: memory ≥ 8 GB

---

## Directory Structure

```text
denoise/
 ├─ audio/                   # STFT, mel, and common audio utilities
 ├─ hifigan/                 # HiFi-GAN vocoder (pruned)
 ├─ variational_autoencoder/ # Encoder / Decoder / AutoencoderKL
 └─ training/
    ├─ dataset.py         # Enhanced dataset (validation supported)
    ├─ train_vae.py       # Training script
    └─ inference.py       # Inference script
```

---

## 1. Environment Setup

```bash
conda create -n vae_denoise python=3.9 -y
conda activate vae_denoise

# Required dependencies
pip install torch torchvision torchaudio
pip install torchmetrics tensorboard soundfile einops librosa scipy pystoi

# If using wandb
pip install wandb
```

---

## 2. Data Preparation

Download the **VoiceBank-DEMAND** dataset.

After extraction, ensure the directory structure:

```text
voicebank-demand/
 ├─ clean_trainset_56spk_wav/
 ├─ noisy_trainset_56spk_wav/
 ├─ clean_trainset_28spk_wav/
 ├─ noisy_trainset_28spk_wav/
 ├─ clean_testset_wav/
 └─ noisy_testset_wav/
```

---

## 3. Training

```bash
python -m denoise.training.train_vae \
  --root /path/to/VoiceBank-DEMAND \
  --epochs 50 \
  --batch_size 8 \
  --exp exp_wandb \
  --use_wandb       #optional
```

Logs and best models are saved in `runs/<exp>/`.


### Metrics:

- `train/recon_l1`: Reconstruction L1 loss
- `train/kl`: KL loss
- `val/total_loss`: Validation total loss
- `test/SI-SDR`, `test/STOI`: Test set metrics

---

## 4. Inference

```bash
python -m denoise.training.inference \
  --ckpt runs/my_experiment/best.ckpt \
  --noisy example_noisy.wav \
  --out denoised.wav
```

This script will:

1. Read and extract the mel spectrogram from `example_noisy.wav`.
2. Denoise using the trained `AutoencoderKL`.
3. Decode back to waveform using HiFi-GAN and save as `denoised.wav`.

---

## 5. Dataset

To download:
```bash
wget -O voicebank-demand.zip https://datashare.ed.ac.uk/download/DS_10283_2791.zip
```
To unzip:
```bash
unzip voicebank-demand.zip -d voicebank-demand
```

---

## 6. FAQ

| Issue                  | Solution                                                                 |
|------------------------|-------------------------------------------------------------------------|
| CUDA out of memory     | Decrease `--batch_size` or lower the `--resolution` parameter          |
| ModuleNotFoundError    | Run scripts with `python -m`, or add project root to `PYTHONPATH`      |
| Slow training          | Use a GPU ≥ RTX 3060; try `--batch_size 4` for minimum viable training |

---

**Happy Denoising!**  