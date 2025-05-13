# test_vae2.py —— DiffWave + VAE 评测脚本
'''python test_vae2.py \
    --root /vast/lb4434/datasets/voicebank-demand \
    --exp vae_rundiff_t \
    --subset 28spk \
    --save_dir ./samples/vae_rundiff_t --mel_stats mel_stats_28spk.pth'''
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio as SISDR,
    ShortTimeObjectiveIntelligibility   as STOI,
    PerceptualEvaluationSpeechQuality   as PESQ,
)
from pysepm import composite as sepm_composite

from training.dataset_new import create_dataloaders
from variational_autoencoder.autoencoder import AutoencoderKL
from audio.make_mel import MelExtractor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", DEVICE)

# -------- DiffWave --------
from speechbrain.inference.vocoders import DiffWaveVocoder
diffwave = DiffWaveVocoder.from_hparams(
    source="speechbrain/tts-diffwave-ljspeech",
    savedir="pretrained_models/tts-diffwave-ljspeech"
)

import torchaudio


def resample_22k_to_16k(wav_tensor: torch.Tensor) -> torch.Tensor:
    return torchaudio.functional.resample(wav_tensor, 22050, 16000)


@torch.no_grad()
def mel_to_waveform(mel: torch.Tensor, mel_extractor: MelExtractor) -> np.ndarray:
    """接受多种形状的梅尔，输出 16-bit PCM numpy (B,N)"""
    # 1) 去掉多余 1-维
    while mel.ndim > 4 and mel.shape[1] == 1:
        mel = mel.squeeze(1)

    # 2) 统一到 (B,80,T)
    if mel.ndim == 4:           # (B,1,80,T)
        mel = mel.squeeze(1)
    elif mel.ndim == 3:
        if mel.shape[1] != mel_extractor.n_mels and mel.shape[2] == mel_extractor.n_mels:
            mel = mel.transpose(1, 2)
    else:
        raise RuntimeError(f"Unexpected mel shape: {mel.shape}")

    # 3) 反归一化并合成
    mel_denorm = mel_extractor.denormalize(mel)  # (B,80,T)
    #mel_denorm = mel_denorm.unsqueeze(1)         # (B,1,80,T)
    wav = diffwave.decode_batch(
        mel_denorm,
        hop_len=256,
        fast_sampling=True,
        fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
    )
    return (wav.cpu().numpy() * 32767).astype(np.int16)


@torch.no_grad()
def evaluate(
    model,
    loader,
    sisdr_m,
    stoi_m,
    pesq_m,
    save_dir: Path = None,
    n_examples=3,
    mel_extractor: MelExtractor = None,
):
    sums = {k: 0.0 for k in ("sisdr", "stoi", "pesq", "csig", "cbak", "covl")}
    sample_ids = set(random.sample(range(len(loader)), min(n_examples, len(loader))))
    pbar = tqdm(loader, desc="Testing")

    for idx, (noisy_mel, clean_mel, mask) in enumerate(pbar):
        # ---------- 推理 ----------
        den_mel, _ = model(noisy_mel.unsqueeze(1).to(DEVICE), sample_posterior=False)

        valid_T   = int(mask[0].sum().item())
        den_mel   = den_mel  [..., :valid_T ]
        clean_mel = clean_mel[..., :valid_T ]
        noisy_mel = noisy_mel[..., :valid_T ]

        den_wav   = mel_to_waveform(den_mel,   mel_extractor) / 32768.0
        clean_wav = mel_to_waveform(clean_mel.unsqueeze(1), mel_extractor) / 32768.0
        noisy_wav = mel_to_waveform(noisy_mel.unsqueeze(1), mel_extractor) / 32768.0

        # ---------- 对齐长度 ----------
        min_len22 = min(den_wav.shape[-1], clean_wav.shape[-1])
        den_wav   = den_wav  [ ..., :min_len22 ]
        clean_wav = clean_wav[ ..., :min_len22 ]

        den_t, clean_t = map(lambda x: torch.from_numpy(x).to(DEVICE), (den_wav, clean_wav))

        # ---------- 16 kHz 版本也对齐 ----------
        den_16k   = resample_22k_to_16k(den_t)
        clean_16k = resample_22k_to_16k(clean_t)
        min_len16 = min(den_16k.shape[-1], clean_16k.shape[-1])
        den_16k, clean_16k = den_16k[..., :min_len16], clean_16k[..., :min_len16]

        # ---------- 评测 ----------
        sums["sisdr"] += sisdr_m(den_t, clean_t).cpu()
        sums["stoi"]  += stoi_m (den_t, clean_t).cpu()
        sums["pesq"]  += pesq_m(den_16k, clean_16k).cpu()

        csig, cbak, covl = sepm_composite(
            clean_16k[0].cpu().numpy().astype("float32"),
            den_16k  [0].cpu().numpy().astype("float32"),
            fs=16000,
        )
        sums["csig"] += csig; sums["cbak"] += cbak; sums["covl"] += covl

        if (save_dir and idx in sample_ids) or idx == 1:
            sf.write(save_dir / f"sample{idx}_noisy.wav",   noisy_wav[0][:min_len22]*32768, 22050)
            sf.write(save_dir / f"sample{idx}_denoised.wav", den_wav[0]*32768, 22050)
            sf.write(save_dir / f"sample{idx}_clean.wav",   clean_wav[0]*32768, 22050)

    for k in sums: sums[k] /= len(loader)
    return sums


def main(cfg):
    out_dir = Path(cfg.save_dir) if cfg.save_dir else None
    if out_dir: out_dir.mkdir(parents=True, exist_ok=True)

    mel_stats = torch.load(cfg.mel_stats, map_location="cpu")
    mel_extractor = MelExtractor(mean=mel_stats["mean"], std=mel_stats["std"])

    _, _, test_loader = create_dataloaders(
        cfg.root, mel_extractor, batch_size=1, subset_type=cfg.subset
    )

    model = AutoencoderKL(ddconfig={
            "ch":128, "out_ch":1, "ch_mult":(1,2,4),
            "num_res_blocks":2, "attn_resolutions":[],
            "dropout":0.0, "resamp_with_conv":True,
            "in_channels":1, "resolution":1024,
            "z_channels":32
        }, embed_dim=32,
        image_key="fbank", time_shuffle=1).to(DEVICE)
    model.load_state_dict(torch.load(Path("runs")/cfg.exp/"best.ckpt",
                                      map_location=DEVICE))

    sisdr = SISDR().to(DEVICE)
    stoi  = STOI(22050, False).to(DEVICE)
    pesq  = PESQ(16000, "wb").to(DEVICE)

    res = evaluate(model, test_loader,
                   sisdr, stoi, pesq, out_dir, mel_extractor=mel_extractor)

    print(f"[TEST] SI-SDR={res['sisdr']:.2f} dB | STOI={res['stoi']:.3f} | "
          f"PESQ={res['pesq']:.3f} | CSIG={res['csig']:.3f} | "
          f"CBAK={res['cbak']:.3f} | COVL={res['covl']:.3f}")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", required=True)
    pa.add_argument("--subset", default="28spk")
    pa.add_argument("--exp",    default="vae_denoise")
    pa.add_argument("--save_dir", default="")
    pa.add_argument("--mel_stats", default="mel_stats.pth")
    main(pa.parse_args())