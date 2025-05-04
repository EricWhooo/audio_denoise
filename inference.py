#!/usr/bin/env python3
"""
Usage:
    python -m denoise.training.inference \
        --ckpt runs/vae_denoise/best.ckpt \
        --noisy noisy.wav \
        --out denoised.wav
"""
import argparse
import soundfile as sf
import torch

from denoise.variational_autoencoder.autoencoder import AutoencoderKL
from denoise.audio.stft import TacotronSTFT
from denoise.audio.tools import wav_to_fbank

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ddconfig():
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
def main(args):
    # ---- build model -------------------------------------------------
    model = AutoencoderKL(
        ddconfig=ddconfig(),
        embed_dim=32,
        image_key="fbank",
    )
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model = model.to(DEVICE).eval()

    # ---- extract fbank from noisy wav --------------------------------
    stft_fn = TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=64,
        sampling_rate=16000,
        mel_fmin=0,
        mel_fmax=8000,
    )
    fbank, _, _ = wav_to_fbank(args.noisy, target_length=1024, fn_STFT=stft_fn)
    fbank = fbank.unsqueeze(0).to(DEVICE)  # (1, T, n_mel)
    fbank = fbank.unsqueeze(0)  # add channel => (B, 1, T, n_mel)

    # ---- forward -----------------------------------------------------
    rec_mel, _ = model(fbank, sample_posterior=False)
    denoised_wav = model.decode_to_waveform(rec_mel).squeeze() / 32768.0  # [-1,1]

    # ---- save --------------------------------------------------------
    sf.write(args.out, denoised_wav.cpu().numpy(), samplerate=16000)
    print(f"✓ Denoised audio saved to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to trained *.ckpt file")
    ap.add_argument("--noisy", required=True, help="Noisy wav file (16 kHz mono)")
    ap.add_argument("--out", required=True, help="Output wav path")
    main(ap.parse_args())