import torch
from torch import nn
from variational_autoencoder.modules import Encoder, Decoder
from variational_autoencoder.distributions import DiagonalGaussianDistribution
from vocoder.diffwave_vocoder import mel2wav_diffwave
from audio.audio_processing import dynamic_range_decompression     # ★ 新增

class AutoencoderKL(nn.Module):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        image_key="fbank",
        embed_dim=None,
        time_shuffle=1,
        subband=1,
        ckpt_path=None,
        reload_from_ckpt=None,
        ignore_keys=[],
        colorize_nlabels=None,
        monitor=None,
        base_learning_rate=1e-5,
    ):
        super().__init__()

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.embed_dim = embed_dim

        if monitor is not None:
            self.monitor = monitor

        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

    # ---------------- Encoder / Decoder ---------------- #
    def encode(self, x):
        x = self.freq_split_subband(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = self.freq_merge_subband(dec)
        return dec

    # ---------------- Waveform helper ------------------ #
    @torch.no_grad()
    def decode_to_waveform(self, dec):
        """
        dec: (B,1,80,T) — 模型输出，已做 ln 压缩
        返回值: numpy int16, (B,N)
        """
        # ln → 线性 → log10
        mel_lin  = dynamic_range_decompression(dec.squeeze(1))      # (B,T,80)
        mel_log10 = torch.log10(mel_lin.clamp(min=1e-5)).unsqueeze(1)  # (B,1,80,T)
        wav = mel2wav_diffwave(mel_log10)      # (B,N) float32 ±1
        return (wav.cpu().numpy() * 32767).astype("int16")

    # ---------------- Forward -------------------------- #
    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    # ---------------- 频带拆分/合并 ---------------------- #
    def freq_split_subband(self, fbank):
        if self.subband == 1 or self.image_key != "stft":
            return fbank
        bs, ch, tstep, fbins = fbank.size()
        return (
            fbank.squeeze(1)
            .reshape(bs, tstep, self.subband, fbins // self.subband)
            .permute(0, 2, 1, 3)
        )

    def freq_merge_subband(self, subband_fbank):
        if self.subband == 1 or self.image_key != "stft":
            return subband_fbank
        bs, sub_ch, tstep, fbins = subband_fbank.size()
        return subband_fbank.permute(0, 2, 1, 3).reshape(bs, tstep, -1).unsqueeze(1)