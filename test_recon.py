from audio.tools import wav_to_fbank
from audio.stft import TacotronSTFT
from hifigan.utilities import get_vocoder, vocoder_infer
import soundfile as sf

AUDIO = "/vast/lb4434/datasets/voicebank-demand/clean_testset_wav/p232_001.wav"

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

stft_fn = build_stft()
mel_params = dict(target_length=512, fn_STFT=stft_fn)
fbank, log_magnitudes_stft, waveform = wav_to_fbank(AUDIO,**mel_params)
vocoder = get_vocoder(None, "cpu")
wav_recon = vocoder_infer(fbank,vocoder)
sf.write("wavform.wav", waveform, 16000)
sf.write("wav_recon.wav", wav_recon, 16000)