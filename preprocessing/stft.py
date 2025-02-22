from typing import Union
import librosa
import numpy as np
import scipy
import torch
import torchaudio
from torchaudio.transforms import Spectrogram, InverseSpectrogram, MelSpectrogram, MFCC


# stft
def compute_stft(
    waveform, sr, backend="torchaudio", n_fft=1024, hop_length=512, win_length=None
):
    if backend == "librosa":
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        stft_result = librosa.stft(
            waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        magnitude = np.abs(stft_result)

    elif backend == "torchaudio":
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform)
        stft_result = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length if win_length else n_fft,
            return_complex=True,
        )
        magnitude = torch.abs(stft_result)

    elif backend == "numpy":
        f, t, stft_result = scipy.signal.stft(
            waveform, fs=sr, nperseg=n_fft, noverlap=hop_length
        )
        magnitude = np.abs(stft_result)

    else:
        raise ValueError(
            "Unsupported backend. Choose from 'librosa', 'torchaudio', or 'numpy'."
        )

    return stft_result, magnitude


# inverse stft
def compute_istft(
    stft_result, sr, backend="librosa", n_fft=1024, hop_length=512, win_length=None
):
    if backend == "librosa":
        if isinstance(stft_result, torch.Tensor):
            stft_result = stft_result.numpy()
        waveform = librosa.istft(
            stft_result, hop_length=hop_length, win_length=win_length
        )

    elif backend == "torchaudio":
        if isinstance(stft_result, np.ndarray):
            stft_result = torch.tensor(stft_result, dtype=torch.complex64)
        waveform = torch.istft(
            stft_result,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length if win_length else n_fft,
        )

    elif backend == "numpy":
        _, waveform = scipy.signal.istft(
            stft_result, fs=sr, nperseg=n_fft, noverlap=hop_length
        )

    else:
        raise ValueError(
            "Unsupported backend. Choose from 'librosa', 'torchaudio', or 'numpy'."
        )

    return waveform


def to_tensor(audio, device="cpu"):
    if isinstance(audio, np.ndarray):
        return torch.tensor(audio, dtype=torch.float32).to(device)
    return audio.to(device)


def compute_mel_spectrogram(
    audio, sr=22050, n_fft=2048, hop_length=512, backend="librosa", device="cpu"
):
    """Compute Mel-Spectrogram."""
    audio = to_tensor(audio, device)
    if backend == "librosa":
        return librosa.feature.melspectrogram(
            y=audio.cpu().numpy(), sr=sr, n_fft=n_fft, hop_length=hop_length
        )
    else:
        mel_spec = MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length
        ).to(device)
        return mel_spec(audio)


def compute_mfcc(audio, sr=22050, n_mfcc=13, backend="librosa", device="cpu"):
    """Compute MFCC features."""
    audio = to_tensor(audio, device)
    if backend == "librosa":
        return librosa.feature.mfcc(y=audio.cpu().numpy(), sr=sr, n_mfcc=n_mfcc)
    else:
        mfcc_transform = MFCC(sample_rate=sr, n_mfcc=n_mfcc).to(device)
        return mfcc_transform(audio)


def spectral_features(audio, sr=22050, backend="librosa", device="cpu"):
    """Compute spectral centroid, bandwidth, and rolloff."""
    audio = to_tensor(audio, device)
    if backend == "librosa":
        spec_centroid = librosa.feature.spectral_centroid(y=audio.cpu().numpy(), sr=sr)
        spec_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio.cpu().numpy(), sr=sr
        )
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio.cpu().numpy(), sr=sr)
    else:
        spec_centroid = torchaudio.functional.spectral_centroid(audio, sample_rate=sr)
        spec_bandwidth = torchaudio.functional.spectral_bandwidth(audio, sample_rate=sr)
        spec_rolloff = torchaudio.functional.spectral_rolloff(audio, sample_rate=sr)
    return spec_centroid, spec_bandwidth, spec_rolloff


def compute_spectrogram(
    waveform: Union[torch.Tensor, np.ndarray],
    sr=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    power=2.0,
    backend="torchaudio",
):
    if backend == "torchaudio":
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        spectrogram_transform = Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=power
        )
        spectrogram = spectrogram_transform(waveform)
        return spectrogram.squeeze(0).numpy()
    elif backend == "librosa":
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        stft = librosa.stft(
            waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        if power is None:
            spectrogram = np.abs(stft)
        else:
            spectrogram = np.abs(stft) ** power
        return spectrogram
    else:
        raise ValueError("Invalid backend. Choose 'torchaudio' or 'librosa'.")
