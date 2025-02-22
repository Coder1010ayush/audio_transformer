import librosa
import numpy as np
import scipy
import torch


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
