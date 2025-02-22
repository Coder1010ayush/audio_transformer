# ------------------- utf-8 encoding* -----------------------
import os
import librosa
import torchaudio
import numpy as np
import torch
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


# loading audio using different-different backends [ librosa , torchaudio , AudioSegment ]
def load_audio(path: str, sr=None, backend="librosa"):

    if backend == "librosa":
        waveform, sr = librosa.load(path=path, sr=sr)
        return waveform, sr

    elif backend == "torchaudio":
        waveform, sample_rate = torchaudio.load(uri=path, normalize=False)
        if sr and sample_rate != sr:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=sr
            )(waveform)
        return waveform.numpy(), sr

    elif backend == "pydub":
        waveform = AudioSegment.from_file(file=path)
        if sr:
            waveform = waveform.set_frame_rate(sr)
        samples = np.array(waveform.get_array_of_samples()).astype(np.float32) / (2**15)
        return samples, waveform.frame_rate

    else:
        raise ValueError(
            f"Unsupported backend Choose from 'librosa' or 'torchaudio' or 'pydub'."
        )


# normalization function
def normalize(waveform, methode="peak", target_rms=0.1):

    if methode == "peak":
        if isinstance(waveform, np.ndarray):
            p_val = np.max(np.abs(waveform))
            waveform = waveform / p_val if p_val > 0 else waveform
            return waveform
        elif isinstance(waveform, torch.Tensor):
            p_val = torch.max(torch.abs(waveform))
            waveform = waveform / p_val if p_val > 0 else waveform
            return waveform
    elif methode == "rms":
        if isinstance(waveform, np.ndarray):
            p_val = np.sqrt(np.mean(np.square(waveform)))
            waveform = (waveform * target_rms) / (p_val + 1e-8)
            return waveform
        elif isinstance(waveform, torch.Tensor):
            p_val = torch.sqrt(torch.mean(torch.square(waveform)))
            waveform = (waveform * target_rms) / (p_val + 1e-8)
            return waveform
    else:
        raise ValueError(f"Unsupported methode is given {methode}")


# resampling function
def resample_audio(waveform: np.ndarray, orig_sr, target_sr, backend="librosa"):

    if orig_sr == target_sr:
        return waveform

    if backend == "librosa":
        return librosa.resample.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

    elif backend == "torchaudio":
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=target_sr
        )
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform)
        return resampler(waveform).numpy()

    else:
        raise ValueError("Unsupported backend. Choose from 'librosa' or 'torchaudio'.")


# convert from stereo to mono
def stereo_to_mono(waveform):

    if isinstance(waveform, np.ndarray):
        if waveform.ndim == 2:
            return np.mean(waveform, axis=0)
        return waveform

    elif isinstance(waveform, torch.Tensor):
        if waveform.dim() == 2:
            return waveform.mean(dim=0)
        return waveform

    else:
        raise TypeError("Unsupported type. Use numpy.ndarray or torch.Tensor.")


# trim the silence in the end or in the begining ( generally found in audio )
def trim_audio(waveform: np.ndarray, sr, top_db=20, backend="librosa"):
    if backend == "librosa":
        trimmed_waveform, _ = librosa.effects.trim(waveform, top_db=top_db)
        return trimmed_waveform

    elif backend == "pydub":
        audio = AudioSegment(
            (waveform * (2**15)).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )
        non_silent_ranges = detect_nonsilent(audio, silence_thresh=-top_db)
        if not non_silent_ranges:
            return waveform
        start, end = non_silent_ranges[0][0], non_silent_ranges[-1][1]
        return waveform[start:end]

    else:
        raise ValueError("Unsupported backend. Choose 'librosa' or 'pydub'.")
