# ------------------------ utf-8* encoding --------------------
# librosa based feature extraction
import os
import sys
import torch
import torchaudio
import librosa
import numpy as np
from typing import Union


def zero_cross_rate_feature(
    audio: Union[np.ndarray, torch.Tensor], fr_rate=2048, hop_length=512, center=True
):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    zcr_val = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=fr_rate, hop_length=hop_length, center=center
    )
    return zcr_val


def rms_feature(
    audio: Union[np.ndarray, torch.Tensor], fr_rate=2048, hop_length=512, center=True
):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    rms = librosa.feature.rms(
        y=audio, frame_length=fr_rate, hop_length=hop_length, center=center
    )


def chromagram_feature(
    audio: Union[np.ndarray, torch.Tensor],
    sr,
    hop_length=512,
    win_length=512,
    n_chroma=12,
    nfft=2048,
    window="hann",
    center=True,
):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    chm_f = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
        n_chroma=n_chroma,
        n_fft=nfft,
        window=window,
        center=center,
    )
    return chm_f
