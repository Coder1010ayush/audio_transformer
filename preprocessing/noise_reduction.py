# ---------------------------- utf-8 encoding* ------------------------
# this file contains all the major kinds of noise cancelation technique
import os
import sys
import numpy as np
import librosa
import torchaudio
import torch
from typing import Union
from preprocessing.stft import compute_stft


def is_audio_empty(
    audio: Union[np.ndarray, torch.Tensor], sr, threshold_db=-60, min_duration_sec=0.1
):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    try:
        duration = librosa.get_duration(y=audio, sr=sr)
        if duration < min_duration_sec:
            return True
        rms = librosa.feature.rms(y=audio)[0]
        db = librosa.amplitude_to_db(rms, ref=np.max)
        if np.mean(db) < threshold_db:
            return True
        if np.mean(np.abs(audio)) < 0.001:
            return True
        if np.var(audio) < 0.0001:
            return True
        return False
    except Exception as e:
        return True


def is_speech_present(
    audio: Union[np.ndarray, torch.Tensor],
    sr,
    speech_threshold=0.5,
    frame_length=2048,
    hop_length=512,
):
    if isinstance(audio, torch.Tensor):
        y = audio.detach().numpy()
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        has_speech = (
            np.mean(rolloff) > sr * 0.1
            and np.mean(centroid) > sr * 0.05
            and np.max(mel_db) - np.min(mel_db) > 40
        )
        return has_speech
    except Exception as e:
        return False


def spectral_subtraction(audio, sr, noise_frames=5, n_fft=1024, hop_length=512):
    stft = compute_stft(audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)

    noise_estimate = torch.mean(magnitude[:, :noise_frames], dim=1, keepdim=True)
    noise_reduced = magnitude - noise_estimate
    noise_reduced = torch.maximum(noise_reduced, 0.01 * noise_estimate)

    enhanced_stft = noise_reduced * torch.exp(1j * phase)
    enhanced_audio = torch.istft(enhanced_stft, n_fft=n_fft, hop_length=hop_length)
    return enhanced_audio.cpu().numpy()


def wiener_filter(audio, sr, n_fft=1024, hop_length=512):
    stft = compute_stft(audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    magnitude = torch.abs(stft)
    power = magnitude**2
    noise_power = torch.mean(power[:, :5], dim=1, keepdim=True)

    snr = torch.maximum(power / (noise_power + 1e-6), torch.tensor(1.0))
    gain = snr / (snr + 1)
    enhanced_stft = gain * stft
    enhanced_audio = torch.istft(enhanced_stft, n_fft=n_fft, hop_length=hop_length)
    return enhanced_audio.cpu().numpy()
