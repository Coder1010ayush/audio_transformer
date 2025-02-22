# ---------------------------- utf-8 encoding* ------------------------
# this file contains all the major kinds of audio augmentation technique
import os
import numpy as np
import torch
import librosa
import torchaudio
from typing import Union, List, Dict

# Time Domain technique for audio augmentation
# Noise Injection – Adding Gaussian, pink, real-world background noise.
# Time Stretching – Speed up or slow down without affecting pitch (Librosa time_stretch).
# Pitch Shifting – Shift frequency up or down (Librosa pitch_shift).
# Volume Perturbation – Random gain increase/decrease.
# Reverberation (Echo) – Apply room impulse response (RIR).
# Clipping/Distortion – Simulate microphone or speaker distortions.
# Waveform Cropping – Randomly remove or shuffle segments of audio


def gaussian_noise(audio_waveform: Union[np.ndarray, torch.Tensor], noise_level=0.005):
    if isinstance(audio_waveform, np.ndarray):
        noise = np.random.normal(loc=0, scale=noise_level, size=audio_waveform.shape)
        return audio_waveform + noise
    elif isinstance(audio_waveform, torch.Tensor):
        audio_waveform = audio_waveform.detach().numpy()
        noise = np.random.normal(loc=0, scale=noise_level, size=audio_waveform.shape)
        return audio_waveform + noise


def time_stretch(
    audio, stretch_factor=1.2, n_fft=2048, hop_length=512, use_custom=True
):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()

    if use_custom:
        stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude, phase = np.abs(stft_matrix), np.angle(stft_matrix)

        time_bins = np.arange(stft_matrix.shape[1]) / stretch_factor
        time_bins = time_bins[time_bins < stft_matrix.shape[1]]
        stretched_magnitude = np.zeros((stft_matrix.shape[0], len(time_bins)))
        stretched_phase = np.zeros((stft_matrix.shape[0], len(time_bins)))
        for i, t in enumerate(time_bins):
            left = int(np.floor(t))
            right = min(left + 1, stft_matrix.shape[1] - 1)
            alpha = t - left  # Fractional part
            stretched_magnitude[:, i] = (1 - alpha) * magnitude[
                :, left
            ] + alpha * magnitude[:, right]
            phase_diff = np.angle(stft_matrix[:, right]) - np.angle(
                stft_matrix[:, left]
            )
            phase_diff = np.unwrap(phase_diff)
            stretched_phase[:, i] = np.angle(stft_matrix[:, left]) + alpha * phase_diff

        stretched_stft = stretched_magnitude * np.exp(1j * stretched_phase)
        stretched_audio = librosa.istft(stretched_stft, hop_length=hop_length)
        return stretched_audio

    else:
        audio = librosa.effects.time_stretch(audio, rate=stretch_factor)


def pitch_shift(audio, sr, n_steps=4, n_fft=2048, hop_length=512, use_custom=True):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    if use_custom:
        stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude, phase = np.abs(stft_matrix), np.angle(stft_matrix)
        pitch_factor = 2 ** (n_steps / 12)  # Semitone to frequency scaling
        num_bins = magnitude.shape[0]
        new_bins = np.linspace(0, num_bins - 1, int(num_bins / pitch_factor))

        stretched_magnitude = np.zeros((len(new_bins), magnitude.shape[1]))
        for t in range(magnitude.shape[1]):
            stretched_magnitude[:, t] = np.interp(
                new_bins, np.arange(num_bins), magnitude[:, t]
            )
        stretched_phase = np.zeros_like(stretched_magnitude)
        for t in range(1, magnitude.shape[1]):
            phase_diff = np.angle(stft_matrix[:, t]) - np.angle(stft_matrix[:, t - 1])
            phase_diff = np.unwrap(phase_diff)
            stretched_phase[:, t] = (
                stretched_phase[:, t - 1] + phase_diff[: len(new_bins)]
            )
        stretched_stft = stretched_magnitude * np.exp(1j * stretched_phase)
        shifted_audio = librosa.istft(stretched_stft, hop_length=hop_length)
        return shifted_audio
    else:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


# Time-Frequency Domain Augmentations (Spectrogram-Based)
# SpecAugment (Google) – Randomly mask frequency and time bands.
# Time Warping – Slightly distort time axis to simulate variations.
# Frequency Masking – Block random frequency bands (useful for speech).
# Time Masking – Block random time sections to prevent overfitting.
# Mixup & Cutmix – Blend two spectrograms to create hybrid data.


# Environmental and Contextual Augmentations
# Room Impulse Response (RIR) Augmentation – Simulate real-world room acoustics.
# Background Noise Injection – Add street, office, or nature sounds for robustness.
# Speed Perturbation – Create natural-sounding variations of speech/music.
