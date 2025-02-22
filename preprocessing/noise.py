# ---------------------------- utf-8 encoding* ------------------------
# this file contains all the major kinds of audio augmentation technique
import os
import numpy as np
import torch
import librosa
import torchaudio
from typing import Union, List, Dict
import scipy.signal
import scipy.signal as signal

# Time Domain technique for audio augmentation
# Noise Injection – Adding Gaussian, pink, real-world background noise.
# Time Stretching – Speed up or slow down without affecting pitch (Librosa time_stretch).
# Pitch Shifting – Shift frequency up or down (Librosa pitch_shift).
# Volume Perturbation – Random gain increase/decrease.
# Reverberation (Echo) – Apply room impulse response (RIR).
# Clipping/Distortion – Simulate microphone or speaker distortions.
# Waveform Cropping – Randomly remove or shuffle segments of audio


def gaussian_noise(audio: Union[np.ndarray, torch.Tensor], noise_level=0.005):
    if isinstance(audio, np.ndarray):
        noise = np.random.normal(loc=0, scale=noise_level, size=audio.shape)
        return audio + noise
    elif isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
        noise = np.random.normal(loc=0, scale=noise_level, size=audio.shape)
        return audio + noise


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


def volume_pertubation(audio: Union[np.ndarray, torch.Tensor], range_gain=(0.3, 0.7)):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    gain_factor = np.random.uniform(*range_gain)
    return audio * gain_factor


def reverberation_noise(audio: Union[np.ndarray, torch.Tensor], rir):
    audio = scipy.signal.fftconvolve(audio, rir, mode="full")[: len(audio)]
    return audio


def linear_clip_waveform(
    audio: Union[np.ndarray, torch.Tensor], methods="soft", threshold=0.3, alpha=2
):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    if methods == "soft":
        audio = audio / (1 + np.abs(audio / threshold) ** alpha)
        return audio
    elif methods == "hard":
        audio = np.clip(audio, -threshold, threshold)
        return audio
    else:
        raise ValueError(f"Unsupported methode {methods} is given")


"""  
Transfer Functions for Analog Distortion ( Studied in Digital Signal Processing )
We can apply different non-linear functions to shape the signal: 
    Hyperbolic Tangent (tanh) – Simulates soft saturation like in vacuum tube amplifiers(stackoverflow) :  x(t)=tanh(g⋅x(t))
    Arctangent (atan) – Smooth compression with softer high-end distortion: x(t) = 2/pi * arctan(g(x(t)))
    Exponential Soft Clipping – Models the gradual transition from linear to saturated : x(t) = x(t)/ (1 + alpha * abs(x(t))) 
    Dynamic Softness – Instead of a fixed clipping threshold
"""


def hyperbolic_distortion(audio: Union[np.ndarray, torch.Tensor], gain_factor=0.3):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    audio = np.tanh(gain_factor * audio)
    return audio


def arctangent_distortion(audio: Union[np.ndarray, torch.Tensor], gain_factor=0.3):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    audio = 2 / np.pi * np.arctanh(gain_factor * audio)
    return audio


def exponential_distortion(audio: Union[np.ndarray, torch.Tensor], alpha=12):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    audio = audio / (1 + alpha * np.abs(audio))
    return audio


def dynamic_softness_distortion(
    audio: Union[np.ndarray, torch.Tensor], beta=0.6, gamma=10
):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    alpha = beta * (1 - np.exp(-gamma * np.abs(audio)))
    return audio / (1 + alpha * np.abs(audio))


def low_pass_filter(audio, sr, cutoff=5000, order=4):
    nyquist = sr / 2  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, audio)


def high_pass_filter(audio, sr, cutoff=100, order=4):
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return signal.filtfilt(b, a, audio)


def dynamic_low_pass_filter(audio, sr, min_cutoff=2000, max_cutoff=8000, order=4):
    nyquist = sr / 2
    envelope = np.abs(signal.hilbert(audio))
    dynamic_cutoff = min_cutoff + (max_cutoff - min_cutoff) * (
        envelope / np.max(envelope)
    )
    filtered_audio = np.zeros_like(audio)
    for i in range(len(audio)):
        normal_cutoff = dynamic_cutoff[i] / nyquist
        b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
        filtered_audio[i] = signal.filtfilt(
            b, a, [audio[i - 1] if i > 0 else 0, audio[i]]
        )[-1]
    return filtered_audio


def dynamic_high_pass_filter(audio, sr, min_cutoff=50, max_cutoff=500, order=4):
    nyquist = sr / 2
    envelope = np.abs(signal.hilbert(audio))
    dynamic_cutoff = min_cutoff + (max_cutoff - min_cutoff) * (
        envelope / np.max(envelope)
    )
    filtered_audio = np.zeros_like(audio)
    for i in range(len(audio)):
        normal_cutoff = dynamic_cutoff[i] / nyquist
        b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
        filtered_audio[i] = signal.filtfilt(
            b, a, [audio[i - 1] if i > 0 else 0, audio[i]]
        )[-1]
    return filtered_audio


def apply_filter(
    audio: Union[np.ndarray, torch.Tensor],
    sr,
    methode="hard",
    thres=None,
    gain_factor=None,
    alpha=None,
    beta=None,
    gamma=None,
    cutoff=None,
    min_cutoff=None,
    max_cutoff=None,
    use_dynamic=False,
):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    if methode == "hard":
        if alpha and thres:
            audio = linear_clip_waveform(
                audio=audio, methods=methode, alpha=alpha, threshold=thres
            )
        audio = linear_clip_waveform(audio=audio, methods=methode)
    elif methode == "soft":
        if alpha and thres:
            audio = linear_clip_waveform(
                audio=audio, methods=methode, alpha=alpha, threshold=thres
            )
        audio = linear_clip_waveform(audio=audio, methods=methode)

    elif methode == "hyperbolic_distortion":
        if gain_factor:
            audio = hyperbolic_distortion(audio=audio, gain_factor=gain_factor)
        audio = hyperbolic_distortion(audio=audio)
    elif methode == "arctangent_distortion":
        if gain_factor:
            audio = arctangent_distortion(audio=audio, gain_factor=gain_factor)
        audio = arctangent_distortion(audio=audio)
    elif methode == "exponential_distortion":
        if alpha:
            audio = exponential_distortion(audio=audio, alpha=alpha)
        audio = exponential_distortion(audio=audio)
    elif methode == "dynamic_softness_distortion":
        if beta and gamma:
            audio = dynamic_softness_distortion(audio=audio, beta=beta, gamma=gamma)
        audio = dynamic_softness_distortion(audio=audio)
    if use_dynamic == False:
        if cutoff:
            audio = low_pass_filter(audio=audio, sr=sr, cutoff=cutoff)
            audio = high_pass_filter(audio=audio, sr=sr, cutoff=cutoff)
        audio = low_pass_filter(audio=audio, sr=sr, cutoff=cutoff)
        audio = high_pass_filter(audio=audio, sr=sr, cutoff=cutoff)
        return audio
    else:
        if min_cutoff and max_cutoff:
            audio = dynamic_low_pass_filter(
                audio=audio, sr=sr, min_cutoff=min_cutoff, max_cutoff=max_cutoff
            )
            audio = high_pass_filter(
                audio=audio, sr=sr, min_cutoff=min_cutoff, max_cutoff=max_cutoff
            )
        audio = dynamic_low_pass_filter(audio=audio, sr=sr)
        audio = high_pass_filter(audio=audio, sr=sr)
        return audio


# Time-Frequency Domain Augmentations (Spectrogram-Based)
# SpecAugment (Google) – Randomly mask frequency and time bands.
# Time Warping – Slightly distort time axis to simulate variations.
# Frequency Masking – Block random frequency bands (useful for speech).
# Time Masking – Block random time sections to prevent overfitting.


def time_warp(spec: Union[np.ndarray, torch.Tensor], warp_factor=5):
    num_freq, num_time = spec.shape
    center = num_time // 2
    warp_amount = np.random.randint(-warp_factor, warp_factor)
    if isinstance(spec, np.ndarray):
        spec = torch.tensor(spec)
    warped_spec = torch.roll(spec, shifts=warp_amount, dims=1)
    return warped_spec.numpy()


def frequency_mask(spec: Union[np.ndarray, torch.Tensor], freq_mask_param=10):
    num_freq, _ = spec.shape
    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, num_freq - f)
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().numpy()
    spec[f0 : f0 + f, :] = 0
    return spec


def time_mask(spec: Union[np.ndarray, torch.Tensor], time_mask_param=10):
    _, num_time = spec.shape
    t = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, num_time - t)
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().numpy()
    spec[:, t0 : t0 + t] = 0
    return spec


def spec_augmentation(
    spec: Union[np.ndarray, torch.Tensor],
    time_mask_param=10,
    freq_mask_param=10,
    warp_factor=5,
):
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().numpy()
    spec = time_warp(spec=spec, warp_factor=warp_factor)
    spec = frequency_mask(spec=spec, freq_mask_param=freq_mask_param)
    spec = time_mask(spec=spec, time_mask_param=time_mask_param)
    return spec


# Environmental and Contextual Augmentations
# Room Impulse Response (RIR) Augmentation – Simulate real-world room acoustics.
# Background Noise Injection – Add street, office, or nature sounds for robustness.
# Speed Perturbation – Create natural-sounding variations of speech/music.


def apply_rir(audio: Union[torch.Tensor, np.ndarray], rir):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    reverberant_audio = signal.convolve(audio, rir, mode="full")[: len(audio)]
    return reverberant_audio


def generate_synthetic_rir(fs=16000, rt60=0.3, room_size=(5, 4, 3), num_reflections=5):
    rir_length = int(fs * rt60)
    rir = np.zeros(rir_length)
    rir[0] = 1
    for _ in range(num_reflections):
        delay = np.random.randint(10, int(fs * 0.02))
        amplitude = np.random.uniform(0.3, 0.7)
        if delay < rir_length:
            rir[delay] += amplitude

    decay_factor = np.linspace(0, 5, rir_length)
    rir += np.exp(-decay_factor) * np.random.uniform(0.5, 1.0)
    rir /= np.max(np.abs(rir))
    return rir


def add_background_noise(audio: Union[np.ndarray, torch.Tensor], noise, snr_db=10):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    audio_power = np.mean(audio**2)
    noise_power = np.mean(noise**2)

    snr_linear = 10 ** (snr_db / 10)
    noise = noise * np.sqrt(audio_power / (snr_linear * noise_power))
    noisy_audio = audio + noise[: len(audio)]
    return noisy_audio


def change_speed(audio: Union[np.ndarray, torch.Tensor], sr, speed_factor=1.1):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().numpy()
    return librosa.resample(audio, orig_sr=sr, target_sr=int(sr * speed_factor))
