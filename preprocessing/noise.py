# ---------------------------- utf-8 encoding* ------------------------
# this file contains all the major kinds of audio augmentation technique
import os
import numpy as np
import torch
import librosa


# Time Domain technique for audio augmentation
# Noise Injection – Adding Gaussian, pink, real-world background noise.
# Time Stretching – Speed up or slow down without affecting pitch (Librosa time_stretch).
# Pitch Shifting – Shift frequency up or down (Librosa pitch_shift).
# Volume Perturbation – Random gain increase/decrease.
# Reverberation (Echo) – Apply room impulse response (RIR).
# Clipping/Distortion – Simulate microphone or speaker distortions.
# Waveform Cropping – Randomly remove or shuffle segments of audio


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
