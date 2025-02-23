# Audio Processing Pipeline

## Overview
This project provides a comprehensive pipeline for audio processing, feature extraction, noise reduction, and data augmentation. It is optimized for machine learning and deep learning applications and leverages GPU acceleration for efficient computation.

## Features
### 1. Audio Loading & Conversion
- Supports multiple audio formats: **WAV, MP3, FLAC, OGG**
- Converts stereo audio to **mono**
- Resampling (e.g., **44.1kHz → 16kHz**)
- Normalizes audio amplitude (**peak or RMS normalization**)
- Trims **leading/trailing silence**

### 2. Spectrogram Representations
#### (a) Time-Frequency Representations
- **Short-Time Fourier Transform (STFT)**
- **Inverse STFT** (reconstruct waveform)
- **Constant-Q Transform (CQT)**
- **Inverse CQT**

#### (b) Mel-based Representations
- **Mel-Spectrogram** (log-scaled frequency)
- **MFCC (Mel-Frequency Cepstral Coefficients)**
- **Inverse Mel-Spectrogram**

#### (c) Other Spectral Features
- **Chroma Feature Extraction** (musical pitch representation)
- **Spectral Centroid** (brightness of a sound)
- **Spectral Bandwidth & Spectral Contrast**
- **Spectral Rolloff** (frequency where energy drops)

### 3. Audio Augmentation (Data Augmentation Techniques)
#### (a) Time-domain Augmentations
- **Time Shifting** (randomly shift audio forward/backward)
- **Volume Perturbation** (increase/decrease amplitude)
- **Add Background Noise** (white noise, pink noise)
- **Reverberation** (simulate echo effects)
- **Time Stretching** (speed up or slow down audio without pitch shift)
- **Pitch Shifting** (change pitch while keeping speed constant)

#### (b) Frequency-domain Augmentations
- **SpecAugment** (Time Masking & Frequency Masking)
- **Equalization** (boost/cut specific frequency ranges)
- **Bandpass & Lowpass Filtering**

### 4. Feature Engineering for ML/DL Models
- **Zero-Crossing Rate (ZCR)**
- **Root Mean Square Energy (RMSE)**
- **Harmonic-to-Noise Ratio (HNR)**
- **Chromagram Features**

### 5. Noise Reduction & Enhancement
- **Spectral Subtraction** (removes stationary noise)
- **Wiener Filtering** (adaptive noise filtering)
- **Adaptive Noise Reduction** (Deep Learning-based methods)

### 6. Voice Activity Detection (VAD) & Speaker Separation
- **Detect & Remove Silence Segments** (Energy-based or Deep Learning-based)
- **Speaker Diarization** (separate multiple speakers in one recording)

### 7. Batch Processing & Parallelism
- **Efficient Processing of Large Datasets** (parallelized audio processing)
- **GPU Acceleration** for spectrogram computation (**using torchaudio/tensorflow**)

## Dependencies
- `librosa`
- `torchaudio`
- `numpy`
- `scipy`
- `pydub` (for audio format conversion)
