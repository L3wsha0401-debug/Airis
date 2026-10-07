"""
Airis Audio Resampling and Pitch Shifting.
Provides fixed-rate 48 kHz software resampling with linear interpolation.
"""

from __future__ import annotations
import numpy as np
import torch


def resample_pitch_audio(
    audio: np.ndarray | torch.Tensor,
    pitch_shift: float = 1.25,
    sample_rate: int = 48000
) -> np.ndarray:
    """
    Software resamples audio array to adjust pitch and duration while preserving
    a fixed hardware playback sample rate (48 kHz).
    Output length is int(round(orig_len / pitch_shift)).
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.flatten()
    orig_len = len(audio)
    if orig_len == 0:
        return np.array([], dtype=np.float32)
    if pitch_shift <= 0:
        pitch_shift = 1.0
    if abs(pitch_shift - 1.0) < 1e-6:
        return audio.copy()
    target_len = int(round(orig_len / pitch_shift))
    if target_len <= 0:
        return np.array([], dtype=np.float32)
    x_orig = np.linspace(0, 1, orig_len, endpoint=False)
    x_target = np.linspace(0, 1, target_len, endpoint=False)
    resampled = np.interp(x_target, x_orig, audio).astype(np.float32)
    return resampled
