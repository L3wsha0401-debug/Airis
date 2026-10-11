"""
Airis Audio Lip-Sync Processor.
Calculates root-mean-square (RMS) amplitude per audio frame, applies noise gating,
dynamic range normalization, and exponential moving average (EMA) smoothing for
smooth Live2D mouth parameter animation.
"""

from __future__ import annotations

import numpy as np
from typing import Union, List, Sequence
import torch


class LipSyncProcessor:
    """
    Analyzes raw audio signals and generates normalized mouth opening parameters
    in the range [0.0, 1.0] for VTube Studio / Live2D models.
    """

    def __init__(
        self,
        min_rms: float = 0.01,
        max_rms: float = 0.20,
        smoothing: float = 0.6,
    ) -> None:
        self.min_rms = float(min_rms)
        self.max_rms = float(max_rms)
        self.smoothing = float(smoothing)
        self._smoothed_value: float = 0.0

    @property
    def current_value(self) -> float:
        """Current smoothed mouth open parameter value."""
        return self._smoothed_value

    def calculate_rms(self, frame: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> float:
        """
        Computes the Root Mean Square (RMS) amplitude of an audio frame:
        RMS = sqrt( (1 / N) * sum(x_i^2) )
        """
        if frame is None:
            return 0.0

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()

        arr = np.asarray(frame, dtype=np.float32)
        if arr.size == 0:
            return 0.0

        if arr.ndim > 1:
            arr = arr.flatten()

        mean_square = np.mean(np.square(arr))
        if np.isnan(mean_square) or mean_square <= 0.0:
            return 0.0

        return float(np.sqrt(mean_square))

    def process_frame(self, frame: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> float:
        """
        Calculates frame RMS, applies noise gating and normalization between
        min_rms and max_rms, and applies EMA smoothing:
        mouth_open_t = alpha * raw_open + (1 - alpha) * mouth_open_{t-1}

        Returns:
            float: Smoothed mouth opening value between 0.0 and 1.0.
        """
        rms = self.calculate_rms(frame)

        # Noise gating & linear scaling
        if rms <= self.min_rms:
            raw_open = 0.0
        else:
            span = max(1e-6, self.max_rms - self.min_rms)
            raw_open = min(1.0, (rms - self.min_rms) / span)

        # Exponential Moving Average (EMA) smoothing
        alpha = np.clip(self.smoothing, 0.0, 1.0)
        self._smoothed_value = float(
            alpha * raw_open + (1.0 - alpha) * self._smoothed_value
        )
        self._smoothed_value = float(np.clip(self._smoothed_value, 0.0, 1.0))
        return self._smoothed_value

    def slice_frames(
        self,
        audio: Union[np.ndarray, torch.Tensor, Sequence[float]],
        sample_rate: int = 48000,
        frame_ms: int = 50,
    ) -> List[np.ndarray]:
        """
        Slices continuous audio into sequential fixed-duration frames (default 50 ms).
        """
        if audio is None:
            return []

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return []

        if arr.ndim > 1:
            arr = arr.flatten()

        frame_samples = max(1, int(sample_rate * (frame_ms / 1000.0)))
        total_samples = len(arr)

        frames = []
        for start_idx in range(0, total_samples, frame_samples):
            chunk = arr[start_idx : start_idx + frame_samples]
            if len(chunk) > 0:
                frames.append(chunk)

        return frames

    def reset(self) -> None:
        """Resets smoothed mouth position to 0.0 (closed)."""
        self._smoothed_value = 0.0
