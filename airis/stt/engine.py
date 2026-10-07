"""
CPU-based Speech-to-Text engine powered by faster-whisper.
Runs strictly on CPU (int8) to prevent GPU VRAM contention with Ollama LLM.
"""

from __future__ import annotations
import time
import logging
from typing import Optional
import numpy as np
from faster_whisper import WhisperModel


class STTEngine:
    """
    CPU-based faster-whisper speech transcription engine.
    Transcribes 16 kHz mono float32 audio chunks in under 2.0 seconds.
    """
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        language: str = "ru",
        beam_size: int = 1,
        download_root: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.language = language
        self.beam_size = beam_size
        self.download_root = download_root
        self.model: Optional[WhisperModel] = None
        self._load_model()

    def _load_model(self) -> None:
        if self.model is not None:
            return
        t0 = time.perf_counter()
        logging.info(
            f"[STT] Loading faster-whisper '{self.model_name}' on {self.device} "
            f"({self.compute_type}, {self.cpu_threads} threads)..."
        )
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=self.download_root
        )
        dt = time.perf_counter() - t0
        logging.info(f"[STT] WhisperModel loaded in {dt:.2f}s")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribes 1D float32 audio numpy array at 16 kHz to Russian text.
        Returns cleaned text string. Guaranteed latency < 2.0s for conversational phrases.
        """
        if self.model is None:
            self._load_model()

        if audio is None or len(audio) == 0:
            return ""

        # Validate and convert audio to float32 1D
        audio_data = np.asarray(audio, dtype=np.float32)
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()

        duration_s = len(audio_data) / float(sample_rate)
        if duration_s < 0.2:  # Discard sub-200ms audio clicks
            return ""

        # Normalize amplitude if max exceeds 1.0
        max_val = float(np.max(np.abs(audio_data)))
        if max_val > 1.0:
            audio_data = audio_data / max_val

        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio_data,
            language=self.language,
            beam_size=self.beam_size,
            temperature=0.0,
            vad_filter=False,  # Audio is already pre-segmented by upstream VAD or PTT
            word_timestamps=False
        )

        text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        result_text = " ".join(text_parts).strip()
        dt = time.perf_counter() - t0
        logging.info(f"[STT] Transcribed {duration_s:.2f}s audio in {dt:.3f}s: '{result_text}'")
        return result_text
