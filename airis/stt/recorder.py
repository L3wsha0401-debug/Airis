"""
Thread-safe continuous microphone capture via sounddevice.
Captures 16 kHz mono float32 audio and provides rolling ring buffer.
"""

from __future__ import annotations
import queue
import logging
import threading
from typing import Optional, Callable
import sounddevice as sd
import numpy as np


class AudioRecorder:
    """
    Continuous microphone stream recorder with rolling ring buffer and subscriber fan-out.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,  # 32 ms chunk at 16 kHz
        device: Optional[int] = None,
        pre_roll_chunks: int = 15  # ~480 ms pre-speech ring buffer
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        self.pre_roll_chunks = pre_roll_chunks

        self._stream: Optional[sd.InputStream] = None
        self._is_recording = threading.Event()
        self._subscribers: list[Callable[[np.ndarray], None]] = []
        self._ring_buffer = queue.Queue(maxsize=self.pre_roll_chunks)
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register subscriber for raw audio chunks."""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[np.ndarray], None]) -> None:
        """Unregister subscriber."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logging.debug(f"[AudioRecorder Status]: {status}")
        arr = np.asarray(indata, dtype=np.float32)
        if arr.ndim > 1:
            chunk = arr[:, 0].copy()
        else:
            chunk = arr.copy()

        # Update rolling pre-roll buffer
        if self._ring_buffer.full():
            try:
                self._ring_buffer.get_nowait()
            except queue.Empty:
                pass
        try:
            self._ring_buffer.put_nowait(chunk)
        except queue.Full:
            pass

        # Broadcast to subscribers
        with self._lock:
            subs = list(self._subscribers)
        for sub in subs:
            try:
                sub(chunk)
            except Exception as e:
                logging.error(f"[AudioRecorder Callback Error]: {e}")

    def get_pre_roll(self) -> np.ndarray:
        """Extracts the accumulated pre-roll chunks as a single continuous array."""
        chunks = list(self._ring_buffer.queue)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def is_recording(self) -> bool:
        return self._is_recording.is_set()

    def start(self) -> None:
        if self._is_recording.is_set():
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._audio_callback
        )
        self._stream.start()
        self._is_recording.set()
        logging.info(f"[AudioRecorder] Microphone stream started at {self.sample_rate} Hz mono")

    def stop(self) -> None:
        self._is_recording.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logging.debug(f"[AudioRecorder Stop Error]: {e}")
            self._stream = None
        logging.info("[AudioRecorder] Microphone stream stopped")
