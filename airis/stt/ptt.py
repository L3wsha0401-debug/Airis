"""
Push-to-Talk (PTT) keyboard listener.
Uses Windows ctypes.windll.user32.GetAsyncKeyState for zero-dependency global hotkey detection.
"""

from __future__ import annotations
import time
import ctypes
import logging
import threading
from typing import Callable, Optional
import numpy as np

from airis.stt.recorder import AudioRecorder

VK_MAP = {
    "space": 0x20,
    "caps_lock": 0x14,
    "caps": 0x14,
    "shift": 0x10,
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "f2": 0x71,
    "f4": 0x73,
    "tab": 0x09,
}


class PTTListener:
    """
    Background Push-to-Talk hotkey monitor.
    Samples keypress state at 15 ms intervals for instant (< 15 ms) barge-in response.
    """
    def __init__(
        self,
        recorder: Optional[AudioRecorder] = None,
        hotkey: str = "space",
        on_barge_in: Optional[Callable[[], None]] = None,
        on_utterance_ready: Optional[Callable[[np.ndarray], None]] = None,
        poll_interval_s: float = 0.015  # 15 ms polling
    ):
        self.recorder = recorder
        self.hotkey = hotkey.lower()
        self.vk_code = VK_MAP.get(self.hotkey, 0x20)
        self.on_barge_in = on_barge_in
        self.on_utterance_ready = on_utterance_ready
        self.poll_interval_s = poll_interval_s

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_pressed = False
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()

        # Subscribe to raw audio chunks from recorder
        if self.recorder is not None:
            self.recorder.subscribe(self._on_audio_chunk)

    @property
    def is_pressed(self) -> bool:
        with self._lock:
            return self._is_pressed

    def _is_key_down(self) -> bool:
        try:
            user32 = getattr(ctypes, "windll", None)
            if user32 is not None and hasattr(user32, "user32"):
                return bool(user32.user32.GetAsyncKeyState(self.vk_code) & 0x8000)
            return False
        except Exception:
            return False

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        with self._lock:
            if self._is_pressed:
                self._buffer.append(chunk)

    def _poll_worker(self) -> None:
        while not self._stop_event.is_set():
            down = self._is_key_down()

            if down and not self._is_pressed:
                # Key pressed (UP -> DOWN)
                with self._lock:
                    self._is_pressed = True
                    pre_roll = self.recorder.get_pre_roll() if self.recorder else np.zeros(0, dtype=np.float32)
                    self._buffer = [pre_roll] if len(pre_roll) > 0 else []

                logging.info(f"[PTT] Hotkey '{self.hotkey}' PRESSED -> Recording started")
                # Immediately fire barge-in callback (< 15 ms)
                if self.on_barge_in:
                    try:
                        self.on_barge_in()
                    except Exception as e:
                        logging.error(f"[PTT Barge-in Callback Error]: {e}")

            elif not down and self._is_pressed:
                # Key released (DOWN -> UP)
                with self._lock:
                    self._is_pressed = False
                    if self._buffer:
                        audio = np.concatenate(self._buffer)
                        self._buffer = []
                    else:
                        audio = np.zeros(0, dtype=np.float32)

                logging.info(f"[PTT] Hotkey '{self.hotkey}' RELEASED -> Finalizing audio ({len(audio)} samples)")
                if len(audio) > 0 and self.on_utterance_ready:
                    try:
                        self.on_utterance_ready(audio)
                    except Exception as e:
                        logging.error(f"[PTT Utterance Callback Error]: {e}")

            time.sleep(self.poll_interval_s)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_worker, name="PTTListenerThread", daemon=True)
        self._thread.start()
        logging.info(f"[PTT] Listener active on hotkey '{self.hotkey}'")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.recorder is not None and hasattr(self.recorder, "unsubscribe"):
            self.recorder.unsubscribe(self._on_audio_chunk)
        logging.info("[PTT] Listener stopped")
