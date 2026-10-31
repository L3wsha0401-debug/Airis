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
    Background Push-to-Talk hotkey monitor with configurable post-roll buffering.
    Samples keypress state at 15 ms intervals for instant (< 15 ms) barge-in response.
    Continues capturing audio for `post_roll_ms` after key release, and seamlessly
    resumes recording if re-pressed during the post-roll window.
    """
    def __init__(
        self,
        recorder: Optional[AudioRecorder] = None,
        hotkey: str = "space",
        on_barge_in: Optional[Callable[[], None]] = None,
        on_utterance_ready: Optional[Callable[[np.ndarray], None]] = None,
        poll_interval_s: float = 0.015,  # 15 ms polling
        post_roll_ms: int = 250,
        ptt_post_roll_ms: Optional[int] = None,
    ):
        self.recorder = recorder
        self.hotkey = hotkey.lower()
        self.vk_code = VK_MAP.get(self.hotkey, 0x20)
        self.on_barge_in = on_barge_in
        self.on_utterance_ready = on_utterance_ready
        self.poll_interval_s = poll_interval_s
        self.post_roll_ms = ptt_post_roll_ms if ptt_post_roll_ms is not None else post_roll_ms

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_pressed = False
        self._release_time: Optional[float] = None
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()

        # Subscribe to raw audio chunks from recorder
        if self.recorder is not None:
            self.recorder.subscribe(self._on_audio_chunk)

    @property
    def is_pressed(self) -> bool:
        """Returns True if user is actively recording or currently in post-roll window."""
        with self._lock:
            return self._is_pressed

    @property
    def is_physically_pressed(self) -> bool:
        """Returns True if the physical hotkey is currently held down."""
        with self._lock:
            return self._is_pressed and self._release_time is None

    @property
    def is_in_post_roll(self) -> bool:
        """Returns True if currently buffering during post-roll window."""
        with self._lock:
            return self._is_pressed and self._release_time is not None

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
            now = time.monotonic()
            audio_to_emit: Optional[np.ndarray] = None
            fire_barge_in = False

            with self._lock:
                if down and not self._is_pressed:
                    # Key pressed from IDLE (UP -> DOWN)
                    self._is_pressed = True
                    self._release_time = None
                    pre_roll = self.recorder.get_pre_roll() if self.recorder else np.zeros(0, dtype=np.float32)
                    self._buffer = [pre_roll] if len(pre_roll) > 0 else []
                    fire_barge_in = True
                    logging.info(f"[PTT] Hotkey '{self.hotkey}' PRESSED -> Recording started")

                elif down and self._is_pressed and self._release_time is not None:
                    # Key re-pressed during post-roll window -> cancel release and seamlessly continue
                    self._release_time = None
                    fire_barge_in = True
                    logging.info(f"[PTT] Hotkey '{self.hotkey}' RE-PRESSED during post-roll -> Resuming recording")

                elif not down and self._is_pressed and self._release_time is None:
                    # Key released (DOWN -> UP) -> start post-roll window
                    if self.post_roll_ms <= 0:
                        # Zero post-roll: finalize immediately
                        self._is_pressed = False
                        self._release_time = None
                        if self._buffer:
                            audio_to_emit = np.concatenate(self._buffer)
                            self._buffer = []
                        else:
                            audio_to_emit = np.zeros(0, dtype=np.float32)
                    else:
                        self._release_time = now
                        logging.debug(f"[PTT] Hotkey '{self.hotkey}' RELEASED -> Post-roll window started ({self.post_roll_ms}ms)")

                elif not down and self._is_pressed and self._release_time is not None:
                    # In post-roll window -> check deadline
                    elapsed_ms = (now - self._release_time) * 1000.0
                    if elapsed_ms >= self.post_roll_ms:
                        self._is_pressed = False
                        self._release_time = None
                        if self._buffer:
                            audio_to_emit = np.concatenate(self._buffer)
                            self._buffer = []
                        else:
                            audio_to_emit = np.zeros(0, dtype=np.float32)
                        logging.info(
                            f"[PTT] Hotkey '{self.hotkey}' post-roll expired ({elapsed_ms:.1f}ms) "
                            f"-> Finalizing audio ({len(audio_to_emit)} samples)"
                        )

            # Callbacks invoked OUTSIDE self._lock to prevent deadlocks
            if fire_barge_in and self.on_barge_in:
                try:
                    self.on_barge_in()
                except Exception as e:
                    logging.error(f"[PTT Barge-in Callback Error]: {e}")

            if audio_to_emit is not None and len(audio_to_emit) > 0 and self.on_utterance_ready:
                try:
                    self.on_utterance_ready(audio_to_emit)
                except Exception as e:
                    logging.error(f"[PTT Utterance Callback Error]: {e}")

            time.sleep(self.poll_interval_s)

        # On loop termination, flush in-flight buffer if stopped during recording/post-roll
        audio_flush: Optional[np.ndarray] = None
        with self._lock:
            if self._buffer and self._is_pressed:
                audio_flush = np.concatenate(self._buffer)
                self._buffer = []
                self._is_pressed = False
                self._release_time = None

        if audio_flush is not None and len(audio_flush) > 0 and self.on_utterance_ready:
            try:
                self.on_utterance_ready(audio_flush)
            except Exception as e:
                logging.error(f"[PTT Utterance Flush Error on stop]: {e}")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_worker, name="PTTListenerThread", daemon=True)
        self._thread.start()
        logging.info(f"[PTT] Listener active on hotkey '{self.hotkey}' (post_roll={self.post_roll_ms}ms)")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.recorder is not None and hasattr(self.recorder, "unsubscribe"):
            self.recorder.unsubscribe(self._on_audio_chunk)
        logging.info("[PTT] Listener stopped")
