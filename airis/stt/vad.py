"""
Real-time Voice Activity Detection (VAD) state machine.
Uses faster-whisper bundled Silero VAD v6 ONNX model with energy floor gating.
Fires speech-start event on 2 consecutive frames (~64 ms) to trigger instant barge-in.
"""

from __future__ import annotations
import logging
from typing import Callable, Optional
import numpy as np
from faster_whisper.vad import get_vad_model
from airis.stt.recorder import AudioRecorder


class VADDetector:
    """
    Continuous Voice Activity Detector tracking speech probability transitions.
    Implements dynamic threshold adaptation during assistant speech to prevent echo triggers.
    """
    def __init__(
        self,
        recorder: Optional[AudioRecorder] = None,
        speech_threshold: float = 0.5,
        speech_threshold_speaking: float = 0.75,
        energy_floor: float = 0.005,
        min_speech_frames: int = 2,  # 2 frames * 32 ms = 64 ms onset confirmation
        silence_timeout_s: float = 0.8,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_utterance_ready: Optional[Callable[[np.ndarray], None]] = None,
        is_assistant_speaking_fn: Optional[Callable[[], bool]] = None,
        vad_model=None
    ):
        self.recorder = recorder
        self.speech_threshold = speech_threshold
        self.speech_threshold_speaking = speech_threshold_speaking
        self.energy_floor = energy_floor
        self.min_speech_frames = min_speech_frames
        self.silence_timeout_s = silence_timeout_s
        self.silence_frames_timeout = max(1, int(round(silence_timeout_s / 0.032)))
        self.on_speech_start = on_speech_start
        self.on_utterance_ready = on_utterance_ready
        self.is_assistant_speaking_fn = is_assistant_speaking_fn

        self.vad_model = vad_model if vad_model is not None else get_vad_model()
        self.state = "IDLE"  # "IDLE" or "RECORDING"
        self._speech_streak = 0
        self._silence_streak = 0
        self._buffer: list[np.ndarray] = []

        if self.recorder is not None:
            self.recorder.subscribe(self.process_chunk)

    def reset(self) -> None:
        """Resets detector state to IDLE and clears buffers."""
        self.state = "IDLE"
        self._speech_streak = 0
        self._silence_streak = 0
        self._buffer.clear()

    def process_chunk(self, chunk: np.ndarray) -> None:
        """
        Processes a single audio frame (typically 512 samples at 16 kHz).
        Updates state machine and fires callbacks on onset or offset.
        """
        chunk_arr = np.asarray(chunk, dtype=np.float32)
        if chunk_arr.ndim > 1:
            chunk_arr = chunk_arr.flatten()

        # Fast energy gate to bypass ONNX calculation on dead silence
        rms = float(np.sqrt(np.mean(chunk_arr ** 2))) if len(chunk_arr) > 0 else 0.0
        if rms < self.energy_floor:
            prob = 0.0
        else:
            try:
                res = self.vad_model(chunk_arr)
                if isinstance(res, (int, float)):
                    prob = float(res)
                elif hasattr(res, "__getitem__"):
                    prob = float(res[0])
                else:
                    prob = float(res)
            except Exception as e:
                logging.debug(f"[VAD Inference Error]: {e}")
                prob = 0.0

        # Dynamic threshold adaptation: elevate threshold if assistant is playing audio (echo suppression)
        thresh = self.speech_threshold
        if self.is_assistant_speaking_fn and self.is_assistant_speaking_fn():
            thresh = self.speech_threshold_speaking

        is_speech = prob >= thresh

        if self.state == "IDLE":
            if is_speech:
                self._speech_streak += 1
                if self._speech_streak >= self.min_speech_frames:
                    # User speech onset detected!
                    self.state = "RECORDING"
                    self._silence_streak = 0
                    logging.info(f"[VAD] User speech onset detected (prob={prob:.2f} >= {thresh:.2f})! Transitioning to RECORDING")
                    # Fire barge-in immediately (< 70 ms total latency)
                    if self.on_speech_start:
                        try:
                            self.on_speech_start()
                        except Exception as e:
                            logging.error(f"[VAD Speech Start Callback Error]: {e}")

                    pre_roll = self.recorder.get_pre_roll() if self.recorder else np.zeros(0, dtype=np.float32)
                    self._buffer = [pre_roll] if len(pre_roll) > 0 else []
                    self._buffer.append(chunk_arr)
            else:
                self._speech_streak = 0

        elif self.state == "RECORDING":
            self._buffer.append(chunk_arr)
            if not is_speech:
                self._silence_streak += 1
                if self._silence_streak >= self.silence_frames_timeout:
                    # Silence duration elapsed -> Utterance complete!
                    self.state = "IDLE"
                    self._speech_streak = 0
                    self._silence_streak = 0
                    logging.info("[VAD] Silence timeout reached. Utterance completed")
                    audio = np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.float32)
                    self._buffer = []

                    if len(audio) > 0 and self.on_utterance_ready:
                        try:
                            self.on_utterance_ready(audio)
                        except Exception as e:
                            logging.error(f"[VAD Utterance Callback Error]: {e}")
            else:
                self._silence_streak = 0

    def stop(self) -> None:
        """Unsubscribes from recorder and cleans up."""
        if self.recorder is not None and hasattr(self.recorder, "unsubscribe"):
            self.recorder.unsubscribe(self.process_chunk)
        self.reset()
