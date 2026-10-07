"""
Unified STT Supervisor coordinating AudioRecorder, STTEngine, PTTListener, VADDetector,
and BargeInController. Emits transcribed Russian text directly to the interaction queue.
"""

from __future__ import annotations
import logging
import threading
from typing import Optional, Callable
import numpy as np

from airis.stt.engine import STTEngine
from airis.stt.recorder import AudioRecorder
from airis.stt.ptt import PTTListener
from airis.stt.vad import VADDetector
from airis.stt.barge_in import BargeInController
from airis.tts.engine import VoiceEngine


class STTListener:
    """
    Supervisor managing continuous speech capture, mode selection (PTT vs VAD),
    barge-in interruption callbacks, and asynchronous background transcription.
    """
    def __init__(
        self,
        voice_engine: Optional[VoiceEngine] = None,
        input_mode: str = "ptt",
        hotkey: str = "space",
        on_transcription: Optional[Callable[[str], None]] = None,
        abort_generation_event: Optional[threading.Event] = None,
        model_name: str = "small",
        cpu_threads: int = 4,
        vad_threshold: float = 0.5,
        vad_threshold_speaking: float = 0.75,
        vad_silence_timeout_s: float = 0.8,
        energy_floor: float = 0.005,
        engine: Optional[STTEngine] = None,
        recorder: Optional[AudioRecorder] = None,
    ):
        self.voice_engine = voice_engine
        self.input_mode = (input_mode or "ptt").lower()
        self.hotkey = hotkey
        self.on_transcription = on_transcription
        self.abort_generation_event = abort_generation_event

        self.recorder = recorder if recorder is not None else AudioRecorder(sample_rate=16000, chunk_size=512)
        self.engine = engine if engine is not None else STTEngine(
            model_name=model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads
        )
        self.barge_in_controller = BargeInController(
            voice_engine=self.voice_engine,
            abort_generation_event=self.abort_generation_event
        )

        self.ptt_listener: Optional[PTTListener] = None
        self.vad_detector: Optional[VADDetector] = None

        if self.input_mode == "ptt":
            self.ptt_listener = PTTListener(
                recorder=self.recorder,
                hotkey=self.hotkey,
                on_barge_in=self.barge_in_controller.trigger_barge_in,
                on_utterance_ready=self._handle_audio_utterance
            )
        elif self.input_mode == "vad":
            self.vad_detector = VADDetector(
                recorder=self.recorder,
                speech_threshold=vad_threshold,
                speech_threshold_speaking=vad_threshold_speaking,
                energy_floor=energy_floor,
                silence_timeout_s=vad_silence_timeout_s,
                on_speech_start=self.barge_in_controller.trigger_barge_in,
                on_utterance_ready=self._handle_audio_utterance,
                is_assistant_speaking_fn=self.voice_engine.is_speaking if self.voice_engine else None
            )

    def _handle_audio_utterance(self, audio: np.ndarray) -> None:
        """Worker dispatch to transcribe audio on CPU in background thread."""
        def _transcribe_worker():
            try:
                text = self.engine.transcribe(audio)
                if text and self.on_transcription:
                    self.on_transcription(text)
            except Exception as e:
                logging.error(f"[STT Transcription Worker Error]: {e}")

        threading.Thread(target=_transcribe_worker, daemon=True, name="STTTranscribeWorker").start()

    def start(self) -> None:
        """Starts audio recording and active listener subsystem."""
        self.recorder.start()
        if self.ptt_listener:
            self.ptt_listener.start()
        logging.info(f"[STTListener] Started in '{self.input_mode}' mode")

    def stop(self) -> None:
        """Stops active listener and audio recording."""
        if self.ptt_listener:
            self.ptt_listener.stop()
        if self.vad_detector:
            self.vad_detector.stop()
        self.recorder.stop()
        logging.info("[STTListener] Stopped")
