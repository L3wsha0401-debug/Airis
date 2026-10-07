"""
Airis Speech-to-Text (STT) Package.
Defines components for CPU-based speech transcription (faster-whisper),
Push-to-Talk (PTT), Voice Activity Detection (VAD), and conversational barge-in playback termination.
"""

from __future__ import annotations

from airis.stt.engine import STTEngine
from airis.stt.recorder import AudioRecorder
from airis.stt.vad import VADDetector
from airis.stt.ptt import PTTListener
from airis.stt.barge_in import BargeInController, BargeInDetector
from airis.stt.listener import STTListener

__all__ = [
    "STTEngine",
    "AudioRecorder",
    "VADDetector",
    "PTTListener",
    "BargeInController",
    "BargeInDetector",
    "STTListener",
]
