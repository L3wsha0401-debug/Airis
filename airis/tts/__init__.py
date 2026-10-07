"""
Airis TTS Package.
Provides text normalization, sentence chunking, audio resampling, and streaming voice engine.
"""

from airis.tts.normalizer import num_to_words_ru, clean_and_normalize_tts_text
from airis.tts.chunker import SentenceChunker
from airis.tts.resampler import resample_pitch_audio
from airis.tts.engine import VoiceEngine

__all__ = [
    "num_to_words_ru",
    "clean_and_normalize_tts_text",
    "SentenceChunker",
    "resample_pitch_audio",
    "VoiceEngine",
]
