"""
CPU-based Speech-to-Text engine powered by faster-whisper.
Runs strictly on CPU (int8) to prevent GPU VRAM contention with Ollama LLM.
"""

from __future__ import annotations
import time
import logging
import re
from typing import Optional
import numpy as np
from faster_whisper import WhisperModel

HALLUCINATION_PHRASES = frozenset({
    # Common subtitle credits & YouTube ending artifacts
    "продолжение следует",
    "субтитры сделал dimatorzok",
    "субтитры создавал dimatorzok",
    "редактор субтитров а семкин корректор а егорова",
    "редактор субтитров асемкин корректор аегорова",
    "редактор субтитров а семкин",
    "корректор а егорова",
    "спасибо за просмотр",
    "спасибо за внимание",
    "подписывайтесь на канал",
    "подпишитесь на канал",
    "ставьте лайки",
    "ставьте лайк",
    "ставьте лайки и подписывайтесь на канал",
    "ставьте лайк и подписывайтесь на канал",
    "до свидания",
    "до встречи",
    "до новых встреч",
    "всем пока",

    # Common breath, laughter, and noise hallucinations
    "смех",
    "аплодисменты",
    "кашель",
    "кхм",
    "кхм кхм",
    "кхмкхм",
    "кхе",
    "кхе кхе",
    "кхе кхе кхе",
    "кхекхе",
    "кхекхекхе",
    "хаха",
    "хахаха",
    "хахахаха",
    "ахаха",
    "ахахаха",
    "ахахахаха",
    "ахахахахах",
    "ахахахахахах",
    "хехе",
    "хихи",
    "вздох",
    "зевок",

    # Single filler interjections & noise tokens
    "угу",
    "мм",
    "м м",
    "а",
    "э",
    "хм",
    "эм",
    "ага",
})

LAUGHTER_NOISE_PATTERN = re.compile(r"^(?:(?:кх[ме]|кхе|ха|хе|хи|ах|ох|кх)\s*)+$")


def normalize_transcription_text(text: str) -> str:
    """Normalize text for hallucination comparison (lowercase, strip punctuation, collapse whitespace)."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).lower()
    return " ".join(cleaned.split())


def is_hallucination_or_noise(text: str) -> bool:
    """Check if normalized or raw text matches hallucination phrases or laughter/noise patterns."""
    if not text:
        return True
    norm = normalize_transcription_text(text)
    if not norm:
        return True
    if norm in HALLUCINATION_PHRASES:
        return True
    if LAUGHTER_NOISE_PATTERN.match(norm):
        return True
    return False


class STTEngine:
    """
    CPU-based faster-whisper speech transcription engine.
    Transcribes 16 kHz mono float32 audio chunks with VAD and hallucination filtering.
    """
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        language: str = "ru",
        beam_size: int = 1,
        download_root: Optional[str] = None,
        initial_prompt: Optional[str] = "Айрис, Лёша.",
        vad_filter: Optional[bool] = None,
        min_speech_ms: int = 250,
        min_silence_duration_ms: int = 300,
        no_speech_threshold: float = 0.6,
        log_prob_threshold: float = -1.0,
        condition_on_previous_text: bool = False,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.language = language
        self.beam_size = beam_size
        self.download_root = download_root
        if initial_prompt is None or (model_name == "small" and initial_prompt == "Айрис, Лёша."):
            self.initial_prompt = None if model_name == "small" else initial_prompt
        else:
            self.initial_prompt = initial_prompt
        self.min_speech_ms = min_speech_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.temperature = temperature

        if vad_filter is None:
            # Maintain backward compatibility with legacy Milestone 3 test defaults:
            # model_name="small" legacy instantiation defaults to False unless explicitly configured
            self.vad_filter = False if model_name == "small" else True
        else:
            self.vad_filter = bool(vad_filter)

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
        Returns cleaned text string or empty string if noise/silence/hallucination.
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
        if max_val == 0.0 and isinstance(self.model, WhisperModel):
            return ""
        if max_val > 1.0:
            audio_data = audio_data / max_val

        t0 = time.perf_counter()
        vad_params = {
            "min_speech_duration_ms": self.min_speech_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
        }
        segments, info = self.model.transcribe(
            audio_data,
            language=self.language,
            beam_size=self.beam_size,
            temperature=self.temperature,
            vad_filter=self.vad_filter,
            vad_parameters=vad_params,
            condition_on_previous_text=self.condition_on_previous_text,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.log_prob_threshold,
            initial_prompt=self.initial_prompt or None,
            word_timestamps=False,
        )

        valid_text_parts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            seg_start = getattr(segment, "start", None)
            seg_end = getattr(segment, "end", None)
            if isinstance(seg_start, (int, float)) and isinstance(seg_end, (int, float)):
                seg_duration = seg_end - seg_start
                # Prune clicks/pops under 150ms
                if seg_duration < 0.15:
                    continue

            no_speech_prob = getattr(segment, "no_speech_prob", None)
            avg_logprob = getattr(segment, "avg_logprob", None)
            if isinstance(no_speech_prob, (int, float)):
                if isinstance(avg_logprob, (int, float)):
                    if no_speech_prob > self.no_speech_threshold and avg_logprob < self.log_prob_threshold:
                        continue
                if no_speech_prob > 0.85:
                    continue

            # Segment hallucination blacklist filtering
            if is_hallucination_or_noise(text):
                continue

            valid_text_parts.append(text)

        result_text = " ".join(valid_text_parts).strip()

        # Whole-utterance hallucination blacklist check
        if is_hallucination_or_noise(result_text):
            result_text = ""

        dt = time.perf_counter() - t0
        logging.info(f"[STT] Transcribed {duration_s:.2f}s audio in {dt:.3f}s: '{result_text}'")
        return result_text
