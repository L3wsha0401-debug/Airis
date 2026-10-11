"""
Autonomous Idle Chatter Module for Airis.
Provides IdleCoordinator state machine, forbidden phrase negative constraints,
and spontaneous initiative prompt building.
"""

from __future__ import annotations
import time
import random
import threading
from typing import Optional, Callable, Any, Sequence, Dict

from airis.config import IdleConfig
from airis.core.prompts import build_system_prompt, load_idle_template

FORBIDDEN_IDLE_PHRASES: tuple[str, ...] = (
    "повисла пауза",
    "пауза",
    "мы молчим",
    "тишина",
    "ты замолчал",
    "ты молчишь",
    "почему ты молчишь",
    "ты тут?",
    "ты куда пропал",
    "ты уснул",
    "чего молчишь",
)

IDLE_INTERNAL_TRIGGER: str = (
    "[Инициатива Айрис: в диалоге возникла естественная пауза. Возьми слово первой, "
    "не упоминая паузу, тишину или молчание. Продолжи тему или поделись спонтанным наблюдением.]"
)


def build_idle_prompt(
    conversation_history: Optional[Sequence[Dict[str, str]]] = None,
    user_name: str = "l3wsha",
    collection_obj: Optional[Any] = None,
    lock: Optional[threading.Lock] = None,
) -> tuple[str, str]:
    """
    Constructs dynamic system prompt and internal initiative trigger for autonomous idle chatter.
    Enforces spontaneous Neuro-sama style initiative and strictly bans pause clichés.
    """
    # Context query for semantic memory retrieval based on recent history
    context_query = "общение и мысли"
    if conversation_history:
        for turn in reversed(conversation_history):
            content = turn.get("content", "").strip()
            if content and not content.startswith("["):
                context_query = content[:100]
                break

    base_sys_prompt = build_system_prompt(
        user_query=context_query,
        user_name=user_name,
        collection_obj=collection_obj,
        lock=lock,
    )

    forbidden_list_str = ", ".join(f'"{phrase}"' for phrase in FORBIDDEN_IDLE_PHRASES)

    # Idle instructions come from the user's local idle_prompt.txt (optional);
    # a neutral built-in template is used when the file is absent.
    idle_instructions = "\n\n" + load_idle_template().format(
        user_name=user_name,
        forbidden_phrases=forbidden_list_str,
    )

    full_system_prompt = base_sys_prompt + idle_instructions
    return full_system_prompt, IDLE_INTERNAL_TRIGGER


class IdleCoordinator:
    """
    State machine coordinating autonomous idle initiative turns, postponement during
    active audio playback or voice recording, and activity resets.
    """
    def __init__(
        self,
        config: Optional[IdleConfig] = None,
        voice_engine: Optional[Any] = None,
        stt_listener: Optional[Any] = None,
        time_fn: Optional[Callable[[], float]] = None,
        random_fn: Optional[Callable[[float, float], float]] = None,
    ):
        self.config = config if config is not None else IdleConfig()
        self.voice_engine = voice_engine
        self.stt_listener = stt_listener
        self.time_fn = time_fn if time_fn is not None else time.time
        self.random_fn = random_fn if random_fn is not None else random.uniform

        self._lock = threading.Lock()
        self.user_has_spoken: bool = False
        self.consecutive_idle_count: int = 0
        self.last_activity_time: float = self.time_fn()
        self.current_deadline: float = self._pick_deadline()

    def _pick_deadline(self) -> float:
        min_s = float(self.config.idle_min_seconds)
        max_s = float(self.config.idle_max_seconds)
        if min_s > max_s:
            min_s, max_s = max_s, min_s
        return self.random_fn(min_s, max_s)

    def record_user_activity(self) -> None:
        """Called whenever user provides text or voice input."""
        with self._lock:
            self.user_has_spoken = True
            self.consecutive_idle_count = 0
            self.last_activity_time = self.time_fn()
            self.current_deadline = self._pick_deadline()

    def record_assistant_activity(self) -> None:
        """Called when assistant finishes speaking (TTS finished)."""
        with self._lock:
            self.last_activity_time = self.time_fn()
            self.current_deadline = self._pick_deadline()

    def check_idle_timeout(self) -> bool:
        """
        Evaluates whether an idle initiative turn should trigger.
        Returns True if deadline elapsed without user/assistant activity or speech/recording.
        """
        with self._lock:
            if not self.config.enabled:
                return False

            if not self.user_has_spoken:
                return False

            if self.consecutive_idle_count >= self.config.max_idle_messages:
                return False

            now = self.time_fn()

            # Postpone if TTS voice engine is synthesizing, queued, or actively playing audio
            if self.voice_engine is not None:
                try:
                    speaking = getattr(self.voice_engine, "is_speaking", False)
                    if callable(speaking):
                        speaking = speaking()
                    if speaking:
                        self.last_activity_time = now
                        return False
                except Exception:
                    pass

            # Postpone if STT listener is recording (PTT pressed or VAD speech active)
            if self.stt_listener is not None:
                try:
                    recording = getattr(self.stt_listener, "is_recording", False)
                    if callable(recording):
                        recording = recording()
                    if recording:
                        self.last_activity_time = now
                        return False
                except Exception:
                    pass

            elapsed = now - self.last_activity_time
            if elapsed >= self.current_deadline:
                self.consecutive_idle_count += 1
                self.last_activity_time = now
                self.current_deadline = self._pick_deadline()
                return True

            return False

    def reset(self) -> None:
        """Resets the coordinator state to initial conditions."""
        with self._lock:
            self.user_has_spoken = False
            self.consecutive_idle_count = 0
            self.last_activity_time = self.time_fn()
            self.current_deadline = self._pick_deadline()


__all__ = [
    "FORBIDDEN_IDLE_PHRASES",
    "IDLE_INTERNAL_TRIGGER",
    "build_idle_prompt",
    "IdleCoordinator",
]
