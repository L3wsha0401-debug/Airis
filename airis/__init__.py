"""
Airis Companion Assistant Package.
Version 0.4.0 (Modular Architecture).
"""

from __future__ import annotations

__version__ = "0.4.0"

from airis.config import AppConfig, AirisConfig, STTConfig, load_config
from airis.core.locks import (
    ollama_lock,
    ollama_generation_lock,
    chroma_lock,
    session_history_lock,
)
from airis.core.lifecycle import (
    ShutdownCoordinator,
    shutdown_coordinator,
    graceful_shutdown,
    register_signal_handlers,
    is_exit_command,
)
from airis.core.prompts import SYSTEM_BASE_TEMPLATE, build_system_prompt
from airis.core.stream import (
    sanitize_text,
    clean_stream_filter,
    clean_artifacts,
    print_header,
    print_user_input_prompt,
)
from airis.core.commands import handle_chat_command
from airis.tts import (
    VoiceEngine,
    SentenceChunker,
    clean_and_normalize_tts_text,
    num_to_words_ru,
    resample_pitch_audio,
)
from airis.memory import (
    ChromaInitResult,
    initialize_and_migrate_chroma,
    recall_memory,
    insert_fact,
    get_core_fact,
    get_current_user_name,
    maybe_store_identity_fact,
    list_memories,
    list_stored_memories,
    forget_memory,
    save_session_history,
    load_session_history,
    dump_memory_log,
    dump_all_memories_to_log,
    log_chat_interaction,
)
from airis.stt import (
    STTEngine,
    AudioRecorder,
    VADDetector,
    PTTListener,
    BargeInController,
    BargeInDetector,
    STTListener,
)

__all__ = [
    "__version__",
    "AppConfig",
    "AirisConfig",
    "STTConfig",
    "load_config",
    "ollama_lock",
    "ollama_generation_lock",
    "chroma_lock",
    "session_history_lock",
    "ShutdownCoordinator",
    "shutdown_coordinator",
    "graceful_shutdown",
    "register_signal_handlers",
    "is_exit_command",
    "SYSTEM_BASE_TEMPLATE",
    "build_system_prompt",
    "sanitize_text",
    "clean_stream_filter",
    "clean_artifacts",
    "print_header",
    "print_user_input_prompt",
    "handle_chat_command",
    "VoiceEngine",
    "SentenceChunker",
    "clean_and_normalize_tts_text",
    "num_to_words_ru",
    "resample_pitch_audio",
    "ChromaInitResult",
    "initialize_and_migrate_chroma",
    "recall_memory",
    "insert_fact",
    "get_core_fact",
    "get_current_user_name",
    "maybe_store_identity_fact",
    "list_memories",
    "list_stored_memories",
    "forget_memory",
    "save_session_history",
    "load_session_history",
    "dump_memory_log",
    "dump_all_memories_to_log",
    "log_chat_interaction",
    "STTEngine",
    "AudioRecorder",
    "VADDetector",
    "PTTListener",
    "BargeInController",
    "BargeInDetector",
    "STTListener",
]
