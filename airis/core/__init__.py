"""
Airis Core Package.
Provides lifecycle coordination, concurrency locks, stream filtering, and prompts.
"""

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
from airis.core.prompts import (
    SystemPromptNotFoundError,
    load_system_template,
    load_idle_template,
    build_system_prompt,
)
from airis.core.stream import (
    sanitize_text,
    clean_stream_filter,
    clean_artifacts,
    print_header,
    print_user_input_prompt,
)
from airis.core.commands import handle_chat_command
from airis.core.idle import (
    FORBIDDEN_IDLE_PHRASES,
    IDLE_INTERNAL_TRIGGER,
    build_idle_prompt,
    IdleCoordinator,
)

__all__ = [
    "ollama_lock",
    "ollama_generation_lock",
    "chroma_lock",
    "session_history_lock",
    "ShutdownCoordinator",
    "shutdown_coordinator",
    "graceful_shutdown",
    "register_signal_handlers",
    "is_exit_command",
    "SystemPromptNotFoundError",
    "load_system_template",
    "load_idle_template",
    "build_system_prompt",
    "sanitize_text",
    "clean_stream_filter",
    "clean_artifacts",
    "print_header",
    "print_user_input_prompt",
    "handle_chat_command",
    "FORBIDDEN_IDLE_PHRASES",
    "IDLE_INTERNAL_TRIGGER",
    "build_idle_prompt",
    "IdleCoordinator",
]

