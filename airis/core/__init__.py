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
    SYSTEM_BASE_TEMPLATE,
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
    "SYSTEM_BASE_TEMPLATE",
    "build_system_prompt",
    "sanitize_text",
    "clean_stream_filter",
    "clean_artifacts",
    "print_header",
    "print_user_input_prompt",
    "handle_chat_command",
]
