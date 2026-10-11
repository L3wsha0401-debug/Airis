"""
Airis System Prompt Templates and Builders.

The persona / system prompt is NOT stored in the source code. It is loaded from a
local ``system_prompt.txt`` file in the project root (ignored by git) that every
user has to write manually. See ``system_prompt.example.txt`` and README.md.
"""

from __future__ import annotations
import os
import threading
from typing import Optional

from airis.config import BASE_DIR

EMOTION_CATALOG = (
    "neutral",
    "happy",
    "smug",
    "surprised",
    "angry",
    "blush",
    "bored",
    "thinking",
)
DEFAULT_EMOTION = "neutral"

SYSTEM_PROMPT_FILE = os.path.join(BASE_DIR, "system_prompt.txt")
IDLE_PROMPT_FILE = os.path.join(BASE_DIR, "idle_prompt.txt")

# Neutral fallback used when the optional idle_prompt.txt is absent.
DEFAULT_IDLE_TEMPLATE = """# SPONTANEOUS INITIATIVE (AUTONOMOUS REMARK DURING A PAUSE):
- The conversation has paused and you take the initiative.
- Do not mention the pause, silence or that {user_name} stopped talking. Avoid these phrases and their variations: {forbidden_phrases}.
- Continue the recent topic from an unexpected angle or share a short spontaneous observation.
- Length: strictly 1-2 short sentences.
"""


class SystemPromptNotFoundError(FileNotFoundError):
    """Raised when the user-provided system prompt file is missing or empty."""


def load_system_template(path: Optional[str] = None) -> str:
    """
    Loads the base system prompt template from ``system_prompt.txt``.

    The template may contain the ``{user_name}`` placeholder. Literal curly braces
    must be escaped as ``{{`` and ``}}``.
    """
    file_path = path if path is not None else SYSTEM_PROMPT_FILE
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        raise SystemPromptNotFoundError(
            f"System prompt file not found: {file_path}\n"
            "Create it manually (for example: copy system_prompt.example.txt "
            "to system_prompt.txt and edit it). See README.md."
        ) from None
    if not text:
        raise SystemPromptNotFoundError(
            f"System prompt file is empty: {file_path}\n"
            "Write your system prompt there. See README.md."
        )
    return text + "\n"


def load_idle_template(path: Optional[str] = None) -> str:
    """
    Loads the optional idle-chatter instruction template from ``idle_prompt.txt``.
    Supports ``{user_name}`` and ``{forbidden_phrases}`` placeholders.
    Falls back to a neutral built-in template if the file is absent or empty.
    """
    file_path = path if path is not None else IDLE_PROMPT_FILE
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        return DEFAULT_IDLE_TEMPLATE
    return (text + "\n") if text else DEFAULT_IDLE_TEMPLATE


def __getattr__(name: str):
    # Backward-compatible lazy attribute: the template is read on first access
    # instead of at import time, so importing the package never requires the file.
    if name == "SYSTEM_BASE_TEMPLATE":
        return load_system_template()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_system_prompt(
    user_query: str,
    user_name: str = "l3wsha",
    collection_obj=None,
    lock: Optional[threading.Lock] = None
) -> str:
    """
    Constructs dynamic system prompt injecting stored user identity facts
    and relevant semantic memories before LLM inference.
    """
    from airis.memory.chroma import (
        get_core_fact, recall_memory, CORE_USER_NAME_ID, get_default_collection
    )
    from airis.core.locks import chroma_lock
    from airis.config import load_config

    cfg = load_config()
    max_dist = cfg.memory.max_distance

    col = collection_obj if collection_obj is not None else get_default_collection()
    use_lock = lock if lock is not None else chroma_lock

    user_name_fact = get_core_fact(CORE_USER_NAME_ID, collection_obj=col, lock=use_lock)
    user_name_str = f"- {user_name_fact}\n" if user_name_fact else ""

    user_memories = recall_memory(
        col,
        user_query,
        entity="user",
        max_distance=max_dist,
        lock=use_lock
    )
    user_mem_str = "\n".join(f"- {m}" for m in user_memories)

    memory_block = ""
    if user_name_str or user_mem_str:
        memory_block += f"\n\n# ФАКТЫ О СОБЕСЕДНИКЕ:\n{user_name_str}{user_mem_str}"

    base_prompt = load_system_template().format(user_name=user_name)
    return base_prompt + memory_block


__all__ = [
    "EMOTION_CATALOG",
    "DEFAULT_EMOTION",
    "SYSTEM_PROMPT_FILE",
    "IDLE_PROMPT_FILE",
    "SystemPromptNotFoundError",
    "load_system_template",
    "load_idle_template",
    "build_system_prompt",
]
