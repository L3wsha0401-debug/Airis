"""
Airis Memory Logging and Interaction Audit.
Outputs structured memory dumps to Logs/iris_memory.log and appends conversation turns to Logs/logs.txt.
"""

from __future__ import annotations
import os
import datetime
import logging
import threading
from typing import Optional

from airis.config import load_config, LOGS_DIR, MEMORY_LOG_FILE, CHAT_LOG_FILE
from airis.core.locks import chroma_lock


def log_chat_interaction(user_name: str, user_text: str, ai_text: str, chat_log_file: Optional[str] = None):
    """Logs conversation turn to logs.txt for audit and debugging."""
    target_path = chat_log_file or CHAT_LOG_FILE
    try:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(f"{user_name} ➔ {user_text}\n")
            f.write(f"Айрис: {ai_text}\n\n")
    except Exception as e:
        logging.error(f"[CHAT LOG] Error writing to {target_path}: {e}")


def dump_memory_log(
    collection_obj=None,
    logs_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    lock: Optional[threading.Lock] = None
) -> str:
    """
    Dumps structured snapshot of saved facts to Logs/iris_memory.log.
    Protected under chroma_lock.
    """
    from airis.memory.chroma import get_default_collection
    col = collection_obj if collection_obj is not None else get_default_collection()
    use_lock = lock if lock is not None else chroma_lock

    if log_file is None:
        target_dir = logs_dir or LOGS_DIR
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "iris_memory.log")
    else:
        target_path = log_file
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    timestamp = datetime.datetime.now().isoformat()
    lines = [f"\n--- ДАМП ПАМЯТИ НА КОНЕЦ СЕССИИ [{timestamp}] ---\n"]

    if col is not None:
        def _get_data():
            try:
                return col.get()
            except Exception as e:
                logging.error(f"[MEMORY] Error reading dump data: {e}")
                return {}

        if use_lock:
            with use_lock:
                data = _get_data()
        else:
            data = _get_data()

        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        user_facts = []
        self_facts = []

        for doc, meta in zip(docs, metas):
            if not doc:
                continue
            entity = meta.get("entity", "") if isinstance(meta, dict) else ""
            if entity == "user" or doc.startswith("Имя собеседника"):
                user_facts.append(doc)
            elif entity == "self":
                self_facts.append(doc)

        lines.append(f"Фактов о пользователе: {len(user_facts)}\n")
        for f in user_facts:
            lines.append(f" -> {f}\n")
        lines.append(f"Фактов об Айрис: {len(self_facts)}\n")
        for f in self_facts:
            lines.append(f" -> {f}\n")
        lines.append("-----------------------------------------\n")

    content = "".join(lines)
    try:
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        logging.error(f"[MEMORY] Error writing memory dump to {target_path}: {e}")

    return content


dump_all_memories_to_log = dump_memory_log
