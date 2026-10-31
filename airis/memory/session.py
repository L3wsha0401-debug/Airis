"""
Airis Session History Persistence Module.
Provides atomic serialization and deserialization of dialogue turns with Windows file-lock mitigation.
"""

from __future__ import annotations
import os
import re
import json
import uuid
import time
import datetime
import logging
import threading
from typing import List, Dict, Optional, Any

from airis.config import load_config, SESSION_HISTORY_FILE
from airis.core.locks import session_history_lock
from airis.core.stream import clean_artifacts

_cfg = load_config()
MAX_HISTORY = _cfg.memory.max_history_turns
conversation_history: List[Dict[str, str]] = []

# Module-level tracking of messages generated in the current process run
session_new_messages: List[Dict[str, str]] = []
_session_tracker_lock = threading.Lock()
_session_summary_cache: Dict[str, tuple[float, Optional[str]]] = {}
_session_cache_lock = threading.Lock()


def record_session_message(role: str, content: str) -> None:
    """Records a dialogue message generated during the current session."""
    if role in ("user", "assistant", "system") and isinstance(content, str) and content.strip():
        with _session_tracker_lock:
            session_new_messages.append({"role": role, "content": content.strip()})


def get_session_new_messages() -> List[Dict[str, str]]:
    """Returns a snapshot copy of new messages accumulated during the current run."""
    with _session_tracker_lock:
        return list(session_new_messages)


def get_new_messages_count() -> int:
    """Returns the count of new dialogue messages generated during the current run."""
    with _session_tracker_lock:
        return len(session_new_messages)


def reset_session_tracker() -> None:
    """Resets the new messages tracker (primarily for testing and restart isolation)."""
    with _session_tracker_lock:
        session_new_messages.clear()
    with _session_cache_lock:
        _session_summary_cache.clear()


def load_session_summary(*args, **kwargs) -> Optional[str]:
    """
    Deserializes and returns the 'last_session_summary' string from session_history.json.
    Returns None if missing, corrupted, or empty.
    Thread-safe and Windows file-lock resilient (up to 10 retries with exponential backoff & mtime caching).
    """
    file_path = None
    lock = kwargs.get("lock", None)
    if len(args) >= 1:
        file_path = args[0]
    else:
        file_path = kwargs.get("file_path", kwargs.get("session_file", SESSION_HISTORY_FILE))
    if len(args) >= 2 and (hasattr(args[1], "__enter__") or hasattr(args[1], "acquire")):
        lock = args[1]
    if file_path is None:
        file_path = SESSION_HISTORY_FILE

    target_path = str(file_path)

    def _do_load():
        if not target_path or not os.path.exists(target_path):
            return None
        try:
            if os.path.getsize(target_path) == 0:
                return None
        except OSError:
            return None

        abs_path = os.path.abspath(target_path)
        try:
            mtime = os.path.getmtime(abs_path)
            with _session_cache_lock:
                if abs_path in _session_summary_cache:
                    cached_mtime, cached_summary = _session_summary_cache[abs_path]
                    if cached_mtime == mtime:
                        return cached_summary
        except OSError:
            pass

        max_retries = 10
        data = None
        for attempt in range(max_retries):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                break
            except (FileNotFoundError, json.JSONDecodeError):
                return None
            except (PermissionError, OSError):
                if attempt == max_retries - 1:
                    return None
                time.sleep(0.005 * (2 ** min(attempt, 5)) + (uuid.uuid4().int % 10) * 0.001)

        summary_result = None
        if isinstance(data, dict):
            summary = data.get("last_session_summary")
            if isinstance(summary, str) and summary.strip():
                summary_result = summary.strip()

        try:
            cur_mtime = os.path.getmtime(abs_path)
            with _session_cache_lock:
                _session_summary_cache[abs_path] = (cur_mtime, summary_result)
        except OSError:
            pass

        return summary_result

    use_lock = lock
    if use_lock:
        with use_lock:
            return _do_load()
    else:
        return _do_load()


def generate_session_summary(
    messages: List[Dict[str, str]],
    user_name: str = "l3wsha",
    model: Optional[str] = None,
    timeout: float = 6.0,
    lock: Optional[threading.Lock] = None,
) -> Optional[str]:
    """
    Generates a concise 2-3 sentence Russian session summary via Ollama under ollama_lock.
    Guarded with an internal timeout thread join (<= 6.0s) and exception handling.
    Returns None if messages < 4 or upon timeout/error.
    """
    if not messages or len(messages) < 4:
        return None

    from airis.core.locks import ollama_lock
    from airis.config import load_config
    import ollama

    cfg = load_config()
    model_name = model or cfg.ollama.model
    use_lock = lock if lock is not None else ollama_lock

    dialogue_lines = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        speaker = user_name if role == "user" else "Айрис"
        dialogue_lines.append(f"{speaker}: {content.strip()}")

    if not dialogue_lines:
        return None

    dialogue_text = "\n".join(dialogue_lines)
    prompt = (
        f"Кратко обобщи прошедший диалог между пользователем ({user_name}) и ИИ-компаньоном Айрис.\n"
        f"Составь краткую сводку диалога ровно в 2-3 предложения на русском языке.\n"
        f"Опиши основные темы, вопросы или результаты беседы без лишних деталей.\n"
        f"Не используй вводных шаблонных фраз вроде 'В данном диалоге...' или 'Пользователь и Айрис говорили о...'. "
        f"Пиши сразу по существу.\n\n"
        f"Диалог:\n{dialogue_text}\n\n"
        f"Сводка (2-3 предложения):"
    )

    result: List[Optional[str]] = [None]

    def _worker():
        try:
            with use_lock:
                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": 0.3,
                        "num_predict": 120,
                    },
                )
                raw_text = response.get("message", {}).get("content", "").strip() if isinstance(response, dict) else ""
                clean = re.sub(r'^["\'`]+|["\'`]+$', '', raw_text).strip()
                if clean and 10 <= len(clean) <= 1200:
                    result[0] = clean
        except Exception as e:
            logging.warning(f"[SESSION SUMMARY] Ollama error generating summary: {e}")

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()
    worker_thread.join(timeout=timeout)

    if worker_thread.is_alive():
        logging.warning(f"[SESSION SUMMARY] Generation timed out after {timeout}s")
        return None

    return result[0]



def save_session_history(
    *args,
    last_session_summary: Optional[str] = None,
    preserve_existing_summary: bool = True,
    **kwargs
) -> None:
    """
    Atomically saves session history to disk with metadata and os.replace.
    Resilient to WinError 5 / 32 on Windows via unique temp files, f.flush(),
    os.fsync(), exponential retry backoff (up to 8 attempts), and clean tmp cleanup.

    Supports polymorphic calls:
      1. save_session_history(file_path, history, max_turns=12, lock=None)
      2. save_session_history(history, file_path=SESSION_HISTORY_FILE, max_turns=12, lock=None)
      3. save_session_history(history, file_path, max_turns=12, lock=None)
      4. save_session_history()  # uses conversation_history and SESSION_HISTORY_FILE
      5. Keyword arguments: session_file, history, max_turns, retention_limit_turns, lock,
         last_session_summary, preserve_existing_summary.
    """
    file_path = None
    history = None
    max_turns = kwargs.get("max_turns", kwargs.get("retention_limit_turns", MAX_HISTORY))
    lock = kwargs.get("lock", None)
    last_session_summary = kwargs.get("last_session_summary", last_session_summary)
    preserve_existing_summary = kwargs.get("preserve_existing_summary", preserve_existing_summary)

    if len(args) == 1:
        if isinstance(args[0], (str, os.PathLike)):
            file_path = str(args[0])
            history = kwargs.get("history", conversation_history)
        else:
            history = args[0]
            file_path = kwargs.get("file_path", kwargs.get("session_file", SESSION_HISTORY_FILE))
    elif len(args) >= 2:
        if isinstance(args[0], (str, os.PathLike)) and not isinstance(args[1], (str, os.PathLike)):
            file_path = str(args[0])
            history = args[1]
        elif isinstance(args[1], (str, os.PathLike)) and not isinstance(args[0], (str, os.PathLike)):
            history = args[0]
            file_path = str(args[1])
        elif isinstance(args[0], (str, os.PathLike)) and isinstance(args[1], (str, os.PathLike)):
            file_path = str(args[0])
            history = args[1]
        else:
            file_path = str(args[0]) if args[0] is not None else SESSION_HISTORY_FILE
            history = args[1]

        if len(args) >= 3:
            if isinstance(args[2], (int, float)):
                max_turns = int(args[2])
            elif hasattr(args[2], "__enter__") or hasattr(args[2], "acquire"):
                lock = args[2]

        if len(args) >= 4:
            if hasattr(args[3], "__enter__") or hasattr(args[3], "acquire"):
                lock = args[3]
    else:
        file_path = kwargs.get("file_path", kwargs.get("session_file", SESSION_HISTORY_FILE))
        history = kwargs.get("history", conversation_history)

    if file_path is None:
        file_path = SESSION_HISTORY_FILE
    if history is None:
        history = conversation_history

    file_path = str(file_path)

    # Snapshot to prevent concurrent mutation
    if isinstance(history, list):
        history_snapshot = list(history)
    else:
        history_snapshot = list(history) if history is not None else []

    # Apply sliding window retention limit (max_turns * 2 messages)
    history_slice = history_snapshot
    if max_turns is not None and max_turns > 0 and len(history_slice) > max_turns * 2:
        history_slice = history_slice[-max_turns * 2:]

    target_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(target_dir, exist_ok=True)
    unique_id = uuid.uuid4().hex
    tmp_path = f"{file_path}.{unique_id}.tmp"

    final_summary: Optional[str] = None
    if last_session_summary is not None and isinstance(last_session_summary, str) and last_session_summary.strip():
        final_summary = last_session_summary.strip()
    elif last_session_summary is None and preserve_existing_summary:
        try:
            existing_summary = load_session_summary(file_path, lock=lock)
            if existing_summary and isinstance(existing_summary, str) and existing_summary.strip():
                final_summary = existing_summary.strip()
        except Exception:
            pass

    payload = {
        "version": "1.0",
        "last_saved": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "retention_limit_turns": max_turns,
        "history": list(history_slice)
    }
    if final_summary:
        payload["last_session_summary"] = final_summary

    def _do_save():
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except (AttributeError, OSError):
                    pass

            max_retries = 20
            for attempt in range(max_retries):
                try:
                    os.replace(tmp_path, file_path)
                    break
                except (PermissionError, OSError):
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.005 * (2 ** min(attempt, 5)) + (uuid.uuid4().int % 15) * 0.001)

            try:
                abs_path = os.path.abspath(file_path)
                cur_mtime = os.path.getmtime(abs_path)
                with _session_cache_lock:
                    _session_summary_cache[abs_path] = (cur_mtime, final_summary)
            except OSError:
                pass
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    use_lock = lock
    if use_lock:
        with use_lock:
            _do_save()
    else:
        _do_save()


def load_session_history(*args, **kwargs) -> List[Dict[str, str]]:
    """
    Deserializes and validates conversation context from disk.
    - Missing or empty file returns [].
    - Corrupt JSON or malformed structure renames file to <file>.corrupted and returns [].
    - Filters damaged turns: preserves role in ('user', 'assistant', 'system') and non-empty content.
    - Applies sliding window retention limit (max_turns * 2 turns).
    - Windows file lock resistant (up to 5 retries with backoff).
    - Thread-safe under lock.
    """
    file_path = None
    max_turns = kwargs.get("max_turns", kwargs.get("retention_limit_turns", MAX_HISTORY))
    lock = kwargs.get("lock", None)

    if len(args) >= 1:
        file_path = args[0]
    else:
        file_path = kwargs.get("file_path", kwargs.get("session_file", kwargs.get("target_path", SESSION_HISTORY_FILE)))

    if len(args) >= 2:
        if isinstance(args[1], (int, float)):
            max_turns = int(args[1])
        elif hasattr(args[1], "__enter__") or hasattr(args[1], "acquire"):
            lock = args[1]

    if len(args) >= 3:
        if hasattr(args[2], "__enter__") or hasattr(args[2], "acquire"):
            lock = args[2]

    if file_path is None:
        file_path = SESSION_HISTORY_FILE

    target_path = str(file_path)

    def _do_load():
        if not target_path:
            return []
        try:
            if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
                return []
        except OSError:
            return []

        max_retries = 5
        data = None
        for attempt in range(max_retries):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                break
            except FileNotFoundError:
                return []
            except json.JSONDecodeError as jde:
                logging.warning(f"[SESSION] Corrupt JSON in {target_path}: {jde}")
                corrupted_path = f"{target_path}.corrupted"
                try:
                    os.replace(target_path, corrupted_path)
                except Exception:
                    pass
                return []
            except (PermissionError, OSError) as e:
                if attempt == max_retries - 1:
                    logging.warning(f"[SESSION] Failed to read {target_path} after {max_retries} retries: {e}")
                    return []
                time.sleep(0.005 * (2 ** attempt))

        if data is None:
            return []

        if not isinstance(data, (dict, list)):
            logging.warning(f"[SESSION] Invalid root type in {target_path}: {type(data).__name__}")
            corrupted_path = f"{target_path}.corrupted"
            try:
                os.replace(target_path, corrupted_path)
            except Exception:
                pass
            return []

        raw_history = data.get("history") if isinstance(data, dict) else data
        if not isinstance(raw_history, list):
            logging.warning(f"[SESSION] Field 'history' in {target_path} is not a list: {type(raw_history).__name__}")
            corrupted_path = f"{target_path}.corrupted"
            try:
                os.replace(target_path, corrupted_path)
            except Exception:
                pass
            return []

        valid_turns = []
        for item in raw_history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role in ("user", "assistant", "system") and isinstance(content, str) and content.strip():
                    clean_content = clean_artifacts(content) if role == "assistant" else content.strip()
                    if clean_content:
                        valid_turns.append({"role": role, "content": clean_content})

        if max_turns is not None and max_turns > 0 and len(valid_turns) > max_turns * 2:
            valid_turns = valid_turns[-max_turns * 2:]

        return valid_turns

    use_lock = lock
    if use_lock:
        with use_lock:
            return _do_load()
    else:
        return _do_load()


__all__ = [
    "save_session_history",
    "load_session_history",
    "conversation_history",
    "MAX_HISTORY",
    "session_new_messages",
    "record_session_message",
    "get_session_new_messages",
    "get_new_messages_count",
    "reset_session_tracker",
    "load_session_summary",
    "generate_session_summary",
]

