"""
Airis Session History Persistence Module.
Provides atomic serialization and deserialization of dialogue turns with Windows file-lock mitigation.
"""

from __future__ import annotations
import os
import json
import uuid
import time
import datetime
import logging
from typing import List, Dict, Optional, Any

from airis.config import load_config, SESSION_HISTORY_FILE
from airis.core.locks import session_history_lock
from airis.core.stream import clean_artifacts

_cfg = load_config()
MAX_HISTORY = _cfg.memory.max_history_turns
conversation_history: List[Dict[str, str]] = []


def save_session_history(*args, **kwargs) -> None:
    """
    Atomically saves session history to disk with metadata and os.replace.
    Resilient to WinError 5 / 32 on Windows via unique temp files, f.flush(),
    os.fsync(), exponential retry backoff (up to 8 attempts), and clean tmp cleanup.

    Supports polymorphic calls:
      1. save_session_history(file_path, history, max_turns=12, lock=None)
      2. save_session_history(history, file_path=SESSION_HISTORY_FILE, max_turns=12, lock=None)
      3. save_session_history(history, file_path, max_turns=12, lock=None)
      4. save_session_history()  # uses conversation_history and SESSION_HISTORY_FILE
      5. Keyword arguments: session_file, history, max_turns, retention_limit_turns, lock.
    """
    file_path = None
    history = None
    max_turns = kwargs.get("max_turns", kwargs.get("retention_limit_turns", MAX_HISTORY))
    lock = kwargs.get("lock", None)

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

    payload = {
        "version": "1.0",
        "last_saved": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "retention_limit_turns": max_turns,
        "history": list(history_slice)
    }

    def _do_save():
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except (AttributeError, OSError):
                    pass

            max_retries = 8
            for attempt in range(max_retries):
                try:
                    os.replace(tmp_path, file_path)
                    break
                except (PermissionError, OSError):
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.005 * (2 ** attempt))
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
