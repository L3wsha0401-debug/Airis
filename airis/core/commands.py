"""
Airis Interactive Chat Command Router.
Handles console commands such as /mute, /memories, /forget <query>, and /name <name>.
"""

from __future__ import annotations
import sys
import logging
from typing import Tuple, Optional, Any
from colorama import Fore, Style

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _safe_print(text: str = "", end: str = "\n") -> None:
    """Prints safely across all terminal encodings (e.g. Windows cp1252/cp866)."""
    try:
        print(text, end=end)
    except (UnicodeEncodeError, OSError):
        try:
            cleaned = text.encode("ascii", errors="backslashreplace").decode("ascii")
            print(cleaned, end=end)
        except Exception:
            pass


def handle_chat_command(
    user_input: str,
    voice_engine: Optional[Any] = None,
    collection: Optional[Any] = None,
    chroma_lock: Optional[Any] = None
) -> Tuple[bool, str]:
    """
    Evaluates and dispatches interactive slash commands:
    /mute, /memories, /forget <query>, /name <name>, /help, /?.
    Returns:
        Tuple[bool, str]: (handled, message)
    """
    if not user_input or not isinstance(user_input, str):
        return False, ""

    raw = user_input.strip()
    if not raw:
        return False, ""

    lower_raw = raw.lower()
    if not raw.startswith("/") and not lower_raw.startswith("!имя"):
        return False, ""

    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    # 1. /mute command: toggles TTS playback
    if cmd == "/mute":
        if voice_engine and hasattr(voice_engine, "toggle_mute"):
            is_muted = voice_engine.toggle_mute()
            if is_muted:
                msg = "  🔇 Звук отключен (Muted). Айрис отвечает только текстом.\n"
                _safe_print(Fore.YELLOW + msg)
            else:
                msg = "  🔊 Звук включен (Unmuted). Голосовой синтез активен.\n"
                _safe_print(Fore.GREEN + msg)
        else:
            msg = "  ⚠ Голосовой движок отключен или не доступен.\n"
            _safe_print(Fore.YELLOW + msg)
        return True, msg

    # 2. /memories, /memory, /память command: formatted memory listing
    if cmd in ("/memories", "/memory", "/память"):
        from airis.memory.chroma import list_stored_memories, get_default_collection
        from airis.core.locks import chroma_lock as default_lock
        col = collection if collection is not None else get_default_collection()
        use_lock = chroma_lock if chroma_lock is not None else default_lock
        mem_data = list_stored_memories(collection_obj=col, lock=use_lock)

        _safe_print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 58)
        _safe_print(Fore.CYAN + Style.BRIGHT + "  🌸 Хранилище воспоминаний Айрис (ChromaDB)")
        _safe_print(Fore.CYAN + Style.BRIGHT + "=" * 58)
        _safe_print(Fore.YELLOW + f"  👤 Имя пользователя: {mem_data['user_name']}\n")

        user_facts = mem_data["user_facts"]
        _safe_print(Fore.MAGENTA + f"  📚 Факты о пользователе ({len(user_facts)}):")
        if user_facts:
            for idx, (fid, ftext) in enumerate(user_facts, 1):
                _safe_print(Fore.WHITE + f"    {idx}. [{fid[:8]}] {ftext}")
        else:
            _safe_print(Fore.LIGHTBLACK_EX + "    (нет сохраненных фактов)")

        self_facts = mem_data["self_facts"]
        _safe_print(Fore.MAGENTA + f"\n  ✨ Факты об Айрис ({len(self_facts)}):")
        if self_facts:
            for idx, (fid, ftext) in enumerate(self_facts, 1):
                _safe_print(Fore.WHITE + f"    {idx}. [{fid[:8]}] {ftext}")
        else:
            _safe_print(Fore.LIGHTBLACK_EX + "    (нет сохраненных фактов)")

        _safe_print(Fore.CYAN + Style.BRIGHT + "=" * 58)
        hint = f"  Всего: {mem_data['total']}. Для удаления: /forget <текст>\n"
        _safe_print(Fore.LIGHTYELLOW_EX + hint)
        return True, "memories_listed"

    # 3. /forget <query> command: locate and remove matching memory
    if cmd == "/forget":
        if not arg:
            msg = "  ⚠ Использование: /forget <описание факта или фраза>\n  Пример: /forget печенье\n  Подсказка: введите /memories для просмотра списка всех фактов.\n"
            _safe_print(Fore.YELLOW + msg)
            return True, msg
        from airis.memory.chroma import forget_memory, get_default_collection
        from airis.core.locks import chroma_lock as default_lock
        col = collection if collection is not None else get_default_collection()
        use_lock = chroma_lock if chroma_lock is not None else default_lock
        success, details = forget_memory(arg, collection_obj=col, lock=use_lock)
        if success:
            msg = f"  ✓ {details}\n"
            _safe_print(Fore.GREEN + msg)
        else:
            msg = f"  ⚠ {details}\n"
            _safe_print(Fore.YELLOW + msg)
        return True, msg

    # 4. /name or !имя command: update user identity
    if cmd in ("/name", "!имя"):
        if not arg:
            msg = "  ⚠ Использование: /name <НовоеИмя>\n"
            _safe_print(Fore.YELLOW + msg)
            return True, msg
        from airis.memory.chroma import maybe_store_identity_fact, get_default_collection
        from airis.core.locks import chroma_lock as default_lock
        col = collection if collection is not None else get_default_collection()
        use_lock = chroma_lock if chroma_lock is not None else default_lock
        success = maybe_store_identity_fact(f"/name {arg}", collection_obj=col, lock=use_lock)
        msg = f"Имя обновлено: {arg}" if success else "Ошибка обновления имени"
        return True, msg

    # 5. /help or /? command: list commands
    if cmd in ("/help", "/?"):
        _safe_print(Fore.CYAN + "\n  Доступные команды:")
        _safe_print(Fore.WHITE + "    /mute            — Включить/выключить голос Айрис")
        _safe_print(Fore.WHITE + "    /memories        — Просмотреть факты в памяти")
        _safe_print(Fore.WHITE + "    /forget <текст>  — Найти и удалить факт из памяти")
        _safe_print(Fore.WHITE + "    /name <Имя>      — Изменить имя собеседника")
        _safe_print(Fore.WHITE + "    /help, /?        — Показать справку по командам")
        _safe_print(Fore.WHITE + "    exit, выход      — Завершить диалог\n")
        return True, "help_listed"

    return False, ""
