"""
Airis Companion Assistant - Main Interactive Entry Point.
"""

from __future__ import annotations
import sys
import os
import queue
import threading
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import time
import logging
import ollama
from colorama import Fore, Style

from airis.config import load_config, IdleConfig
from airis.core.locks import ollama_lock, session_history_lock, chroma_lock
from airis.core.lifecycle import (
    shutdown_coordinator,
    register_signal_handlers,
    is_exit_command,
)
from airis.core.prompts import (
    build_system_prompt,
    load_system_template,
    SystemPromptNotFoundError,
)
from airis.core.idle import IdleCoordinator, build_idle_prompt

from airis.core.stream import (
    print_header,
    print_user_input_prompt,
    clean_stream_filter,
    clean_artifacts,
    format_emotion_badge,
    EmotionStreamParser,
)
from airis.core.commands import handle_chat_command
from airis.tts import VoiceEngine, SentenceChunker
from airis.avatar import AvatarController
from airis.memory import (
    collection,
    chroma_client,
    get_current_user_name,
    maybe_store_identity_fact,
    load_session_history,
    save_session_history,
    conversation_history,
    memory_queue,
    memory_thread,
    memory_stop_event,
    log_chat_interaction,
)
from airis.stt import STTListener


class InteractionInputCoordinator:
    """Coordinates concurrent CLI text input, STT voice input, and autonomous idle initiative."""
    def __init__(
        self,
        stt_listener: STTListener | None = None,
        voice_engine: VoiceEngine | None = None,
        idle_coordinator: IdleCoordinator | None = None,
        idle_config: IdleConfig | None = None,
        barge_in_controller: Any | None = None,
        abort_generation_event: threading.Event | None = None,
        start_thread: bool = True,
    ):
        self.input_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self.stt_listener = stt_listener
        self.voice_engine = voice_engine
        self.barge_in_controller = barge_in_controller
        self.abort_generation_event = abort_generation_event
        self.stop_event = threading.Event()

        if idle_coordinator is not None:
            self.idle = idle_coordinator
        elif idle_config is not None:
            self.idle = IdleCoordinator(
                config=idle_config,
                voice_engine=voice_engine,
                stt_listener=stt_listener,
            )
        else:
            self.idle = None
        self.idle_coordinator = self.idle

        if self.stt_listener:
            self.stt_listener.on_transcription = self._on_voice_text

        self._text_thread = None
        if start_thread:
            self._text_thread = threading.Thread(target=self._text_reader, name="TextReaderThread", daemon=True)
            self._text_thread.start()

    def _on_voice_text(self, text: str) -> None:
        if text and text.strip():
            if self.idle:
                self.idle.record_user_activity()
            self.input_queue.put(("voice", text.strip()))

    def _text_reader(self) -> None:
        while not self.stop_event.is_set():
            try:
                line = input()
                if line is not None:
                    stripped = line.strip()
                    if stripped:
                        if self.barge_in_controller:
                            self.barge_in_controller.trigger_barge_in()
                        elif self.abort_generation_event:
                            self.abort_generation_event.set()
                            if self.voice_engine:
                                self.voice_engine.interrupt()
                    if self.idle:
                        self.idle.record_user_activity()
                    self.input_queue.put(("text", stripped))
            except (EOFError, KeyboardInterrupt):
                self.input_queue.put(("exit", None))
                break
            except Exception:
                break

    def get_next_input(self, timeout: float | None = None) -> tuple[str, str | None]:
        deadline = (time.time() + timeout) if timeout is not None else None
        while not self.stop_event.is_set():
            poll_timeout = 0.05
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise queue.Empty
                poll_timeout = min(0.05, remaining)

            try:
                return self.input_queue.get(timeout=poll_timeout)
            except queue.Empty:
                if self.idle and self.idle.check_idle_timeout():
                    return ("idle", None)
                if deadline is not None and time.time() >= deadline:
                    raise queue.Empty

        return ("exit", None)

    def stop(self) -> None:
        self.stop_event.set()


def execute_idle_turn(
    cfg: Any = None,
    conversation_history: list[dict[str, str]] | None = None,
    session_history_lock: threading.Lock | None = None,
    abort_generation_event: threading.Event | None = None,
    voice_engine: VoiceEngine | None = None,
    coordinator: InteractionInputCoordinator | None = None,
    current_user: str = "l3wsha",
    collection: Any | None = None,
    chroma_lock: threading.Lock | None = None,
    session_file: str | None = None,
    max_history: int = 10,
    ollama_lock: threading.Lock | None = None,
    avatar_controller: Any | None = None,
) -> bool:
    """
    Executes an autonomous spontaneous idle chatter turn.

    Lifecycle:
    1. Pre-condition Check 1: Aborts immediately if user input is queued or barge-in is set.
    2. Builds dynamic idle prompt and Neuro-sama style initiative prompt via build_idle_prompt().
    3. Pre-condition Check 2: Aborts if user input arrived during ChromaDB memory retrieval.
    4. Clears abort_generation_event only after verifying queue is empty.
    5. Streams tokens from Ollama through clean_stream_filter and SentenceChunker.
    6. Intercepts conversational barge-in mid-stream, halts VoiceEngine, discards partial
       utterance from conversation_history, and records user activity.
    7. On clean completion, appends assistant turn to conversation_history, persists session,
       and records assistant activity.

    Returns:
        bool: True if an idle turn was cleanly completed and persisted;
              False if aborted (barge-in, pending user input, empty response, or error).
    """
    if cfg is None:
        cfg = load_config()
    if conversation_history is None:
        conversation_history = []
    if session_history_lock is None:
        session_history_lock = threading.Lock()
    if abort_generation_event is None:
        abort_generation_event = threading.Event()
    if ollama_lock is None:
        from airis.core.locks import ollama_lock as _ollama_lock
        ollama_lock = _ollama_lock

    if avatar_controller is None and voice_engine is not None and getattr(voice_engine, "avatar_controller", None) is not None:
        avatar_controller = voice_engine.avatar_controller

    # Pre-condition check 1: Abort if user input arrived before dispatch
    if (coordinator is not None and not coordinator.input_queue.empty()) or abort_generation_event.is_set():
        if coordinator is not None and getattr(coordinator, "idle", None):
            coordinator.idle.record_user_activity()
        return False

    dynamic_system_prompt, internal_trigger = build_idle_prompt(
        conversation_history=conversation_history,
        user_name=current_user,
        collection_obj=collection,
        lock=chroma_lock,
    )

    # Pre-condition check 2: Abort if user input arrived during Chroma memory retrieval
    if (coordinator is not None and not coordinator.input_queue.empty()) or abort_generation_event.is_set():
        if coordinator is not None and getattr(coordinator, "idle", None):
            coordinator.idle.record_user_activity()
        return False

    with session_history_lock:
        history_snapshot = list(conversation_history)

    messages = [{'role': 'system', 'content': dynamic_system_prompt}]
    messages.extend(history_snapshot)
    messages.append({'role': 'user', 'content': internal_trigger})

    print(Fore.MAGENTA + Style.BRIGHT + "\nАйрис: " + Fore.WHITE, end="", flush=True)

    full_response = ""
    chunker = SentenceChunker() if voice_engine else None
    abort_generation_event.clear()

    badge_printed = False

    def on_emotion_detected(emotion: str):
        nonlocal badge_printed
        if not badge_printed:
            sys.stdout.write(format_emotion_badge(emotion))
            sys.stdout.flush()
            badge_printed = True
        if avatar_controller:
            try:
                avatar_controller.set_emotion(emotion)
            except Exception:
                pass

    parser = EmotionStreamParser(on_emotion=on_emotion_detected)

    try:
        model_name = getattr(getattr(cfg, "ollama", None), "model", "gemma2:9b")
        options = getattr(getattr(cfg, "ollama", None), "options", {})
        with ollama_lock:
            raw_stream = ollama.chat(
                model=model_name,
                messages=messages,
                options=options,
                stream=True,
            )

            filtered_stream = clean_stream_filter(raw_stream)
            for clean_chunk in parser.process_stream(filtered_stream):
                if abort_generation_event.is_set() or (coordinator is not None and not coordinator.input_queue.empty()):
                    if not abort_generation_event.is_set():
                        abort_generation_event.set()
                    print(Fore.YELLOW + "\n[Айрис прервана пользователем]")
                    break
                full_response += clean_chunk
                sys.stdout.write(clean_chunk)
                sys.stdout.flush()
                if voice_engine and chunker:
                    for sentence in chunker.feed(clean_chunk):
                        if abort_generation_event.is_set() or (coordinator is not None and not coordinator.input_queue.empty()):
                            if not abort_generation_event.is_set():
                                abort_generation_event.set()
                            break
                        voice_engine.feed_sentence(sentence)

        print("\n")
    except Exception as e:
        print(Fore.RED + f"\n[Ошибка генерации]: {e}\n")
        if voice_engine:
            voice_engine.flush()
        return False

    if not abort_generation_event.is_set() and voice_engine and chunker:
        for sentence in chunker.flush():
            if abort_generation_event.is_set() or (coordinator is not None and not coordinator.input_queue.empty()):
                if not abort_generation_event.is_set():
                    abort_generation_event.set()
                break
            voice_engine.feed_sentence(sentence)

    if not abort_generation_event.is_set() and voice_engine:
        voice_engine.flush()

    # Conversational Barge-In interruption check: discard partial idle speech
    if abort_generation_event.is_set() or (coordinator is not None and not coordinator.input_queue.empty()):
        if not abort_generation_event.is_set():
            abort_generation_event.set()
        if voice_engine:
            voice_engine.interrupt()
        if coordinator is not None and getattr(coordinator, "idle", None):
            coordinator.idle.record_user_activity()
        return False

    cleaned_ai_response = clean_artifacts(full_response)
    if not cleaned_ai_response:
        return False

    if coordinator is not None and getattr(coordinator, "idle", None):
        coordinator.idle.record_assistant_activity()

    with session_history_lock:
        conversation_history.append({'role': 'assistant', 'content': cleaned_ai_response})
        if len(conversation_history) > max_history * 2:
            conversation_history[:] = conversation_history[-max_history * 2:]
        history_to_save = list(conversation_history)

    if session_file:
        try:
            save_session_history(session_file, history_to_save, max_turns=max_history, lock=session_history_lock)
        except Exception as e:
            logging.error(f"[SESSION] Ошибка автосохранения истории сессии: {e}")

    chat_log_file = getattr(getattr(cfg, "logging", None), "chat_log_file", None)
    log_chat_interaction(current_user, "[Автономная реплика]", cleaned_ai_response, chat_log_file=chat_log_file)
    return True


def chat():
    """Main interactive terminal chat loop."""
    print_header()
    cfg = load_config()

    try:
        load_system_template()
    except SystemPromptNotFoundError as e:
        print(Fore.RED + f"\n[ОШИБКА] {e}" + Style.RESET_ALL)
        sys.exit(1)

    avatar_controller = None
    if getattr(cfg, "avatar", None) and cfg.avatar.enabled:
        try:
            avatar_controller = AvatarController(cfg.avatar)
            avatar_controller.start()
        except Exception as e:
            logging.warning(f"[Avatar] AvatarController initialization warning: {e}")
            avatar_controller = None

    voice_engine = None
    if cfg.tts.enabled:
        try:
            voice_engine = VoiceEngine(
                model_path=cfg.tts.model_path,
                speaker=cfg.tts.speaker,
                pitch_shift=cfg.tts.pitch_shift,
                sample_rate=cfg.tts.sample_rate,
                lazy_model=cfg.tts.lazy_model,
                muted=cfg.tts.muted,
                avatar_controller=avatar_controller,
            )
        except Exception as e:
            logging.warning(f"[TTS] VoiceEngine initialization warning: {e}")
            voice_engine = None

    # Barge-In abort event for interrupting active LLM streaming generation
    abort_generation_event = threading.Event()

    stt_listener = None
    if getattr(cfg, "stt", None) and cfg.stt.enabled:
        try:
            stt_listener = STTListener(
                voice_engine=voice_engine,
                input_mode=cfg.stt.input_mode,
                hotkey=cfg.stt.ptt_hotkey,
                abort_generation_event=abort_generation_event,
                model_name=cfg.stt.model_name,
                cpu_threads=cfg.stt.cpu_threads,
                vad_threshold=cfg.stt.vad_threshold,
                vad_threshold_speaking=cfg.stt.vad_threshold_speaking,
                vad_silence_timeout_s=cfg.stt.vad_silence_timeout_s,
                energy_floor=cfg.stt.energy_floor,
            )
            stt_listener.start()
        except Exception as e:
            logging.warning(f"[STT] STTListener initialization warning: {e}")
            stt_listener = None

    # Restore session history
    session_file = cfg.session.session_history_file
    max_history = cfg.session.max_history_turns
    loaded_history = load_session_history(session_file, max_turns=max_history, lock=session_history_lock)
    with session_history_lock:
        conversation_history.clear()
        conversation_history.extend(loaded_history)
        if conversation_history:
            print(Fore.CYAN + f"  🌸 Восстановлена история диалога: {len(conversation_history)} сообщений.\n")

    # Register lifecycle hooks
    register_signal_handlers(
        coordinator=shutdown_coordinator,
        voice_engine=voice_engine,
        memory_queue=memory_queue,
        memory_thread=memory_thread,
        memory_stop_event=memory_stop_event,
        session_file=session_file,
        history=conversation_history,
        client=chroma_client,
        logs_dir=cfg.logging.logs_dir,
    )

    idle_coordinator = IdleCoordinator(
        config=getattr(cfg, "idle", None),
        voice_engine=voice_engine,
        stt_listener=stt_listener,
    )

    coordinator = InteractionInputCoordinator(
        stt_listener=stt_listener,
        voice_engine=voice_engine,
        idle_coordinator=idle_coordinator,
        barge_in_controller=stt_listener.barge_in_controller if stt_listener else None,
        abort_generation_event=abort_generation_event,
    )

    while True:
        current_user = get_current_user_name()
        try:
            if coordinator.input_queue.empty():
                print(Fore.YELLOW + Style.BRIGHT + f"{current_user} ➔ " + Fore.RESET, end="", flush=True)
            source, raw_input = coordinator.get_next_input()
            if source == "exit":
                print(Fore.CYAN + "\n\nЗавершение сеанса...")
                break
            if source == "voice":
                user_input = raw_input or ""
                print(Fore.YELLOW + Style.BRIGHT + f"\n{current_user} (голос) ➔ " + Fore.WHITE + user_input)
            elif source == "text":
                user_input = raw_input or ""
            elif source == "idle":
                user_input = ""
            else:
                user_input = raw_input or ""
        except (KeyboardInterrupt, EOFError):
            print(Fore.CYAN + "\n\nЗавершение сеанса...")
            break

        if source == "idle":
            execute_idle_turn(
                cfg=cfg,
                conversation_history=conversation_history,
                session_history_lock=session_history_lock,
                abort_generation_event=abort_generation_event,
                voice_engine=voice_engine,
                coordinator=coordinator,
                current_user=current_user,
                collection=collection,
                chroma_lock=chroma_lock,
                session_file=session_file,
                max_history=max_history,
                avatar_controller=avatar_controller,
            )
            continue

        if is_exit_command(user_input):
            print(Fore.CYAN + "\nДо связи! Увидимся в следующем запуске.")
            break

        if not user_input.strip():
            continue

        # Check interactive commands (/mute, /memories, /forget, /name)
        handled, _ = handle_chat_command(user_input, voice_engine=voice_engine, collection=collection, chroma_lock=chroma_lock)
        if handled:
            continue

        if maybe_store_identity_fact(user_input):
            continue

        dynamic_system_prompt = build_system_prompt(user_input, current_user)

        with session_history_lock:
            history_snapshot = list(conversation_history)

        messages = [{'role': 'system', 'content': dynamic_system_prompt}]
        messages.extend(history_snapshot)
        messages.append({'role': 'user', 'content': user_input})

        print(Fore.MAGENTA + Style.BRIGHT + "Айрис: " + Fore.WHITE, end="", flush=True)

        full_response = ""
        chunker = SentenceChunker() if voice_engine else None
        abort_generation_event.clear()

        badge_printed = False

        def on_emotion_detected(emotion: str):
            nonlocal badge_printed
            if not badge_printed:
                sys.stdout.write(format_emotion_badge(emotion))
                sys.stdout.flush()
                badge_printed = True
            if avatar_controller:
                try:
                    avatar_controller.set_emotion(emotion)
                except Exception:
                    pass

        parser = EmotionStreamParser(on_emotion=on_emotion_detected)

        try:
            with ollama_lock:
                raw_stream = ollama.chat(
                    model=cfg.ollama.model,
                    messages=messages,
                    options=cfg.ollama.options,
                    stream=True,
                )

                filtered_stream = clean_stream_filter(raw_stream)
                for clean_chunk in parser.process_stream(filtered_stream):
                    if abort_generation_event.is_set():
                        print(Fore.YELLOW + "\n[Айрис прервана пользователем]")
                        break
                    full_response += clean_chunk
                    sys.stdout.write(clean_chunk)
                    sys.stdout.flush()
                    if voice_engine and chunker:
                        for sentence in chunker.feed(clean_chunk):
                            if abort_generation_event.is_set():
                                break
                            voice_engine.feed_sentence(sentence)

            print("\n")
        except Exception as e:
            print(Fore.RED + f"\n[Ошибка генерации]: {e}\n")
            if voice_engine:
                voice_engine.flush()
            continue

        if not abort_generation_event.is_set() and voice_engine and chunker:
            for sentence in chunker.flush():
                if abort_generation_event.is_set():
                    break
                voice_engine.feed_sentence(sentence)

        if not abort_generation_event.is_set() and voice_engine:
            voice_engine.flush()

        cleaned_ai_response = clean_artifacts(full_response)

        memory_queue.put(("user", user_input))
        if cleaned_ai_response:
            memory_queue.put(("self", cleaned_ai_response))

        if coordinator.idle:
            coordinator.idle.record_assistant_activity()

        with session_history_lock:
            conversation_history.append({'role': 'user', 'content': user_input})
            conversation_history.append({'role': 'assistant', 'content': cleaned_ai_response})

            if len(conversation_history) > max_history * 2:
                conversation_history[:] = conversation_history[-max_history * 2:]
            history_to_save = list(conversation_history)

        try:
            save_session_history(session_file, history_to_save, max_turns=max_history, lock=session_history_lock)
        except Exception as e:
            logging.error(f"[SESSION] Ошибка автосохранения истории сессии: {e}")

        log_chat_interaction(current_user, user_input, cleaned_ai_response, chat_log_file=cfg.logging.chat_log_file)

    coordinator.stop()
    if stt_listener:
        stt_listener.stop()
    if avatar_controller:
        try:
            avatar_controller.stop()
        except Exception:
            pass

    shutdown_coordinator.trigger(
        voice_engine=voice_engine,
        memory_queue=memory_queue,
        memory_thread=memory_thread,
        memory_stop_event=memory_stop_event,
        session_file=session_file,
        history=conversation_history,
        client=chroma_client,
        logs_dir=cfg.logging.logs_dir,
        avatar_controller=avatar_controller,
    )


__all__ = [
    "InteractionInputCoordinator",
    "execute_idle_turn",
    "chat",
]


if __name__ == "__main__":
    chat()
