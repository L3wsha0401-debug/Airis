"""
Airis Companion Assistant - Main Interactive Entry Point.
"""

from __future__ import annotations
import sys
import os
import queue
import threading

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

import logging
import ollama
from colorama import Fore, Style

from airis.config import load_config
from airis.core.locks import ollama_lock, session_history_lock, chroma_lock
from airis.core.lifecycle import (
    shutdown_coordinator,
    register_signal_handlers,
    is_exit_command,
)
from airis.core.prompts import build_system_prompt
from airis.core.stream import (
    print_header,
    print_user_input_prompt,
    clean_stream_filter,
    clean_artifacts,
)
from airis.core.commands import handle_chat_command
from airis.tts import VoiceEngine, SentenceChunker
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
    """Coordinates concurrent CLI text input and STT voice input."""
    def __init__(self, stt_listener: STTListener | None = None):
        self.input_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self.stt_listener = stt_listener
        self.stop_event = threading.Event()

        if self.stt_listener:
            self.stt_listener.on_transcription = self._on_voice_text

        self._text_thread = threading.Thread(target=self._text_reader, name="TextReaderThread", daemon=True)
        self._text_thread.start()

    def _on_voice_text(self, text: str) -> None:
        if text and text.strip():
            self.input_queue.put(("voice", text.strip()))

    def _text_reader(self) -> None:
        while not self.stop_event.is_set():
            try:
                line = input()
                if line is not None:
                    self.input_queue.put(("text", line.strip()))
            except (EOFError, KeyboardInterrupt):
                self.input_queue.put(("exit", None))
                break
            except Exception:
                break

    def get_next_input(self, timeout: float | None = None) -> tuple[str, str | None]:
        return self.input_queue.get(timeout=timeout)

    def stop(self) -> None:
        self.stop_event.set()


def chat():
    """Main interactive terminal chat loop."""
    print_header()
    cfg = load_config()

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

    coordinator = None
    if stt_listener is not None:
        coordinator = InteractionInputCoordinator(stt_listener=stt_listener)

    while True:
        current_user = get_current_user_name()
        if coordinator is not None:
            try:
                if coordinator.input_queue.empty():
                    print(Fore.YELLOW + Style.BRIGHT + f"{current_user} ➔ " + Fore.RESET, end="", flush=True)
                source, raw_input = coordinator.get_next_input()
                if source == "exit":
                    print(Fore.CYAN + "\n\nЗавершение сеанса...")
                    break
                user_input = raw_input or ""
                if source == "voice":
                    print(Fore.YELLOW + Style.BRIGHT + f"\n{current_user} (голос) ➔ " + Fore.WHITE + user_input)
            except (KeyboardInterrupt, EOFError):
                print(Fore.CYAN + "\n\nЗавершение сеанса...")
                break
        else:
            try:
                user_input = print_user_input_prompt(current_user)
            except (KeyboardInterrupt, EOFError):
                print(Fore.CYAN + "\n\nЗавершение сеанса...")
                break

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

        try:
            with ollama_lock:
                raw_stream = ollama.chat(
                    model=cfg.ollama.model,
                    messages=messages,
                    options=cfg.ollama.options,
                    stream=True,
                )

                for clean_chunk in clean_stream_filter(raw_stream):
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

    if coordinator:
        coordinator.stop()
    if stt_listener:
        stt_listener.stop()

    shutdown_coordinator.trigger(
        voice_engine=voice_engine,
        memory_queue=memory_queue,
        memory_thread=memory_thread,
        memory_stop_event=memory_stop_event,
        session_file=session_file,
        history=conversation_history,
        client=chroma_client,
        logs_dir=cfg.logging.logs_dir,
    )


if __name__ == "__main__":
    chat()
