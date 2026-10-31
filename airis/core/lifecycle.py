"""
Airis Application Lifecycle Coordination.
Provides thread-safe atomic shutdown coordination, signal handlers, and clean teardown.
"""

from __future__ import annotations
import sys
import os
import time
import signal
import atexit
import logging
import threading
import datetime
from typing import Optional, Any


def is_exit_command(user_input: str) -> bool:
    """
    Checks if user input is an exit command (exit, выход, quit, :q, /exit, пока).
    """
    if not isinstance(user_input, str):
        return False
    return user_input.strip().lower() in ("exit", "выход", "quit", ":q", "/exit", "пока")


def graceful_shutdown(
    voice_engine=None,
    memory_queue=None,
    memory_thread=None,
    memory_stop_event=None,
    session_file=None,
    history=None,
    client=None,
    logs_dir=None,
    timeout: float = 8.0,
    _save_session_fn=None,
    _dump_memory_fn=None,
    avatar_controller=None,
    **kwargs
) -> None:
    """
    Coordinates orderly process termination:
    1. Drains and stops audio queue of voice_engine.
    2. Waits for memory_queue tasks to complete, signals memory_stop_event, and joins memory_thread.
    3. Atomically saves session history to session_file.
    4. Dumps memory snapshot to Logs/iris_memory.log.
    5. Shuts down ChromaDB system client.
    """
    from airis.memory.session import save_session_history
    from airis.memory.logger import dump_memory_log

    # 0. Stop avatar controller if active
    if avatar_controller and hasattr(avatar_controller, "stop") and callable(avatar_controller.stop):
        try:
            avatar_controller.stop(timeout=timeout)
        except Exception:
            pass

    # 1. Drain and stop voice playback
    if voice_engine and hasattr(voice_engine, "audio_queue") and voice_engine.audio_queue is not None:
        if hasattr(voice_engine, "stop") and callable(voice_engine.stop):
            try:
                voice_engine.stop()
            except Exception:
                pass
        t0 = time.time()
        while not voice_engine.audio_queue.empty() and (time.time() - t0) < timeout:
            try:
                voice_engine.audio_queue.get_nowait()
                voice_engine.audio_queue.task_done()
            except Exception:
                break
    elif voice_engine and hasattr(voice_engine, "stop") and callable(voice_engine.stop):
        try:
            voice_engine.stop()
        except Exception:
            pass

    # 2. Stop memory worker and drain queue
    end_drain = time.time() + timeout
    if memory_queue is not None:
        if memory_thread and hasattr(memory_thread, "is_alive") and memory_thread.is_alive():
            while getattr(memory_queue, "unfinished_tasks", 0) > 0 and time.time() < end_drain and memory_thread.is_alive():
                time.sleep(0.05)
        else:
            while not memory_queue.empty() and time.time() < end_drain:
                try:
                    memory_queue.get_nowait()
                    memory_queue.task_done()
                except Exception:
                    break

    if memory_stop_event is not None:
        try:
            memory_stop_event.set()
        except Exception:
            pass

    if memory_thread and hasattr(memory_thread, "is_alive") and memory_thread.is_alive():
        try:
            remaining = max(0.1, end_drain - time.time())
            memory_thread.join(timeout=remaining)
        except Exception:
            pass

    if memory_queue is not None and not getattr(memory_queue, "empty", lambda: True)():
        while not memory_queue.empty():
            try:
                memory_queue.get_nowait()
                memory_queue.task_done()
            except Exception:
                break

    # 2.5 Generate or preserve session summary
    new_msgs = kwargs.get("new_messages", None)
    if new_msgs is None:
        try:
            from airis.memory.session import get_session_new_messages
            new_msgs = get_session_new_messages()
        except Exception:
            new_msgs = []

    new_count = kwargs.get("new_messages_count", None)
    if new_count is None:
        new_count = len(new_msgs) if new_msgs is not None else 0

    session_summary = None
    if new_count >= 4 and session_file:
        try:
            from airis.memory.session import generate_session_summary
            from airis.memory.chroma import get_current_user_name
            from airis.core.locks import ollama_lock
            user_name = kwargs.get("user_name") or get_current_user_name()
            summary_timeout = kwargs.get("summary_timeout", max(2.0, min(6.0, timeout - 2.0)))
            session_summary = generate_session_summary(
                new_msgs,
                user_name=user_name,
                timeout=summary_timeout,
                lock=kwargs.get("ollama_lock", ollama_lock),
            )
        except Exception as e:
            logging.warning(f"[LIFECYCLE] Error generating session summary on shutdown: {e}")

    # 3. Persist session history
    if session_file and history is not None:
        try:
            save_fn = _save_session_fn or save_session_history
            try:
                save_fn(session_file, history, last_session_summary=session_summary)
            except TypeError:
                save_fn(session_file, history)
        except Exception as e:
            logging.error(f"[LIFECYCLE] Error saving session history to {session_file}: {e}")

    # 4. Dump memory snapshot to log file
    if logs_dir:
        try:
            dump_fn = _dump_memory_fn or dump_memory_log
            dump_fn(logs_dir=logs_dir)
        except Exception as e:
            try:
                os.makedirs(logs_dir, exist_ok=True)
                dump_path = os.path.join(logs_dir, "iris_memory.log")
                with open(dump_path, "a", encoding="utf-8") as f:
                    f.write(f"\n--- ДАМП ПАМЯТИ НА КОНЕЦ СЕССИИ [{datetime.datetime.now().isoformat()}] ---\n")
            except Exception:
                pass

    # 5. Release ChromaDB client resources
    if client and hasattr(client, "_system"):
        try:
            client._system.stop()
        except Exception:
            pass


class ShutdownCoordinator:
    """
    Thread-safe atomic lifecycle coordinator ensuring single execution of
    teardown procedures even under concurrent signals, exit commands, or atexit.
    """
    def __init__(self, shutdown_fn=graceful_shutdown, default_timeout: float = 8.0):
        self._shutdown_fn = shutdown_fn
        self._default_timeout = default_timeout
        self._is_shutting_down = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._execution_count = 0

    def trigger(self, **kwargs) -> bool:
        """
        Executes shutdown procedure if not already triggered.
        Returns True if this invocation performed the teardown, False if redundant.
        """
        with self._shutdown_lock:
            if self._is_shutting_down.is_set():
                return False
            self._is_shutting_down.set()
            self._execution_count += 1

        if self._shutdown_fn and callable(self._shutdown_fn):
            if "timeout" not in kwargs:
                kwargs["timeout"] = self._default_timeout
            self._shutdown_fn(**kwargs)
        return True

    @property
    def execution_count(self) -> int:
        return self._execution_count

    @property
    def is_shutting_down(self) -> bool:
        return self._is_shutting_down.is_set()


# Default singleton coordinator
shutdown_coordinator = ShutdownCoordinator(shutdown_fn=graceful_shutdown, default_timeout=8.0)


def register_signal_handlers(coordinator: Optional[ShutdownCoordinator] = None, **shutdown_kwargs):
    """
    Safely registers SIGINT, SIGTERM, and atexit hooks.
    Windows and secondary thread safe.
    """
    coord = coordinator if coordinator is not None else shutdown_coordinator

    def _do_shutdown(source="signal"):
        logging.info(f"[LIFECYCLE] Shutdown triggered via {source}")
        coord.trigger(**shutdown_kwargs)

    if threading.current_thread() is threading.main_thread():
        try:
            def _signal_handler(sig, frame):
                logging.info(f"[LIFECYCLE] Caught signal {sig}")
                _do_shutdown(source=f"signal_{sig}")
                sys.exit(0)

            signal.signal(signal.SIGINT, _signal_handler)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _signal_handler)
        except (ValueError, OSError) as e:
            logging.debug(f"[LIFECYCLE] Skipping signal handler registration: {e}")

    try:
        atexit.register(lambda: _do_shutdown(source="atexit"))
    except Exception as e:
        logging.debug(f"[LIFECYCLE] Skipping atexit registration: {e}")
