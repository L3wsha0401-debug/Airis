"""
Airis Streaming Voice Engine.
Executes asynchronous 2-stage speech synthesis and playback pipeline with
hardware-safe 48 kHz output, dynamic mute support, and conversational barge-in interruption.
"""

from __future__ import annotations
import os
import time
import queue
import logging
import threading
from typing import Optional, Union
import numpy as np
import torch
import sounddevice as sd
from colorama import Fore

from airis.tts.normalizer import clean_and_normalize_tts_text
from airis.tts.chunker import SentenceChunker
from airis.tts.resampler import resample_pitch_audio


class VoiceEngine:
    """
    Streaming Voice Engine executing asynchronous 2-stage synthesis and playback.
    Guarantees fixed 48 kHz audio playback rate via software pitch resampling.
    Supports on-the-fly muting and low-latency barge-in playback termination.
    """
    def __init__(
        self,
        model_path: str = r"C:\LLM\Instruments\model.pt",
        speaker: str = "baya",
        pitch_shift: float = 1.25,
        sample_rate: int = 48000,
        lazy_model: bool = True,
        muted: bool = False
    ):
        self.model_path = model_path
        self.speaker = speaker
        self.pitch_shift = pitch_shift
        self.sample_rate = sample_rate
        self.device = torch.device("cpu")
        self.model = None
        self._muted = bool(muted)
        self._mute_lock = threading.Lock()
        self._generation_id: int = 0
        self._epoch_lock = threading.Lock()

        self.sentence_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.driver_error_count = 0
        self.stop_event = threading.Event()
        self._is_playing = threading.Event()
        self._is_synthesizing = threading.Event()

        if not lazy_model and os.path.exists(self.model_path):
            self._load_model()

        self.synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self.synth_thread.start()

        self.play_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.play_thread.start()

    @property
    def generation_id(self) -> int:
        """Current generation/epoch counter for synthesis invalidation."""
        with self._epoch_lock:
            return self._generation_id

    @property
    def is_muted(self) -> bool:
        with self._mute_lock:
            return self._muted

    @is_muted.setter
    def is_muted(self, val: bool) -> None:
        self.set_muted(val)

    @property
    def muted(self) -> bool:
        with self._mute_lock:
            return self._muted

    @muted.setter
    def muted(self, val: bool) -> None:
        self.set_muted(val)

    def set_muted(self, val: bool) -> None:
        with self._mute_lock:
            self._muted = bool(val)
            now_muted = self._muted
        if now_muted:
            self._silence_and_drain_playback()

    def _silence_and_drain_playback(self) -> None:
        try:
            sd.stop()
        except Exception:
            pass
        self._is_playing.clear()
        self._is_synthesizing.clear()

        with self._epoch_lock:
            self._generation_id += 1

        # Drain sentence queue to prevent background synthesis during mute
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
                self.sentence_queue.task_done()
            except (queue.Empty, ValueError):
                break

        # Drain audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except (queue.Empty, ValueError):
                break

    def toggle_mute(self) -> bool:
        """
        Thread-safe dynamic mute toggle.
        When muting, immediately silences hardware audio via sd.stop()
        and clears audio_queue and _is_playing.
        Returns:
            bool: True if muted, False if unmuted.
        """
        with self._mute_lock:
            self._muted = not self._muted
            now_muted = self._muted
        if now_muted:
            self._silence_and_drain_playback()
        return now_muted

    def interrupt(self) -> float:
        """
        Instantly halts active audio playback and purges all queues (< 200 ms latency).
        Aborts active sounddevice stream and drains pending sentence and audio queues.
        Returns:
            float: Elapsed interruption latency in milliseconds.
        """
        t0 = time.perf_counter()

        with self._epoch_lock:
            self._generation_id += 1

        # 1. Purge synthesis sentence queue
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
                self.sentence_queue.task_done()
            except (queue.Empty, ValueError):
                break

        # 2. Purge pending audio playback queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except (queue.Empty, ValueError):
                break

        # 3. Halt sounddevice output immediately
        try:
            if hasattr(sd, "_last_callback") and sd._last_callback and hasattr(sd._last_callback, "stream") and sd._last_callback.stream:
                try:
                    sd._last_callback.stream.abort()
                except Exception:
                    pass
            sd.stop()
        except Exception as e:
            logging.debug(f"[VoiceEngine Interrupt]: {e}")

        self._is_playing.clear()
        self._is_synthesizing.clear()

        dt_ms = (time.perf_counter() - t0) * 1000
        return dt_ms

    def is_speaking(self) -> bool:
        """Returns True if the assistant is synthesizing, queued, or playing audio."""
        return (
            self._is_playing.is_set()
            or not self.audio_queue.empty()
            or self._is_synthesizing.is_set()
            or not self.sentence_queue.empty()
        )

    def _load_model(self):
        if self.model is not None:
            return
        if not os.path.isfile(self.model_path):
            print(Fore.YELLOW + f"[TTS] Скачиваю модель Silero V4 в {self.model_path}...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.hub.download_url_to_file("https://models.silero.ai/models/tts/ru/v4_ru.pt", self.model_path)

        if os.path.exists(self.model_path):
            print(Fore.CYAN + "[TTS] Загрузка голосового движка...")
            torch.set_num_threads(4)
            self.model = torch.package.PackageImporter(self.model_path).load_pickle("tts_models", "model")
            self.model.to(self.device)
            try:
                for _ in range(2):
                    _ = self.model.apply_tts(text="Разминка", speaker=self.speaker, sample_rate=self.sample_rate)
            except Exception:
                pass

    def feed_sentence(self, sentence: str) -> bool:
        """Enqueues sentence for asynchronous normalization, synthesis, and playback."""
        if self.stop_event.is_set():
            return False
        if self.is_muted:
            return False
        if not sentence or not str(sentence).strip():
            return False
        norm = clean_and_normalize_tts_text(str(sentence))
        if not norm:
            return False
        self.sentence_queue.put(norm)
        return True

    def play_chunk(self, audio_data: Union[np.ndarray, torch.Tensor]) -> None:
        """Directly queues raw audio chunk for playback (used in tests and direct injection)."""
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        arr = np.asarray(audio_data, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.flatten()
        self.audio_queue.put(arr)

    def speak(self, text: str) -> None:
        """Backward-compatibility alias feeding text through SentenceChunker."""
        chunker = SentenceChunker()
        for sent in chunker.feed(text):
            self.feed_sentence(sent)
        for sent in chunker.flush():
            self.feed_sentence(sent)

    def _synth_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                norm_text = self.sentence_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if norm_text is None:
                self.sentence_queue.task_done()
                break

            if self.is_muted:
                self.sentence_queue.task_done()
                continue

            with self._epoch_lock:
                current_gen = self._generation_id

            try:
                with self._epoch_lock:
                    if self._generation_id != current_gen or self.is_muted:
                        continue

                self._is_synthesizing.set()
                if self.model is None:
                    self._load_model()
                if self.model is not None:
                    raw_audio = self.model.apply_tts(
                        text=norm_text,
                        speaker=self.speaker,
                        sample_rate=self.sample_rate
                    )
                    resampled = resample_pitch_audio(
                        raw_audio,
                        pitch_shift=self.pitch_shift,
                        sample_rate=self.sample_rate
                    )
                    with self._epoch_lock:
                        is_valid = (self._generation_id == current_gen) and not self.is_muted

                    if is_valid:
                        self.audio_queue.put(resampled)
                    else:
                        logging.debug(
                            f"[VoiceEngine Synth Worker]: Discarded stale in-flight synthesis "
                            f"(chunk_gen={current_gen}, current_gen={self._generation_id}, muted={self.is_muted})"
                        )
            except Exception as e:
                logging.error(f"[TTS Synthesis Error]: {e}")
            finally:
                self._is_synthesizing.clear()
                self.sentence_queue.task_done()

    def _playback_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if audio_data is None:
                self.audio_queue.task_done()
                break

            if self.is_muted:
                self.audio_queue.task_done()
                continue

            try:
                self._is_playing.set()
                if not self.is_muted:
                    sd.play(audio_data, samplerate=self.sample_rate)
                    sd.wait()
            except Exception as e:
                self.driver_error_count += 1
                logging.warning(f"[Playback Driver Error]: {e}")
            finally:
                self._is_playing.clear()
                self.audio_queue.task_done()

    def flush(self, timeout: float = 15.0) -> None:
        """Waits for pending sentences to synthesize and audio queue playback to complete."""
        if self.is_muted:
            return
        t0 = time.time()
        while not self.sentence_queue.empty() and (time.time() - t0) < timeout:
            if self.is_muted:
                return
            time.sleep(0.01)
        while self._is_synthesizing.is_set() and (time.time() - t0) < timeout:
            if self.is_muted:
                return
            time.sleep(0.01)
        while (not self.audio_queue.empty() or self._is_playing.is_set()) and (time.time() - t0) < timeout:
            if self.is_muted:
                return
            time.sleep(0.01)

    def stop(self, timeout: float = 2.0) -> None:
        """Orderly shutdown of worker threads and queue drain."""
        self.stop_event.set()
        with self._epoch_lock:
            self._generation_id += 1
        self.sentence_queue.put(None)
        self.audio_queue.put(None)
        try:
            sd.stop()
        except Exception:
            pass
        if hasattr(self, "synth_thread") and self.synth_thread.is_alive():
            self.synth_thread.join(timeout=timeout)
        if hasattr(self, "play_thread") and self.play_thread.is_alive():
            self.play_thread.join(timeout=timeout)

        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
                self.sentence_queue.task_done()
            except (queue.Empty, ValueError):
                break

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except (queue.Empty, ValueError):
                break
