"""
Airis Live2D Avatar Controller.
Manages VTube Studio lifecycle on a dedicated asynchronous worker thread,
providing non-blocking synchronous APIs for audio lip-sync and emotion dispatch.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional, Union, Sequence, Any
import numpy as np
import torch

from airis.config import AvatarConfig
from airis.avatar.vts import VTSClient, StatusBool
from airis.avatar.lip_sync import LipSyncProcessor

logger = logging.getLogger("airis.avatar.controller")


class AvatarController:
    """
    High-level Live2D avatar controller.
    Runs an asyncio event loop on a dedicated background thread to supervise
    VTube Studio connectivity, execute real-time RMS lip-sync parameter updates,
    and trigger expression hotkeys without blocking main interaction loops.
    """

    def __init__(
        self,
        config: Optional[AvatarConfig] = None,
        vts_client: Optional[VTSClient] = None,
    ) -> None:
        self.config = config or AvatarConfig()
        if vts_client is not None:
            self.vts_client = vts_client
        else:
            self.vts_client = VTSClient(
                host=self.config.host,
                port=self.config.port,
                plugin_name=self.config.plugin_name,
                developer=self.config.developer,
                token_path=self.config.token_path,
            )

        self.lip_sync = LipSyncProcessor(
            min_rms=self.config.min_rms,
            max_rms=self.config.max_rms,
            smoothing=self.config.smoothing,
        )

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started_event = threading.Event()
        self._stop_event = threading.Event()
        self._task_lock = threading.Lock()
        self._lip_sync_task: Optional[asyncio.Task] = None
        self._supervisor_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> StatusBool:
        """Indicates whether the underlying VTS client is connected."""
        return StatusBool(bool(self.vts_client and self.vts_client.is_connected))

    @property
    def is_authenticated(self) -> StatusBool:
        """Indicates whether the underlying VTS client is authenticated."""
        return StatusBool(bool(self.vts_client and self.vts_client.is_authenticated))

    def start(self) -> None:
        """
        Starts the background supervisor thread and event loop.
        Non-blocking to the application; initializes connection in background.
        """
        if not self.config.enabled:
            logger.info("[AvatarController] Avatar subsystem disabled in configuration.")
            return

        with self._task_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._started_event.clear()
            self._thread = threading.Thread(
                target=self._run_event_loop,
                name="AvatarWorkerThread",
                daemon=True,
            )
            self._thread.start()

        # Wait briefly for event loop initialization
        self._started_event.wait(timeout=2.0)

    def _run_event_loop(self) -> None:
        """Dedicated thread target running the asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started_event.set()

        try:
            self._supervisor_task = self._loop.create_task(self._reconnect_supervisor())
            self._loop.run_forever()
        finally:
            try:
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                # Graceful client disconnect
                if self.vts_client:
                    self._loop.run_until_complete(self.vts_client.close())
            except Exception as e:
                logger.debug(f"[AvatarController] Error during loop teardown: {e}")
            finally:
                self._loop.close()

    async def _reconnect_supervisor(self) -> None:
        """
        Periodically checks connection status and reconnects to VTube Studio if offline.
        """
        while not self._stop_event.is_set():
            try:
                if self.config.enabled and not self.vts_client.is_connected:
                    connected = await self.vts_client.connect()
                    if connected:
                        await self.vts_client.authenticate()
            except Exception as e:
                logger.debug(f"[AvatarController] Reconnect attempt error: {e}")

            # Sleep in short increments to allow rapid responsive shutdown
            interval = max(0.1, float(self.config.reconnect_interval_s))
            steps = max(1, int(interval / 0.1))
            for _ in range(steps):
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(0.1)

    def trigger_connect(self) -> None:
        """
        Thread-safe trigger for immediate connection attempt without waiting
        for reconnect_interval_s.
        """
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._do_connect_once(), self._loop)

    async def _do_connect_once(self) -> bool:
        if self.config.enabled and not self.vts_client.is_connected:
            connected = await self.vts_client.connect()
            if connected:
                await self.vts_client.authenticate()
            return self.is_connected
        return self.is_connected

    def start_audio_playback(
        self,
        audio_data: Union[np.ndarray, torch.Tensor, Sequence[float]],
        sample_rate: int = 48000,
    ) -> None:
        """
        Begins streaming real-time RMS mouth updates to VTube Studio for the given audio.
        Non-blocking: schedules background async task and returns immediately.
        """
        if (
            not self.config.enabled
            or self._stop_event.is_set()
            or self._loop is None
            or not self._loop.is_running()
            or not self.is_connected
            or audio_data is None
        ):
            return

        frames = self.lip_sync.slice_frames(
            audio_data,
            sample_rate=sample_rate,
            frame_ms=self.config.frame_duration_ms,
        )
        if not frames:
            return

        with self._task_lock:
            if self._lip_sync_task is not None and not self._lip_sync_task.done():
                self._lip_sync_task.cancel()

            self._lip_sync_task = asyncio.run_coroutine_threadsafe(
                self._stream_lip_sync(frames),
                self._loop,
            )

    async def _stream_lip_sync(self, frames: list[np.ndarray]) -> None:
        """
        Iterates over audio frames, sends MouthOpen parameter updates,
        and ensures mouth closure upon completion or interruption.
        """
        param_name = self.config.mouth_open_parameter
        delay_s = max(0.005, self.config.frame_duration_ms / 1000.0)
        was_cancelled = False

        try:
            for frame in frames:
                if self._stop_event.is_set():
                    break
                val = self.lip_sync.process_frame(frame)
                if self.vts_client and self.vts_client.is_connected:
                    await self.vts_client.set_parameter(param_name, val)
                await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            was_cancelled = True
        except Exception as e:
            logger.debug(f"[AvatarController] Lip-sync streaming error: {e}")
        finally:
            self.lip_sync.reset()
            if not was_cancelled and self.vts_client and self.vts_client.is_connected:
                try:
                    await self.vts_client.set_parameter(param_name, 0.0)
                except Exception:
                    pass

    def reset_mouth(self) -> None:
        """
        Cancels active lip-sync task and immediately resets MouthOpen parameter to 0.0 (< 10 ms).
        Synchronous and thread-safe.
        """
        self.lip_sync.reset()

        with self._task_lock:
            if self._lip_sync_task is not None and not self._lip_sync_task.done():
                self._lip_sync_task.cancel()
                self._lip_sync_task = None

        if self._loop is not None and self._loop.is_running() and self.is_connected:
            try:
                asyncio.run_coroutine_threadsafe(self._send_mouth_zero(), self._loop)
            except RuntimeError:
                pass

    async def _send_mouth_zero(self) -> None:
        """Directly sends MouthOpen = 0.0 parameter to VTS."""
        if self.vts_client and self.vts_client.is_connected:
            try:
                await self.vts_client.set_parameter(self.config.mouth_open_parameter, 0.0)
            except Exception as e:
                logger.debug(f"[AvatarController] Mouth zero update error: {e}")

    def set_emotion(self, tag: str) -> None:
        """
        Normalizes emotion tag (e.g. '[happy]' -> 'happy'), looks up configured hotkey,
        and dispatches HotkeyTriggerRequest over WebSocket.
        Synchronous, non-blocking.
        """
        if not tag or not self.config.enabled:
            return

        clean_tag = str(tag).strip("[] \t\r\n").lower()
        hotkey_id = self.config.expressions.get(clean_tag)
        if not hotkey_id:
            logger.debug(f"[AvatarController] No hotkey mapped for emotion: '{clean_tag}'")
            return

        if self._loop is not None and self._loop.is_running() and self.is_connected:
            try:
                asyncio.run_coroutine_threadsafe(self._send_hotkey(hotkey_id), self._loop)
            except RuntimeError:
                pass

    async def _send_hotkey(self, hotkey_id: str) -> None:
        """Dispatches hotkey trigger request to VTS."""
        if self.vts_client and self.vts_client.is_connected:
            try:
                await self.vts_client.trigger_hotkey(hotkey_id)
            except Exception as e:
                logger.debug(f"[AvatarController] Error triggering hotkey '{hotkey_id}': {e}")

    def stop(self, timeout: float = 2.0) -> None:
        """
        Orderly shutdown of background event loop, WebSocket connection, and thread.
        """
        self._stop_event.set()
        self.reset_mouth()

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            self._thread = None

        self._loop = None
