"""
Airis VTube Studio (VTS) WebSocket Client.
Provides a defensive wrapper over pyvts with robust exception shielding,
token caching, and seamless offline fallback.
"""

from __future__ import annotations

import os
import io
import contextlib
import asyncio
import logging
from typing import Optional, Any
import pyvts
import websockets

logger = logging.getLogger("airis.avatar.vts")


class StatusBool(int):
    """Integer/boolean subclass that can be evaluated directly or invoked as a callable."""
    def __new__(cls, val: bool | int) -> StatusBool:
        return super().__new__(cls, bool(val))

    def __call__(self) -> bool:
        return bool(self)


class VTSClient:
    """
    Asynchronous VTube Studio API client wrapping pyvts.
    Shields all WebSocket and networking errors so caller code never crashes
    when VTube Studio is offline or unavailable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8001,
        plugin_name: str = "Airis Live2D Controller",
        developer: str = "Airis Team",
        token_path: str = "Logs/vts_token.txt",
    ) -> None:
        self.host = host
        self.port = port
        self.plugin_name = plugin_name
        self.developer = developer
        self.token_path = token_path

        self._vts: Optional[pyvts.vts] = None
        self._is_connected: bool = False
        self._is_authenticated: bool = False
        self._lock: Optional[asyncio.Lock] = None

    @property
    def is_connected(self) -> StatusBool:
        return StatusBool(self._is_connected)

    @property
    def is_authenticated(self) -> StatusBool:
        return StatusBool(self._is_authenticated)

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _ensure_token_dir(self) -> None:
        """Ensures the parent directory for token storage exists."""
        try:
            token_dir = os.path.dirname(os.path.abspath(self.token_path))
            if token_dir:
                os.makedirs(token_dir, exist_ok=True)
        except Exception as e:
            logger.debug(f"[VTSClient] Error creating token dir: {e}")

    async def connect(self) -> bool:
        """
        Connects to VTube Studio WebSocket API server.
        Shields all connection exceptions (ConnectionRefusedError, OSError,
        WebSocketException, asyncio.TimeoutError) and returns False on failure.
        """
        self._ensure_token_dir()
        plugin_info = {
            "plugin_name": self.plugin_name,
            "developer": self.developer,
            "authentication_token_path": self.token_path,
            "plugin_icon": None,
        }
        vts_api_info = {
            "version": "1.0",
            "name": "VTubeStudioPublicAPI",
            "host": self.host,
            "port": self.port,
        }

        try:
            self._vts = pyvts.vts(plugin_info=plugin_info, vts_api_info=vts_api_info)
            await self._vts.connect()
            # Verify connection status in pyvts
            if getattr(self._vts, "websocket", None) is not None:
                self._is_connected = True
                return True
            else:
                self._is_connected = False
                return False
        except (
            OSError,
            ConnectionRefusedError,
            websockets.exceptions.WebSocketException,
            asyncio.TimeoutError,
            Exception,
        ) as e:
            logger.debug(f"[VTSClient] Connection to {self.host}:{self.port} failed: {e}")
            self._is_connected = False
            self._is_authenticated = False
            return False

    async def authenticate(self, force_reauth: bool = False) -> bool:
        """
        Authenticates against the VTube Studio API.
        Attempts using cached token first. If token is invalid or rejected,
        attempts one forced re-authentication request.
        Returns True if authenticated, False otherwise.
        """
        if not self._is_connected or self._vts is None or getattr(self._vts, "websocket", None) is None:
            self._is_authenticated = False
            return False

        try:
            async with self.lock:
                # 1. Fetch token (from disk or from VTS server)
                # pyvts prints plugin_icon to stdout; silence it for clean console
                with contextlib.redirect_stdout(io.StringIO()):
                    await self._vts.request_authenticate_token(force=force_reauth)
                    auth_ok = await self._vts.request_authenticate()

                # 2. If initial auth failed with cached token, force a fresh token request
                if not auth_ok and not force_reauth:
                    logger.info("[VTSClient] Cached token rejected, requesting fresh token...")
                    with contextlib.redirect_stdout(io.StringIO()):
                        await self._vts.request_authenticate_token(force=True)
                        auth_ok = await self._vts.request_authenticate()

            self._is_authenticated = bool(auth_ok)
            return self._is_authenticated
        except (
            OSError,
            ConnectionRefusedError,
            websockets.exceptions.WebSocketException,
            asyncio.TimeoutError,
            Exception,
        ) as e:
            logger.debug(f"[VTSClient] Authentication failed: {e}")
            self._is_authenticated = False
            return False

    async def set_parameter(self, name: str, value: float, weight: float = 1.0) -> bool:
        """
        Updates a Live2D model parameter value (e.g. MouthOpen) in VTube Studio.
        """
        if (
            not self._is_connected
            or not self._is_authenticated
            or self._vts is None
            or getattr(self._vts, "websocket", None) is None
        ):
            return False

        try:
            msg = self._vts.vts_request.requestSetParameterValue(
                parameter=str(name),
                value=float(value),
                weight=float(weight),
            )
            async with self.lock:
                if (
                    not self._is_connected
                    or not self._is_authenticated
                    or self._vts is None
                    or getattr(self._vts, "websocket", None) is None
                ):
                    return False
                response = await self._vts.request(msg)
            return bool(response and "data" in response)
        except (
            OSError,
            websockets.exceptions.WebSocketException,
            asyncio.TimeoutError,
            Exception,
        ) as e:
            logger.debug(f"[VTSClient] set_parameter '{name}' failed: {e}")
            self._is_connected = False
            self._is_authenticated = False
            return False

    async def trigger_hotkey(self, hotkey_id: str) -> bool:
        """
        Triggers an expression or hotkey by ID in VTube Studio.
        """
        if (
            not self._is_connected
            or not self._is_authenticated
            or self._vts is None
            or getattr(self._vts, "websocket", None) is None
        ):
            return False

        try:
            msg = self._vts.vts_request.requestTriggerHotKey(hotkeyID=str(hotkey_id))
            async with self.lock:
                if (
                    not self._is_connected
                    or not self._is_authenticated
                    or self._vts is None
                    or getattr(self._vts, "websocket", None) is None
                ):
                    return False
                response = await self._vts.request(msg)
            return bool(response and "data" in response)
        except (
            OSError,
            websockets.exceptions.WebSocketException,
            asyncio.TimeoutError,
            Exception,
        ) as e:
            logger.debug(f"[VTSClient] trigger_hotkey '{hotkey_id}' failed: {e}")
            self._is_connected = False
            self._is_authenticated = False
            return False

    async def close(self) -> None:
        """
        Gracefully closes the WebSocket connection.
        """
        self._is_connected = False
        self._is_authenticated = False
        if self._vts is not None and getattr(self._vts, "websocket", None) is not None:
            try:
                await self._vts.close()
            except Exception as e:
                logger.debug(f"[VTSClient] Error closing websocket: {e}")
            finally:
                self._vts = None
