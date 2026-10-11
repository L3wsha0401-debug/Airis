"""
Airis Live2D Avatar Subsystem.
Provides VTube Studio (VTS) WebSocket client, real-time RMS audio lip-sync,
and non-blocking AvatarController lifecycle supervisor.
"""

from __future__ import annotations

from airis.config import AvatarConfig
from airis.avatar.vts import VTSClient
from airis.avatar.lip_sync import LipSyncProcessor
from airis.avatar.controller import AvatarController

__all__ = [
    "AvatarConfig",
    "VTSClient",
    "LipSyncProcessor",
    "AvatarController",
]
