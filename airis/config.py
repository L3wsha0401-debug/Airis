"""
Airis Configuration Module.
Provides dataclass schemas, YAML loading, and deep fallback defaults.
"""

from __future__ import annotations
import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List
import yaml

# Base directory for the Airis installation (C:\LLM\Airis)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "Logs")
MEMORY_LOG_FILE = os.path.join(LOGS_DIR, "iris_memory.log")
CHAT_LOG_FILE = os.path.join(LOGS_DIR, "logs.txt")
SESSION_HISTORY_FILE = os.path.join(LOGS_DIR, "session_history.json")

DEFAULT_STOP_TOKENS = [
    "</tools>", "<tools>",
    "</schema>", "<schema>",
    "</think>", "<think>",
    "</tool_call>", "<tool_call>",
    "<start_of_turn>", "<end_of_turn>",
    "\n\n",
    "\nuser", "user",
    "\nassistant", "assistant",
    "L3wsha:", "Айрис:",
    "<|im_end|>", "<|im_start|>",
    "<|eot_id|>", "<|start_header_id|>",
    "</", "<|"
]


@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434"
    model: str = "gemma2:9b"
    embedding_model: str = "bge-m3"
    options: Dict[str, Any] = field(default_factory=lambda: {
        'num_ctx': 4096,
        'temperature': 0.4,
        'top_p': 0.9,
        'min_p': 0.08,
        'repeat_penalty': 1.1,
        'repeat_last_n': 64,
        'stop': list(DEFAULT_STOP_TOKENS)
    })


@dataclass
class TTSConfig:
    enabled: bool = True
    model_path: str = r"C:\LLM\Instruments\model.pt"
    speaker: str = "baya"
    pitch_shift: float = 1.25
    sample_rate: int = 48000
    lazy_model: bool = False
    muted: bool = False


@dataclass
class STTConfig:
    enabled: bool = True
    model_name: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 4
    language: str = "ru"
    beam_size: int = 1
    sample_rate: int = 16000
    input_mode: str = "ptt"
    ptt_hotkey: str = "space"
    vad_threshold: float = 0.5
    vad_threshold_speaking: float = 0.75
    vad_silence_timeout_s: float = 0.8
    energy_floor: float = 0.005
    barge_in: bool = True

    @property
    def mode(self) -> str:
        return self.input_mode

    @mode.setter
    def mode(self, val: str) -> None:
        self.input_mode = val

    @property
    def barge_in_enabled(self) -> bool:
        return self.barge_in

    @barge_in_enabled.setter
    def barge_in_enabled(self, val: bool) -> None:
        self.barge_in = val

    @property
    def vad_energy_threshold(self) -> float:
        return self.energy_floor

    @vad_energy_threshold.setter
    def vad_energy_threshold(self, val: float) -> None:
        self.energy_floor = val

    @property
    def vad_silence_duration_s(self) -> float:
        return self.vad_silence_timeout_s

    @vad_silence_duration_s.setter
    def vad_silence_duration_s(self, val: float) -> None:
        self.vad_silence_timeout_s = val



@dataclass
class MemoryConfig:
    chroma_dir: str = "chroma_db"
    results_count: int = 3
    max_distance: float = 0.35
    dedup_threshold: float = 0.25
    max_history_turns: int = 12
    session_history_file: str = "Logs/session_history.json"
    memory_log_file: str = "Logs/iris_memory.log"
    chat_log_file: str = "Logs/logs.txt"


@dataclass
class SessionConfig:
    max_history_turns: int = 12
    session_history_file: str = "Logs/session_history.json"


@dataclass
class LoggingConfig:
    logs_dir: str = "Logs"
    memory_log_file: str = "Logs/iris_memory.log"
    chat_log_file: str = "Logs/logs.txt"


@dataclass
class PersonalityConfig:
    user_name_default: str = "l3wsha"
    max_emojis_per_response: int = 2
    allow_cookie_signature: bool = False


@dataclass
class AppConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> AppConfig:
        """
        Constructs an AppConfig instance from a dictionary, ensuring resilience against
        missing, empty, or None section headers.
        """
        config = cls()
        if not isinstance(data, dict):
            return config

        section_factories = {
            "ollama": OllamaConfig,
            "tts": TTSConfig,
            "stt": STTConfig,
            "memory": MemoryConfig,
            "session": SessionConfig,
            "logging": LoggingConfig,
            "personality": PersonalityConfig,
        }

        for section, factory in section_factories.items():
            val = data.get(section)
            if isinstance(val, dict):
                target_sec = getattr(config, section, None)
                if target_sec is None or not hasattr(target_sec, "__dataclass_fields__"):
                    target_sec = factory()
                    setattr(config, section, target_sec)
                _merge_dict_into_dataclass(target_sec, val)
            else:
                # Fall back to default dataclass instance when None or not a dict
                setattr(config, section, factory())

        for key, value in data.items():
            if key not in section_factories and hasattr(config, key) and value is not None:
                setattr(config, key, value)

        return config


# Backward-compatible alias
AirisConfig = AppConfig


def _merge_dict_into_dataclass(target_obj: Any, source_dict: Dict[str, Any]) -> None:
    """Recursively updates dataclass fields from a dictionary."""
    if not isinstance(source_dict, dict):
        return
    key_aliases = {
        "mode": "input_mode",
        "barge_in_enabled": "barge_in",
        "vad_energy_threshold": "energy_floor",
        "vad_silence_duration_s": "vad_silence_timeout_s",
    }
    for key, value in source_dict.items():
        canonical_key = key_aliases.get(key, key)
        target_key = canonical_key if hasattr(target_obj, canonical_key) else key
        if hasattr(target_obj, target_key):
            current_val = getattr(target_obj, target_key)
            if hasattr(current_val, "__dataclass_fields__"):
                if isinstance(value, dict):
                    _merge_dict_into_dataclass(current_val, value)
                # If value is None or not a dict, retain current_val (default dataclass instance)
            elif isinstance(current_val, dict) and isinstance(value, dict):
                # Deep merge dictionaries (e.g. options)
                merged = copy.deepcopy(current_val)
                merged.update(value)
                setattr(target_obj, target_key, merged)
            else:
                if value is not None:
                    setattr(target_obj, target_key, value)


def load_config(config_path: str | None = None) -> AppConfig:
    """
    Loads AppConfig from YAML file with automatic fallback to sensible defaults.
    Ensures missing sections or fields inherit defaults gracefully.
    """
    config = AppConfig()

    if config_path is None:
        primary_path = os.path.join(BASE_DIR, "config.yaml")
        if os.path.exists(primary_path):
            config_path = primary_path
        elif os.path.exists("config.yaml"):
            config_path = "config.yaml"
        else:
            config_path = primary_path

    if not os.path.exists(config_path):
        return config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        if isinstance(raw_data, dict):
            config = AppConfig.from_dict(raw_data)
    except Exception as e:
        logging.warning(f"[CONFIG] Error loading {config_path}: {e}. Using defaults.")

    return config
