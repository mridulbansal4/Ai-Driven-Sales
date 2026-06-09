"""Centralized environment configuration with validation and defaults.

Every other module imports settings from here instead of calling ``os.getenv``
directly. Values are read once at module import; tests can override by
re-importing or by mutating ``settings`` attributes before processors are
constructed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Project layout: this file lives at backend/pipeline/core/config.py
BACKEND_DIR: Path = Path(__file__).resolve().parents[2]
PROJECT_ROOT: Path = BACKEND_DIR.parent


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", name, raw, default)
        return default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default %s", name, raw, default)
        return default


def _str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or default


@dataclass(frozen=True)
class WhisperSettings:
    model: str
    device: str
    compute_type: str
    language: str  # "auto" or an ISO code
    legacy_dlls: bool


@dataclass(frozen=True)
class KokoroSettings:
    voice_en: str
    voice_hi: str


@dataclass(frozen=True)
class OllamaSettings:
    url: str
    model: str
    temperature: float
    max_tokens: int
    request_timeout_s: float


@dataclass(frozen=True)
class Settings:
    log_level: str
    whisper: WhisperSettings
    kokoro: KokoroSettings
    ollama: OllamaSettings


def _load() -> Settings:
    return Settings(
        log_level=_str("LOG_LEVEL", "INFO").upper(),
        whisper=WhisperSettings(
            model=_str("WHISPER_MODEL", "large-v3-turbo"),
            device=_str("WHISPER_DEVICE", "cuda").lower(),
            compute_type=_str("WHISPER_COMPUTE", "int8"),
            language=_str("WHISPER_LANGUAGE", "auto").lower(),
            legacy_dlls=_bool("WHISPER_LEGACY_DLLS", False),
        ),
        kokoro=KokoroSettings(
            voice_en=_str("KOKORO_VOICE_EN", "af_heart"),
            voice_hi=_str("KOKORO_VOICE_HI", "hf_alpha"),
        ),
        ollama=OllamaSettings(
            url=_str("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            model=_str("OLLAMA_MODEL", "qwen2.5:7b"),
            temperature=_float("OLLAMA_TEMPERATURE", 0.6),
            max_tokens=_int("OLLAMA_MAX_TOKENS", 120),
            request_timeout_s=_float("OLLAMA_TIMEOUT", 120.0),
        ),
    )


settings: Settings = _load()


def reload_settings() -> Settings:
    """Re-read environment variables. Mainly for tests."""
    global settings
    settings = _load()
    return settings


def load_dotenv_from_backend() -> None:
    """Load ``backend/.env`` regardless of the current working directory."""
    from dotenv import load_dotenv

    env_path = BACKEND_DIR / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        reload_settings()
    else:
        logger.info("No .env file at %s; using process environment only", env_path)


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once, respecting ``LOG_LEVEL`` from .env."""
    chosen = (level or settings.log_level or "INFO").upper()
    numeric = getattr(logging, chosen, logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(numeric)
