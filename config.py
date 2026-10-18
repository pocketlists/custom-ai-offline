from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    pass


def _env(name, default=None):
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _env_int(name, default, lo=None, hi=None):
    raw = _env(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be an integer")
    if lo is not None and value < lo:
        raise ConfigError(f"{name} must be >= {lo}")
    if hi is not None and value > hi:
        raise ConfigError(f"{name} must be <= {hi}")
    return value


@dataclass(frozen=True)
class Settings:
    bot_token: str | None
    hf_token: str | None
    cpu_threads: int
    max_upload_mb: int
    max_input_chars: int
    max_photo_mb: int
    max_reference_audio_mb: int
    max_reference_audio_seconds: int
    queue_size: int
    job_timeout: int
    data_dir: Path
    log_level: str
    ffmpeg_bin: str
    openvoice_repo: str | None
    openvoice_ckpt: str | None

    @classmethod
    def load(cls):
        base = Path(__file__).resolve().parent
        data = Path(_env("BOT_DATA_DIR") or (base / "runtime")).resolve()
        data.mkdir(parents=True, exist_ok=True)
        cpu_default = max(1, min(4, os.cpu_count() or 1))
        return cls(
            bot_token=_env("TELEGRAM_BOT_TOKEN"),
            hf_token=_env("HF_TOKEN"),
            cpu_threads=_env_int("CPU_THREADS", cpu_default, 1, os.cpu_count() or 1),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 45, 1, 2000),
            max_input_chars=_env_int("MAX_INPUT_CHARS", 1500, 50, 10000),
            max_photo_mb=_env_int("MAX_PHOTO_MB", 10, 1, 50),
            max_reference_audio_mb=_env_int("MAX_REFERENCE_AUDIO_MB", 15, 1, 50),
            max_reference_audio_seconds=_env_int("MAX_REFERENCE_AUDIO_SECONDS", 30, 3, 120),
            queue_size=_env_int("QUEUE_SIZE", 8, 1, 100),
            job_timeout=_env_int("JOB_TIMEOUT", 900, 60, 7200),
            data_dir=data,
            log_level=(_env("LOG_LEVEL") or "INFO").upper(),
            ffmpeg_bin=_env("FFMPEG_BIN") or "ffmpeg",
            openvoice_repo=_env("OPENVOICE_REPO_DIR"),
            openvoice_ckpt=_env("OPENVOICE_CHECKPOINT_DIR"),
        )


def require_token(settings: Settings) -> str:
    token = settings.bot_token
    if not token:
        raise ConfigError(
            "TELEGRAM_BOT_TOKEN is not set. Add it under "
            "GitHub repo > Settings > Secrets and variables > Actions > New repository secret."
        )
    if ":" not in token or len(token.split(":", 1)[1]) < 20:
        raise ConfigError(
            "TELEGRAM_BOT_TOKEN looks malformed. Expected '<bot_id>:<secret>'. "
            "Value was not printed for security."
        )
    return token
