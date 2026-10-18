from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from telegram import Bot, Message
from telegram.error import RetryAfter, TelegramError

from config import Settings
from models import ValidationError

logger = logging.getLogger(__name__)

CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def validate_text(text: str, max_chars: int) -> str:
    if not isinstance(text, str):
        raise ValidationError("Input must be text")
    cleaned = CONTROL.sub("", unicodedata.normalize("NFC", text).strip())
    if not cleaned:
        raise ValidationError("Input text is empty")
    if len(cleaned) > max_chars:
        raise ValidationError(f"Input is too long ({len(cleaned)} > {max_chars})")
    return cleaned


def sanitize_filename(name: str, fallback: str = "file") -> str:
    stem = Path(name).name
    stem = UNSAFE.sub("_", stem).strip("._")
    return stem[:80] or fallback


def safe_path(base: Path, filename: str) -> Path:
    base = base.resolve()
    target = (base / sanitize_filename(filename)).resolve()
    if base not in target.parents and target != base:
        raise ValidationError("Resolved path escapes temp directory")
    return target


def unique_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}{suffix}"


def ffmpeg_available(binary: str = "ffmpeg") -> bool:
    return shutil.which(binary) is not None


def _run_ffmpeg(args: list[str], timeout: int = 120) -> None:
    result = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, check=False, shell=False,
    )
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace").strip()[-400:]
        raise ValidationError(f"ffmpeg failed: {err}")


def probe_duration(path: Path, ffmpeg_bin: str = "ffmpeg") -> float:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    if not ffmpeg_available(ffmpeg_bin):
        return 0.0
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-v", "error",
             "-show_entries", "format=duration", "-of", "json", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, shell=False,
        )
        if result.returncode != 0:
            return 0.0
        payload = json.loads(result.stdout.decode("utf-8", "replace"))
        return float(payload["format"]["duration"])
    except Exception:
        return 0.0


def to_ogg_opus(input_path: Path, output_path: Path, ffmpeg_bin="ffmpeg") -> Path:
    if not ffmpeg_available(ffmpeg_bin):
        return input_path
    _run_ffmpeg([
        ffmpeg_bin, "-y", "-hide_banner", "-v", "error",
        "-i", str(input_path), "-ac", "1", "-ar", "48000",
        "-c:a", "libopus", "-b:a", "48k", str(output_path),
    ])
    return output_path


def to_wav_16k(input_path: Path, output_path: Path, ffmpeg_bin="ffmpeg") -> Path:
    if not ffmpeg_available(ffmpeg_bin):
        raise ValidationError("ffmpeg is not installed on this runner")
    _run_ffmpeg([
        ffmpeg_bin, "-y", "-hide_banner", "-v", "error",
        "-i", str(input_path), "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(output_path),
    ])
    return output_path


def split_wav(path: Path, out_dir: Path, max_bytes: int) -> list[Path]:
    import math
    import wave
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.getnframes()
        raw = w.readframes(frames)
    if not (rate and width and ch):
        raise ValidationError("Invalid WAV header")
    bytes_per_sec = rate * ch * width
    max_sec = max(1.0, (max_bytes * 0.92) / bytes_per_sec)
    total_sec = frames / float(rate)
    parts = max(1, math.ceil(total_sec / max_sec))
    if parts == 1:
        return [path]
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_per_part = int(max_sec * rate)
    result: list[Path] = []
    for i in range(parts):
        start = i * frames_per_part
        chunk = raw[start * ch * width:(start + frames_per_part) * ch * width]
        if not chunk:
            continue
        target = out_dir / f"{path.stem}_p{i + 1:02d}.wav"
        with wave.open(str(target), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(width)
            w.setframerate(rate)
            w.writeframes(chunk)
        result.append(target)
    return result or [path]


def validate_image(path: Path, max_mb: int) -> tuple[int, int, str]:
    from PIL import Image
    if not path.exists():
        raise ValidationError("Image not found")
    if path.stat().st_size > max_mb * 1024 * 1024:
        raise ValidationError(f"Image is larger than {max_mb} MB")
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            fmt = (im.format or "").upper()
    except Exception as exc:
        raise ValidationError("File is not a valid image") from exc
    if fmt not in {"JPEG", "PNG", "WEBP"}:
        raise ValidationError(f"Unsupported image format: {fmt}")
    if w < 8 or h < 8:
        raise ValidationError("Image dimensions are too small")
    if w * h > 40_000_000:
        raise ValidationError("Image resolution is too large")
    return w, h, fmt


def resize_image(path: Path, out: Path, max_side: int = 1280) -> Path:
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        im.save(out, format="JPEG", quality=88, optimize=True)
    return out


def validate_audio(path: Path, max_mb: int, max_sec: int) -> float:
    if not path.exists():
        raise ValidationError("Audio file not found")
    if path.stat().st_size > max_mb * 1024 * 1024:
        raise ValidationError(f"Audio is larger than {max_mb} MB")
    duration = probe_duration(path)
    if duration <= 0:
        raise ValidationError("Audio duration could not be read")
    if duration < 2.0:
        raise ValidationError("Reference audio must be at least 2 seconds")
    if duration > max_sec:
        raise ValidationError(f"Reference audio must be at most {max_sec} seconds")
    return duration


def _read_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    p = Path("/proc/meminfo")
    if not p.exists():
        return info
    for line in p.read_text(errors="ignore").splitlines():
        k, _, rest = line.partition(":")
        parts = rest.strip().split()
        if parts and parts[0].isdigit():
            info[k] = int(parts[0]) * 1024
    return info


def system_snapshot(data_dir: Path | None = None) -> dict:
    mem = _read_meminfo()
    disk = shutil.disk_usage(str(data_dir or Path.cwd()))
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "ram_total_gb": round(mem.get("MemTotal", 0) / 1e9, 2),
        "ram_free_gb": round(mem.get("MemAvailable", 0) / 1e9, 2),
        "disk_free_gb": round(disk.free / 1e9, 2),
        "torch": "not installed",
        "cuda": False,
        "ffmpeg": ffmpeg_available(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return info


@dataclass
class Job:
    user_id: int
    chat_id: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timeout: float = 900.0
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    status_message_id: int | None = None
    result: Any = None
    error: str | None = None


class TaskQueue:
    def __init__(self, max_size: int, workers: int = 1):
        self._q: asyncio.Queue[Job] = asyncio.Queue(maxsize=max_size)
        self._workers = max(1, workers)
        self._handlers: dict[str, Callable[[Job], Awaitable[None]]] = {}
        self._tasks: list[asyncio.Task] = []
        self._running = False

    def register(self, kind: str, handler: Callable[[Job], Awaitable[None]]) -> None:
        self._handlers[kind] = handler

    @property
    def size(self) -> int:
        return self._q.qsize()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks = [asyncio.create_task(self._worker(i), name=f"qworker-{i}")
                       for i in range(self._workers)]

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def submit(self, job: Job) -> bool:
        if job.kind not in self._handlers:
            raise KeyError(f"No handler registered for job kind '{job.kind}'")
        try:
            self._q.put_nowait(job)
            return True
        except asyncio.QueueFull:
            return False

    async def _worker(self, index: int) -> None:
        while self._running:
            try:
                job = await self._q.get()
            except asyncio.CancelledError:
                return
            try:
                if job.cancel_event.is_set():
                    job.error = "Cancelled before start"
                    continue
                handler = self._handlers[job.kind]
                await asyncio.wait_for(handler(job), timeout=job.timeout)
            except asyncio.TimeoutError:
                job.error = f"Job timed out after {job.timeout:.0f}s"
                logger.warning("Job %s timed out", job.job_id)
            except asyncio.CancelledError:
                job.error = "Cancelled"
                raise
            except Exception as exc:
                job.error = str(exc)
                logger.exception("Job %s failed", job.job_id)
            finally:
                self._q.task_done()


async def _retry(call, attempts: int = 4, base_delay: float = 1.5):
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except RetryAfter as exc:
            wait = min(float(exc.retry_after), 30.0)
            logger.warning("Telegram flood control, waiting %.1fs", wait)
            await asyncio.sleep(wait)
            last = exc
        except TelegramError as exc:
            last = exc
            if attempt == attempts:
                break
            await asyncio.sleep(base_delay * attempt)
    if last is not None:
        raise last
    raise RuntimeError("Telegram call failed without a concrete error")


def _caption(title: str, i: int, total: int) -> str:
    return f"{title} — Part {i}/{total}" if total > 1 else title


async def deliver_voice(bot: Bot, chat_id: int, audio_path: Path, title: str,
                        settings: Settings,
                        status_message: Message | None = None) -> None:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    
    # 1. Pehle OGG mein convert karo
    ogg = to_ogg_opus(audio_path, audio_path.with_suffix(".ogg"),
                      ffmpeg_bin=settings.ffmpeg_bin)
    
    # 2. Agar OGG file size limit ke andar hai, seedha bhej do
    if ogg.stat().st_size <= max_bytes:
        with ogg.open("rb") as fh:
            await _retry(lambda fh=fh: bot.send_voice(
                chat_id=chat_id, voice=fh,
                caption=title,
                read_timeout=180, write_timeout=180,
            ))
        return
        
    # 3. Agar limit se bada hai, toh ORIGINAL WAV ko split karo
    parts = split_wav(audio_path, audio_path.parent, max_bytes)
    total = len(parts)
    
    for i, part in enumerate(parts, start=1):
        # Har WAV chunk ko OGG mein convert karo
        part_ogg = to_ogg_opus(part, part.with_suffix(".ogg"), ffmpeg_bin=settings.ffmpeg_bin)
        with part_ogg.open("rb") as fh:
            await _retry(lambda fh=fh, i=i: bot.send_voice(
                chat_id=chat_id, voice=fh,
                caption=_caption(title, i, total),
                read_timeout=180, write_timeout=180,
            ))
        if status_message is not None and total > 1:
            await status_message.edit_text(f"Sending part {i}/{total}…")


async def deliver_document(bot: Bot, chat_id: int, path: Path,
                           caption: str, settings: Settings) -> None:
    size = path.stat().st_size
    if size > settings.max_upload_mb * 1024 * 1024:
        raise ValidationError(
            f"File is {size / 1024 / 1024:.1f} MB, exceeds "
            f"{settings.max_upload_mb} MB delivery limit")
    with path.open("rb") as fh:
        await _retry(lambda fh=fh: bot.send_document(
            chat_id=chat_id, document=fh, caption=caption,
            read_timeout=180, write_timeout=180,
        ))
