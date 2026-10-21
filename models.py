from __future__ import annotations

import gc
import logging
import os
import threading
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from config import Settings

logging.getLogger("phonemizer").setLevel(logging.ERROR)


class ModelStatus(str, Enum):
    UNTESTED = "UNTESTED"
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_VERIFICATION = "NEEDS_VERIFICATION"
    TOO_LARGE = "TOO_LARGE_FOR_RUNNER"
    TOO_SLOW = "TOO_SLOW_FOR_CPU"
    SKIPPED = "SKIPPED"


class Category(str, Enum):
    TTS = "tts"
    STT = "stt"
    CHAT = "chat"
    VISION = "vision"
    VOICE_CLONING = "voice_cloning"


class ModelError(RuntimeError):
    pass


class DependencyMissing(ModelError):
    pass


class ValidationError(ModelError):
    pass


@dataclass
class ModelSpec:
    key: str
    display_name: str
    category: Category
    repo_id: str | None
    license: str
    dependencies: list[str] = field(default_factory=list)
    download_size_mb: int | None = None
    cpu_compatibility: str = "unknown"
    memory_notes: str = ""
    languages: list[str] = field(default_factory=list)
    status: ModelStatus = ModelStatus.UNTESTED
    last_error: str | None = None
    experimental: bool = False
    notes: str = ""

    def public(self):
        return {
            "key": self.key,
            "display_name": self.display_name,
            "category": self.category.value,
            "license": self.license,
            "download_size_mb": self.download_size_mb,
            "memory_notes": self.memory_notes,
            "languages": self.languages,
            "status": self.status.value,
            "last_error": self.last_error,
            "experimental": self.experimental,
            "notes": self.notes,
        }


REGISTRY: dict[str, ModelSpec] = {
    # ================= TTS =================
    "kokoro": ModelSpec(
        key="kokoro",
        display_name="⭐ Kokoro-82M — Best Quality (EN + HI)",
        category=Category.TTS,
        repo_id="hexgrad/Kokoro-82M",
        license="Apache-2.0",
        dependencies=["kokoro-onnx", "onnxruntime"],
        download_size_mb=340,
        cpu_compatibility="excellent",
        memory_notes="~400 MB RAM. CPU par real-time se bhi tez.",
        languages=["en", "hi"],
        notes="Sabse best CPU voice. EN voices: af_heart, af_bella, am_adam. "
              "HI voices: hf_alpha, hf_beta, hm_alpha. Format: `en:am_adam text` ya `hi text`.",
    ),
    "piper-hi-rohan": ModelSpec(
        key="piper-hi-rohan",
        display_name="Piper Hindi — Rohan (Male) ⚡Fast",
        category=Category.TTS,
        repo_id="rhasspy/piper-voices",
        license="CC-BY-NC-SA-4.0 (Non-commercial)",
        dependencies=["piper-tts", "onnxruntime"],
        download_size_mb=60,
        cpu_compatibility="excellent",
        memory_notes="~200 MB RAM. Real-time se tez.",
        languages=["hi"],
        notes="Halka ONNX voice. Personal/research use.",
    ),
    "piper-hi-priyamvada": ModelSpec(
        key="piper-hi-priyamvada",
        display_name="Piper Hindi — Priyamvada (Female) ⚡Fast",
        category=Category.TTS,
        repo_id="rhasspy/piper-voices",
        license="CC-BY-NC-SA-4.0 (Non-commercial)",
        dependencies=["piper-tts", "onnxruntime"],
        download_size_mb=60,
        cpu_compatibility="excellent",
        memory_notes="~200 MB RAM. Real-time se tez.",
        languages=["hi"],
        notes="Halka ONNX voice. Personal/research use.",
    ),
    "piper-hi-pratham": ModelSpec(
        key="piper-hi-pratham",
        display_name="Piper Hindi — Pratham (Male) ⚡Fast",
        category=Category.TTS,
        repo_id="rhasspy/piper-voices",
        license="CC-BY-NC-SA-4.0 (Non-commercial)",
        dependencies=["piper-tts", "onnxruntime"],
        download_size_mb=60,
        cpu_compatibility="excellent",
        memory_notes="~200 MB RAM. Real-time se tez.",
        languages=["hi"],
        notes="Halka ONNX voice. Personal/research use.",
    ),
    "piper-en-amy": ModelSpec(
        key="piper-en-amy",
        display_name="Piper English — Amy (Female, US) ⚡Fast",
        category=Category.TTS,
        repo_id="rhasspy/piper-voices",
        license="MIT-style (voice data varies)",
        dependencies=["piper-tts", "onnxruntime"],
        download_size_mb=60,
        cpu_compatibility="excellent",
        memory_notes="~200 MB RAM. Real-time se tez.",
        languages=["en"],
    ),
    "piper-en-lessac": ModelSpec(
        key="piper-en-lessac",
        display_name="Piper English — Lessac (Female, US) ⚡Fast",
        category=Category.TTS,
        repo_id="rhasspy/piper-voices",
        license="MIT-style (voice data varies)",
        dependencies=["piper-tts", "onnxruntime"],
        download_size_mb=60,
        cpu_compatibility="excellent",
        memory_notes="~200 MB RAM. Real-time se tez.",
        languages=["en"],
        notes="Amy se thodi zyada saaf awaaz.",
    ),
    "piper-en-alan": ModelSpec(
        key="piper-en-alan",
        display_name="Piper English — Alan (Male, UK) ⚡Fast",
        category=Category.TTS,
        repo_id="rhasspy/piper-voices",
        license="MIT-style (voice data varies)",
        dependencies=["piper-tts", "onnxruntime"],
        download_size_mb=60,
        cpu_compatibility="excellent",
        memory_notes="~200 MB RAM. Real-time se tez.",
        languages=["en"],
    ),
    "mms-hi": ModelSpec(
        key="mms-hi",
        display_name="MMS Hindi (Meta) — Natural",
        category=Category.TTS,
        repo_id="facebook/mms-tts-hin",
        license="CC-BY-NC-4.0 (Non-commercial)",
        dependencies=["transformers", "torch"],
        download_size_mb=130,
        cpu_compatibility="good",
        memory_notes="~600 MB RAM. Ek line ke liye kuch seconds.",
        languages=["hi"],
        notes="Hindi text Devanagari me bhejo (जैसे: नमस्ते दुनिया). Roman Hindi par quality girti hai.",
    ),
    "mms-en": ModelSpec(
        key="mms-en",
        display_name="MMS English (Meta)",
        category=Category.TTS,
        repo_id="facebook/mms-tts-eng",
        license="CC-BY-NC-4.0 (Non-commercial)",
        dependencies=["transformers", "torch"],
        download_size_mb=130,
        cpu_compatibility="good",
        memory_notes="~600 MB RAM.",
        languages=["en"],
    ),
    "indic-parler-tts": ModelSpec(
        key="indic-parler-tts",
        display_name="Indic Parler-TTS (Hindi) 🐌 Heavy",
        category=Category.TTS,
        repo_id="ai4bharat/indic-parler-tts",
        license="Apache-2.0 (Verify model card)",
        dependencies=["transformers", "torch", "parler-tts"],
        download_size_mb=3800,
        cpu_compatibility="slow",
        memory_notes="~4-6 GB RAM float32. CPU par minutes per sentence.",
        languages=["hi"],
        status=ModelStatus.TOO_SLOW,
        experimental=True,
        notes="Gated model, HF_TOKEN chahiye. Menu me hidden hai (TOO_SLOW).",
    ),
    # ================= STT (Voice -> Text) =================
    "whisper-base": ModelSpec(
        key="whisper-base",
        display_name="Whisper Base ⚡Fast (HI + EN)",
        category=Category.STT,
        repo_id="Systran/faster-whisper-base",
        license="MIT",
        dependencies=["faster-whisper"],
        download_size_mb=145,
        cpu_compatibility="excellent",
        memory_notes="~1 GB RAM (int8). Chhoti voice notes ke liye best.",
        languages=["hi", "en"],
        notes="Voice note bhejo, text ban jayega. Language auto-detect hota hai.",
    ),
    "whisper-small": ModelSpec(
        key="whisper-small",
        display_name="⭐ Whisper Small — Zyada Accurate (HI + EN)",
        category=Category.STT,
        repo_id="Systran/faster-whisper-small",
        license="MIT",
        dependencies=["faster-whisper"],
        download_size_mb=490,
        cpu_compatibility="good",
        memory_notes="~2 GB RAM (int8). Hindi par Base se clearly better.",
        languages=["hi", "en"],
        notes="Lambi ya saaf accuracy wali transcription ke liye.",
    ),
    # ================= CHAT =================
    "qwen-chat-0.5b": ModelSpec(
        key="qwen-chat-0.5b", display_name="Qwen2.5 0.5B ⚡Fast Chat",
        category=Category.CHAT, repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        license="Apache-2.0", dependencies=["transformers", "torch", "accelerate"],
        download_size_mb=1000, cpu_compatibility="good",
        memory_notes="~1.5 GB RAM (int8 dynamic). Chhote sawal-jawab ke liye.",
        languages=["en", "hi"],
    ),
    "qwen-chat-1.5b": ModelSpec(
        key="qwen-chat-1.5b", display_name="⭐ Qwen2.5 1.5B — Smart Chat",
        category=Category.CHAT, repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        license="Apache-2.0", dependencies=["transformers", "torch", "accelerate"],
        download_size_mb=3100, cpu_compatibility="slow",
        memory_notes="~3-4 GB RAM (int8 dynamic). Jawab slow par samajhdar.",
        languages=["en", "hi"],
    ),
    "hf-api-chat": ModelSpec(
        key="hf-api-chat", display_name="☁️ HF API — Qwen2.5 7B (Online, Fast)",
        category=Category.CHAT, repo_id="Qwen/Qwen2.5-7B-Instruct",
        license="Apache-2.0", dependencies=["huggingface_hub"],
        download_size_mb=0, cpu_compatibility="excellent",
        memory_notes="Download 0 MB. Hugging Face server par chalta hai — CPU free.",
        languages=["en", "hi"],
        notes="GitHub secret HF_TOKEN zaroori hai. Internet chahiye; local models se tez aur smart.",
    ),
    # ================= VISION =================
    "moondream2": ModelSpec(
        key="moondream2", display_name="⭐ Moondream2 — Fast Vision",
        category=Category.VISION, repo_id="vikhyatk/moondream2",
        license="Apache-2.0",
        dependencies=["transformers", "torch", "timm", "einops"],
        download_size_mb=3800, cpu_compatibility="good",
        memory_notes="~4-5 GB RAM float32. Photo ka jawab ~10-40s me.",
        languages=["en"],
        notes="Photo + sawal bhejo. English me jawab deta hai.",
    ),
    "smolvlm2": ModelSpec(
        key="smolvlm2", display_name="SmolVLM2 2.2B — Quality Vision 🐌",
        category=Category.VISION, repo_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        license="Apache-2.0",
        dependencies=["transformers", "torch", "accelerate"],
        download_size_mb=4600, cpu_compatibility="slow",
        memory_notes="~5-6 GB RAM float32. Moondream se slow, jawab zyada detailed.",
        languages=["en"],
        experimental=True,
    ),
    # ================= VOICE CLONING =================
    "xtts-v2": ModelSpec(
        key="xtts-v2", display_name="⭐ XTTS v2 — Voice Cloning (HI + EN)",
        category=Category.VOICE_CLONING, repo_id="coqui/XTTS-v2",
        license="CPML (Non-commercial only)",
        dependencies=["coqui-tts", "torch", "soundfile"],
        download_size_mb=1900, cpu_compatibility="slow",
        memory_notes="~4 GB RAM. 6 sec sample se clone; CPU par ek line ko ~30-90s lag sakte hain.",
        languages=["hi", "en"],
        experimental=True,
        notes="requirements.txt me coqui-tts wali line uncomment karke install karo. "
              "Text se pehle `hi:` ya `en:` lagao language ke liye.",
    ),
}


def get_spec(key):
    try:
        return REGISTRY[key]
    except KeyError as exc:
        raise ValidationError(f"Unknown model key: {key}") from exc


def by_category(cat):
    cat = Category(cat) if isinstance(cat, str) else cat
    return [s for s in REGISTRY.values() if s.category == cat]


def selectable(cat):
    return [s for s in by_category(cat)
            if s.status not in {ModelStatus.NEEDS_VERIFICATION,
                                ModelStatus.TOO_LARGE,
                                ModelStatus.TOO_SLOW}]


def _write_wav(path, samples, rate):
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def _verify_wav(path, min_sec=0.3):
    if not path.exists() or path.stat().st_size < 1024:
        raise ModelError("Generated audio missing or empty")
    with wave.open(str(path), "rb") as w:
        frames, rate = w.getnframes(), w.getframerate()
    if rate <= 0:
        raise ModelError("Audio has invalid sample rate")
    duration = frames / float(rate)
    if duration < min_sec:
        raise ModelError(f"Audio too short: {duration:.2f}s")
    return duration


def _download(url, dest: Path):
    """Simple resumable-enough download with a browser UA (GitHub/HF CDN)."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "cpu-bot/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)
    return dest


_LOAD_LOCK = threading.Lock()


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._active_key: str | None = None
        self._active = None

    @property
    def active_key(self):
        return self._active_key

    def _unload_locked(self):
        if self._active is not None:
            try:
                unload = getattr(self._active, "unload", None)
                if unload:
                    unload()
            except Exception:
                pass
            self._active = None
        self._active_key = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload(self):
        with _LOAD_LOCK:
            self._unload_locked()

    def get(self, key):
        with _LOAD_LOCK:
            if self._active_key == key and self._active is not None:
                return self._active
            self._unload_locked()
            spec = get_spec(key)
            factory = _FACTORIES.get(key)
            if factory is None:
                raise ValidationError(f"No backend registered for model '{key}'")
            instance = factory(self.settings, spec)
            instance.load()
            self._active_key = key
            self._active = instance
            return instance
