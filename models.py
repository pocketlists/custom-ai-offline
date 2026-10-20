from __future__ import annotations

import gc
import logging
import re
import subprocess
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
    # ---------- Piper (Fast CPU) ----------
    "piper-hi-rohan": ModelSpec(
        key="piper-hi-rohan",
        display_name="Piper Hindi (Rohan, Male)",
        category=Category.TTS,
        repo_id="rhasspy/piper-voices",
        license="CC-BY-NC-SA-4.0 (Non-commercial)",
        dependencies=["piper-tts", "onnxruntime"],
        download_size_mb=60,
        cpu_compatibility="excellent",
        memory_notes="~200 MB RAM. Faster than real-time on CPU.",
        languages=["hi"],
        status=ModelStatus.UNTESTED,
        notes="Lightweight ONNX voice. Personal/research use only.",
    ),
    "piper-hi-priyamvada": ModelSpec(
        key="piper-hi-priyamvada",
        display_name="Piper Hindi (Priyamvada, Female)",
        category=Category.TTS,
        repo_id="rhasspy/piper-voices",
        license="CC-BY-NC-SA-4.0 (Non-commercial)",
        dependencies=["piper-tts", "onnxruntime"],
        download_size_mb=60,
        cpu_compatibility="excellent",
        memory_notes="~200 MB RAM. Faster than real-time on CPU.",
        languages=["hi"],
        status=ModelStatus.UNTESTED,
        notes="Lightweight ONNX voice. Personal/research use only.",
    ),
    # ---------- Indic Parler-TTS (Heavy) ----------
    "indic-parler-tts": ModelSpec(
        key="indic-parler-tts",
        display_name="Indic Parler-TTS (Hindi)",
        category=Category.TTS,
        repo_id="ai4bharat/indic-parler-tts",
        license="Apache-2.0 (Verify model card)",
        dependencies=["transformers", "torch", "parler-tts"],
        download_size_mb=3800,
        cpu_compatibility="slow",
        memory_notes="~4-6 GB RAM in float32. May take minutes per sentence on CPU.",
        languages=["hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or"],
        status=ModelStatus.TOO_SLOW,
        experimental=True,
        notes="Gated model. Requires HF_TOKEN. Use voice description.",
    ),
    # ---------- VoxCPM2 (Very Heavy) ----------
    "voxcpm2": ModelSpec(
        key="voxcpm2",
        display_name="VoxCPM2 (Hindi)",
        category=Category.TTS,
        repo_id="openbmb/VoxCPM2",
        license="Apache-2.0",
        dependencies=["voxcpm", "torch", "soundfile"],
        download_size_mb=8000,
        cpu_compatibility="very_slow",
        memory_notes="~8-10 GB RAM in float32. Several minutes per short phrase on CPU.",
        languages=["hi", "en", "ta", "te", "bn", "gu", "kn", "ml", "mr", "pa", "or"],
        status=ModelStatus.TOO_SLOW,
        experimental=True,
        notes="2B diffusion model. Real-time on GPU, not on 4-core CPU.",
    ),
    # ---------- VEXYL-TTS (API Client) ----------
    "vexyl-tts": ModelSpec(
        key="vexyl-tts",
        display_name="VEXYL-TTS (External Server)",
        category=Category.TTS,
        repo_id="vexyl-ai/vexyl-tts",
        license="Apache-2.0",
        dependencies=["requests"],
        download_size_mb=0,
        cpu_compatibility="unknown",
        memory_notes="Requires externally hosted VEXYL-TTS server.",
        languages=["hi", "ta", "te", "kn", "ml", "bn", "gu", "mr", "pa", "or"],
        status=ModelStatus.NEEDS_VERIFICATION,
        experimental=True,
        notes="Client adapter only. Set VEXYL_TTS_URL env variable.",
    ),
    # ---------- Chat ----------
    "qwen-chat-0.5b": ModelSpec(
        key="qwen-chat-0.5b", display_name="Qwen2.5 0.5B Instruct",
        category=Category.CHAT, repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        license="Apache-2.0", dependencies=["transformers", "torch", "accelerate"],
        download_size_mb=1000, cpu_compatibility="good",
        memory_notes="~2 GB RAM in float32", languages=["en", "zh", "hi"],
        status=ModelStatus.UNTESTED,
    ),
    "qwen-chat-1.5b": ModelSpec(
        key="qwen-chat-1.5b", display_name="Qwen2.5 1.5B Instruct",
        category=Category.CHAT, repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        license="Apache-2.0", dependencies=["transformers", "torch", "accelerate"],
        download_size_mb=3000, cpu_compatibility="slow",
        memory_notes="~4-6 GB RAM in float32", languages=["en", "zh", "hi"],
        status=ModelStatus.UNTESTED,
    ),
    # ---------- Vision ----------
    "qwen-vl-3b": ModelSpec(
        key="qwen-vl-3b", display_name="Qwen2.5-VL 3B Instruct",
        category=Category.VISION, repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        license="Apache-2.0",
        dependencies=["transformers", "qwen-vl-utils", "torch", "torchvision", "accelerate"],
        download_size_mb=7000, cpu_compatibility="very_slow",
        memory_notes="~8-10 GB RAM in float32", languages=["en", "zh"],
        status=ModelStatus.UNTESTED, experimental=True,
    ),
    # ---------- Voice Cloning ----------
    "openvoice-v2": ModelSpec(
        key="openvoice-v2", display_name="OpenVoice V2",
        category=Category.VOICE_CLONING, repo_id="myshell-ai/OpenVoiceV2",
        license="MIT (Verify checkpoints)",
        dependencies=["openvoice", "melo-tts", "torch", "soundfile", "librosa"],
        download_size_mb=2500, cpu_compatibility="slow",
        memory_notes="Slower than real-time on CPU", languages=["en"],
        status=ModelStatus.NEEDS_VERIFICATION, experimental=True,
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
    clipped = np.clip(samples, -1.0, 1.0)
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


# ============================================================
# Piper Backend
# ============================================================
class PiperBackend:
    def __init__(self, settings, spec):
        self.settings = settings
        self.spec = spec
        self._voice_dir = None
        self._piper = None
        self._model_path = None

    def load(self):
        try:
            import piper
            import onnxruntime
        except ImportError as exc:
            raise DependencyMissing(
                "piper-tts and onnxruntime are required. "
                "Run: pip install piper-tts onnxruntime"
            ) from exc
        self._piper = piper
        self._voice_dir = Path(self.settings.data_dir) / "piper_voices"
        self._voice_dir.mkdir(parents=True, exist_ok=True)
        voice_name = "hi_IN-rohan-medium" if "rohan" in self.spec.key else "hi_IN-priyamvada-medium"
        model_path = self._voice_dir / f"{voice_name}.onnx"
        
        if not model_path.exists():
            self._download_voice(voice_name)
        self._model_path = model_path

    def _download_voice(self, voice_name):
        import urllib.request
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/{}/medium/{}.onnx"
        url = base_url.format(voice_name, voice_name)
        json_url = url + ".json"
        model_path = self._voice_dir / f"{voice_name}.onnx"
        json_path = self._voice_dir / f"{voice_name}.onnx.json"
        urllib.request.urlretrieve(url, model_path)
        urllib.request.urlretrieve(json_url, json_path)

    def unload(self):
        self._piper = None
        self._model_path = None

    def synthesize(self, text, output_path, language="hi", voice=None, speed=1.0, description=None):
        if self._piper is None or self._model_path is None:
            raise ModelError("Piper not loaded")
        cleaned = text.strip()
        if not cleaned:
            raise ValidationError("Text is empty")
        started = time.monotonic()
        
        cmd = [
            "piper",
            "--model", str(self._model_path),
            "--output_file", str(output_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=cleaned.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=True,
            )
        except FileNotFoundError:
            voice_obj = self._piper.PiperVoice.load(str(self._model_path))
            with wave.open(str(output_path), "wb") as wav:
                voice_obj.synthesize(cleaned, wav)
        except subprocess.CalledProcessError as exc:
            raise ModelError(f"Piper failed: {exc.stderr.decode(errors='replace')[:300]}")
            
        duration = _verify_wav(output_path)
        return {
            "path": output_path,
            "duration": duration,
            "language": "hi",
            "voice": self.spec.key,
            "elapsed": time.monotonic() - started,
            "sample_rate": 22050,
        }


# ============================================================
# Parler-TTS Backend
# ============================================================
class ParlerBackend:
    def __init__(self, settings, spec):
        self.settings = settings
        self.spec = spec
        self._model = None
        self._tokenizer = None
        self._torch = None

    def load(self):
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise DependencyMissing("transformers and torch are required for Parler-TTS.") from exc
        torch.set_num_threads(self.settings.cpu_threads)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.spec.repo_id, token=self.settings.hf_token)
            try:
                from parler_tts import ParlerTTSForConditionalGeneration
                self._model = ParlerTTSForConditionalGeneration.from_pretrained(
                    self.spec.repo_id, token=self.settings.hf_token, torch_dtype=torch.float32
                ).to("cpu").eval()
            except ImportError:
                raise DependencyMissing("parler-tts package required. Run: pip install parler-tts")
            self._torch = torch
        except Exception as exc:
            raise ModelError(f"Parler load failed: {exc}") from exc

    def unload(self):
        self._model = None
        self._tokenizer = None
        self._torch = None

    def synthesize(self, text, output_path, language="hi", voice=None, speed=1.0, description=None):
        if self._model is None or self._tokenizer is None:
            raise ModelError("Parler not loaded")
        cleaned = text.strip()
        if not cleaned:
            raise ValidationError("Text is empty")
        description = description or "A clear Hindi voice, moderate pace, no background noise."
        started = time.monotonic()
        try:
            inputs = self._tokenizer(cleaned, return_tensors="pt")
            description_ids = self._tokenizer(description, return_tensors="pt")
            with self._torch.inference_mode():
                generation = self._model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    prompt_input_ids=description_ids.input_ids,
                    prompt_attention_mask=description_ids.attention_mask,
                )
            audio = generation.cpu().numpy().squeeze()
        except Exception as exc:
            raise ModelError(f"Parler inference failed: {exc}") from exc
            
        _write_wav(output_path, audio, 44100)
        duration = _verify_wav(output_path)
        return {
            "path": output_path, "duration": duration, "language": "hi",
            "voice": "parler", "elapsed": time.monotonic() - started, "sample_rate": 44100,
        }


# ============================================================
# VoxCPM Backend
# ============================================================
class VoxCPMBackend:
    def __init__(self, settings, spec):
        self.settings = settings
        self.spec = spec
        self._model = None
        self._sf = None

    def load(self):
        try:
            from voxcpm import VoxCPM
            import soundfile as sf
        except ImportError as exc:
            raise DependencyMissing("voxcpm and soundfile required. Run: pip install voxcpm soundfile") from exc
        try:
            self._model = VoxCPM.from_pretrained(self.spec.repo_id, device="cpu", optimize=False)
            self._sf = sf
        except Exception as exc:
            raise ModelError(f"VoxCPM load failed: {exc}") from exc

    def unload(self):
        self._model = None
        self._sf = None

    def synthesize(self, text, output_path, language="hi", voice=None, speed=1.0, description=None):
        if self._model is None:
            raise ModelError("VoxCPM not loaded")
        cleaned = text.strip()
        if not cleaned:
            raise ValidationError("Text is empty")
        started = time.monotonic()
        try:
            audio = self._model.generate(text=cleaned, language="hi", max_length=500)
            self._sf.write(str(output_path), audio, 48000)
        except Exception as exc:
            raise ModelError(f"VoxCPM inference failed: {exc}") from exc
            
        duration = _verify_wav(output_path)
        return {
            "path": output_path, "duration": duration, "language": "hi",
            "voice": "voxcpm2", "elapsed": time.monotonic() - started, "sample_rate": 48000,
        }


# ============================================================
# Vexyl Backend (External API)
# ============================================================
class VexylBackend:
    def __init__(self, settings, spec):
        self.settings = settings
        self.spec = spec
        self._base_url = None

    def load(self):
        import os
        self._base_url = os.environ.get("VEXYL_TTS_URL", "").rstrip("/")
        if not self._base_url:
            raise ModelError("VEXYL_TTS_URL env var not set.")

    def unload(self):
        self._base_url = None

    def synthesize(self, text, output_path, language="hi", voice=None, speed=1.0, description=None):
        if not self._base_url:
            raise ModelError("VEXYL base URL not configured")
        import requests
        cleaned = text.strip()
        if not cleaned:
            raise ValidationError("Text is empty")
        started = time.monotonic()
        payload = {"text": cleaned, "language": "hi", "speaker": voice or "default"}
        try:
            resp = requests.post(f"{self._base_url}/synthesize", json=payload, timeout=120)
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
        except Exception as exc:
            raise ModelError(f"VEXYL request failed: {exc}") from exc
            
        duration = _verify_wav(output_path)
        return {
            "path": output_path, "duration": duration, "language": "hi",
            "voice": "vexyl", "elapsed": time.monotonic() - started, "sample_rate": 44100,
        }


# ============================================================
# Chat Backend
# ============================================================
class ChatBackend:
    def __init__(self, settings, spec):
        self.settings = settings
        self.spec = spec
        self._model = None
        self._tokenizer = None
        self._torch = None

    def load(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise DependencyMissing(f"transformers/torch missing ({exc})") from exc
        torch.set_num_threads(self.settings.cpu_threads)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.repo_id, trust_remote_code=False)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.spec.repo_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=False
            ).to("cpu").eval()
            self._torch = torch
        except Exception as exc:
            raise ModelError(f"{self.spec.display_name} load failed: {exc}") from exc

    def unload(self):
        self._model = None
        self._tokenizer = None
        self._torch = None

    def generate(self, messages, max_new_tokens=256, temperature=0.7, top_p=0.9):
        if self._model is None or self._tokenizer is None:
            raise ModelError("Chat model not loaded")
        if not messages or messages[-1].get("role") != "user":
            raise ValidationError("Last message must have role 'user'")
        if not 16 <= max_new_tokens <= 1024:
            raise ValidationError("max_new_tokens must be 16-1024")
        started = time.monotonic()
        try:
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_len = int(inputs["input_ids"].shape[-1])
            with self._torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 0.01),
                    top_p=top_p,
                    repetition_penalty=1.08,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        except Exception as exc:
            raise ModelError(f"Generation failed: {exc}") from exc
        generated = output[0][input_len:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise ModelError("Model returned an empty response")
        return {
            "text": text, "elapsed": time.monotonic() - started,
            "prompt_tokens": input_len, "completion_tokens": int(generated.shape[-1]),
        }


# ============================================================
# Vision Backend
# ============================================================
class VisionBackend:
    def __init__(self, settings, spec):
        self.settings = settings
        self.spec = spec
        self._model = None
        self._processor = None
        self._torch = None

    def load(self):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise DependencyMissing(f"Vision deps missing: {exc}") from exc
        torch.set_num_threads(self.settings.cpu_threads)
        try:
            self._processor = AutoProcessor.from_pretrained(self.spec.repo_id)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.spec.repo_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
            ).to("cpu").eval()
            self._torch = torch
        except Exception as exc:
            raise ModelError(f"{self.spec.display_name} load failed: {exc}") from exc

    def unload(self):
        self._model = None
        self._processor = None
        self._torch = None

    def describe(self, image_path, question, max_new_tokens=128):
        if self._model is None or self._processor is None:
            raise ModelError("Vision model not loaded")
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise DependencyMissing("qwen-vl-utils missing") from exc
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question.strip() or "Describe this image briefly."},
            ],
        }]
        started = time.monotonic()
        try:
            prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            with self._torch.inference_mode():
                generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
            input_ids = inputs["input_ids"]
            trimmed = [generated[i][int(input_ids[i].shape[-1]):] for i in range(generated.shape[0])]
            answer = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        except Exception as exc:
            raise ModelError(f"Vision inference failed: {exc}") from exc
        if not answer:
            raise ModelError("Vision model returned empty answer")
        return {"text": answer, "elapsed": time.monotonic() - started}


# ============================================================
# OpenVoice Backend
# ============================================================
class OpenVoiceBackend:
    def __init__(self, settings, spec):
        self.settings = settings
        self.spec = spec
        self._converter = None
        self._base_tts = None
        self._se = None
        self._torch = None

    def load(self):
        import sys
        if self.settings.openvoice_repo:
            repo = Path(self.settings.openvoice_repo).expanduser().resolve()
            if not repo.exists():
                raise ModelError("OPENVOICE_REPO_DIR does not exist")
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
        try:
            import torch
            from melo.api import TTS as MeloTTS
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
        except ImportError as exc:
            raise DependencyMissing("OpenVoice deps missing.") from exc
        if not self.settings.openvoice_ckpt:
            raise ModelError("OPENVOICE_CHECKPOINT_DIR not set.")
        ckpt = Path(self.settings.openvoice_ckpt).expanduser().resolve()
        conv_dir = ckpt / "converter"
        if not (conv_dir / "config.json").exists():
            raise ModelError(f"OpenVoice converter not found at {conv_dir}")
        torch.set_num_threads(self.settings.cpu_threads)
        try:
            self._converter = ToneColorConverter(str(conv_dir / "config.json"), device="cpu")
            self._converter.load_ckpt(str(conv_dir / "checkpoint.pth"))
            self._base_tts = MeloTTS(language="EN", device="cpu")
            self._se = se_extractor
            self._torch = torch
        except Exception as exc:
            raise ModelError(f"OpenVoice load failed: {exc}") from exc

    def unload(self):
        self._converter = None
        self._base_tts = None
        self._se = None
        self._torch = None

    def clone(self, text, reference_wav, output_path, language="EN"):
        if language != "EN":
            raise ValidationError("Only English base speaker is enabled")
        cleaned = text.strip()
        if not cleaned:
            raise ValidationError("Text is empty")
        if not Path(reference_wav).exists():
            raise ValidationError("Reference audio missing")
        started = time.monotonic()
        temp_base = Path(str(output_path) + ".base.wav")
        try:
            target_se, _ = self._se.get_se(str(reference_wav), self._converter, vad=True)
            speakers = self._base_tts.hps.data.spk2id
            spk = speakers.get("EN-Default") or next(iter(speakers.values()))
            self._base_tts.tts_to_file(cleaned, spk, str(temp_base), speed=1.0)
            source_se, _ = self._se.get_se(str(temp_base), self._converter, vad=True)
            self._converter.convert(
                audio_src_path=str(temp_base),
                src_se=source_se, tgt_se=target_se,
                output_path=str(output_path), message="@cpu-bot",
            )
        except Exception as exc:
            raise ModelError(f"Voice cloning failed: {exc}") from exc
        finally:
            temp_base.unlink(missing_ok=True)
        if not Path(output_path).exists() or Path(output_path).stat().st_size < 1024:
            raise ModelError("Voice cloning produced no audio")
        return {"path": Path(output_path), "elapsed": time.monotonic() - started}


_FACTORIES = {
    "piper-hi-rohan": PiperBackend,
    "piper-hi-priyamvada": PiperBackend,
    "indic-parler-tts": ParlerBackend,
    "voxcpm2": VoxCPMBackend,
    "vexyl-tts": VexylBackend,
    "qwen-chat-0.5b": ChatBackend,
    "qwen-chat-1.5b": ChatBackend,
    "qwen-vl-3b": VisionBackend,
    "openvoice-v2": OpenVoiceBackend,
}
