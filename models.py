from __future__ import annotations

import gc
import re
import threading
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from config import Settings


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
    VOICE_CLONING = "voice_cloning"
    CHAT = "chat"
    VISION = "vision"


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
    "kokoro": ModelSpec(
        key="kokoro", display_name="Kokoro 82M", category=Category.TTS,
        repo_id="hexgrad/Kokoro-82M", license="Apache-2.0",
        dependencies=["kokoro", "torch", "soundfile"],
        download_size_mb=330, cpu_compatibility="good",
        memory_notes="~2 GB RAM including PyTorch. Near real-time on modern CPUs.",
        languages=["en", "hi", "es", "fr", "it", "pt"],
        notes="Hindi voices: hf_alpha, hf_beta, hm_omega, hm_psi.",
    ),
    "mms-tts-hin": ModelSpec(
        key="mms-tts-hin", display_name="MMS-TTS Hindi", category=Category.TTS,
        repo_id="facebook/mms-tts-hin", license="CC-BY-NC-4.0",
        dependencies=["transformers", "torch", "soundfile"],
        download_size_mb=145, cpu_compatibility="good",
        memory_notes="Tiny VITS model, fast on CPU",
        languages=["hi"],
        notes="Non-commercial license. Use only for research/testing.",
    ),
    "qwen-chat-0.5b": ModelSpec(
        key="qwen-chat-0.5b", display_name="Qwen2.5 0.5B Instruct",
        category=Category.CHAT, repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        license="Apache-2.0", dependencies=["transformers", "torch", "accelerate"],
        download_size_mb=1000, cpu_compatibility="good",
        memory_notes="~2 GB RAM in float32",
        languages=["en", "zh", "hi"],
        notes="Recommended first chat model for CPU.",
    ),
    "qwen-chat-1.5b": ModelSpec(
        key="qwen-chat-1.5b", display_name="Qwen2.5 1.5B Instruct",
        category=Category.CHAT, repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        license="Apache-2.0", dependencies=["transformers", "torch", "accelerate"],
        download_size_mb=3000, cpu_compatibility="slow",
        memory_notes="~4-6 GB RAM in float32",
        languages=["en", "zh", "hi"],
        notes="Slower than 0.5B on CPU.",
    ),
    "granite-2b": ModelSpec(
        key="granite-2b", display_name="Granite 3.1 2B Instruct",
        category=Category.CHAT, repo_id="ibm-granite/granite-3.1-2b-instruct",
        license="Apache-2.0", dependencies=["transformers", "torch", "accelerate"],
        download_size_mb=5000, cpu_compatibility="slow",
        memory_notes="~6-8 GB RAM in float32",
        languages=["en"],
        notes="Manual test only.",
    ),
    "bonsai-1.7b": ModelSpec(
        key="bonsai-1.7b", display_name="Bonsai 1.7B",
        category=Category.CHAT, repo_id=None, license="unknown",
        cpu_compatibility="unknown", status=ModelStatus.NEEDS_VERIFICATION,
        notes="No verified official repository. Provide the exact model card.",
    ),
    "bonsai-4b": ModelSpec(
        key="bonsai-4b", display_name="Bonsai 4B",
        category=Category.CHAT, repo_id=None, license="unknown",
        cpu_compatibility="unknown", status=ModelStatus.NEEDS_VERIFICATION,
        notes="No verified official repository. Provide the exact model card.",
    ),
    "bonsai-8b": ModelSpec(
        key="bonsai-8b", display_name="Bonsai 8B",
        category=Category.CHAT, repo_id=None, license="unknown",
        cpu_compatibility="too_large", status=ModelStatus.NEEDS_VERIFICATION,
        notes="Likely too large for CPU runner without quantization.",
    ),
    "bonsai-27b": ModelSpec(
        key="bonsai-27b", display_name="Bonsai 27B",
        category=Category.CHAT, repo_id=None, license="unknown",
        cpu_compatibility="too_large", status=ModelStatus.TOO_LARGE,
        notes="Not practical on a 15 GiB CPU runner.",
    ),
    "qwen-vl-3b": ModelSpec(
        key="qwen-vl-3b", display_name="Qwen2.5-VL 3B Instruct",
        category=Category.VISION, repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        license="Apache-2.0",
        dependencies=["transformers", "qwen-vl-utils", "torch", "torchvision", "accelerate"],
        download_size_mb=7000, cpu_compatibility="very_slow",
        memory_notes="~8-10 GB RAM in float32, several minutes per image",
        languages=["en", "zh"], experimental=True,
        notes="Manual test only.",
    ),
    "openvoice-v2": ModelSpec(
        key="openvoice-v2", display_name="OpenVoice V2",
        category=Category.VOICE_CLONING, repo_id="myshell-ai/OpenVoiceV2",
        license="MIT (verify checkpoint license)",
        dependencies=["openvoice", "melo-tts", "torch", "soundfile", "librosa"],
        download_size_mb=2500, cpu_compatibility="slow",
        memory_notes="Slower than real-time on CPU",
        languages=["en"], status=ModelStatus.NEEDS_VERIFICATION,
        experimental=True,
        notes="Requires manual repo + checkpoint setup. Consent required.",
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


KOKORO_LANG = {"en": "a", "hi": "h", "es": "e", "fr": "f", "it": "i", "pt": "p"}
KOKORO_VOICES = {
    "en": ["af_heart", "af_bella", "am_michael", "am_adam", "bf_emma", "bm_george"],
    "hi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "es": ["ef_dora", "em_alex"],
    "fr": ["ff_siwis"],
    "it": ["if_sara", "im_nicola"],
    "pt": ["pf_dora", "pm_alex"],
}
DEFAULT_VOICE = {"en": "af_heart", "hi": "hf_alpha", "es": "ef_dora",
                 "fr": "ff_siwis", "it": "if_sara", "pt": "pf_dora"}


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


def _to_numpy(value):
    """Convert a value to a 1-D float32 numpy array, or return None if not audio-like."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray):
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return array


def _coerce_audio(item):
    """
    Kokoro (and similar TTS pipelines) can yield audio in several shapes:
      - A dataclass/namedtuple with `.audio` (Kokoro 0.9.4+ uses a dataclass)
      - A plain tuple `(graphemes, phonemes, audio_tensor)`
      - A dict with an "audio" key
      - A raw tensor/ndarray
    This helper picks the first valid audio payload it can find.
    """
    # 1. Object with `.audio` attribute (dataclass or namedtuple)
    audio_attr = getattr(item, "audio", None)
    if audio_attr is not None:
        array = _to_numpy(audio_attr)
        if array is not None:
            return array
        # Batched audio (list/tuple of tensors)
        if isinstance(audio_attr, (list, tuple)):
            parts = [p for p in (_to_numpy(v) for v in audio_attr) if p is not None]
            if parts:
                return np.concatenate(parts)

    # 2. Dict with "audio" key
    if isinstance(item, dict) and "audio" in item:
        array = _to_numpy(item["audio"])
        if array is not None:
            return array

    # 3. Direct tensor/ndarray
    array = _to_numpy(item)
    if array is not None:
        return array

    # 4. Tuple/list — scan each element
    if isinstance(item, (tuple, list)):
        for value in item:
            array = _to_numpy(value)
            if array is not None:
                return array

    raise ModelError(
        f"TTS returned unexpected output type: {type(item).__name__} "
        f"(attrs sample: {[a for a in dir(item) if not a.startswith('_')][:6]})"
    )


def _split_text(text, max_chars=420):
    cleaned = re.sub(r"[ \t]+", " ", text.strip())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if not cleaned:
        raise ValidationError("Text is empty after cleaning")
    sentences = re.split(r"(?<=[.!?।])\s+|\n{2,}", cleaned)
    chunks, current = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
        else:
            for start in range(0, len(sentence), max_chars):
                part = sentence[start:start + max_chars].strip()
                if part:
                    chunks.append(part)
            current = ""
    if current:
        chunks.append(current)
    return chunks or [cleaned[:max_chars]]


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


class KokoroBackend:
    def __init__(self, settings, spec):
        self.settings = settings
        self.spec = spec
        self._pipelines = {}
        self._KPipeline = None
        self._torch = None

    def load(self):
        try:
            import torch
            from kokoro import KPipeline
        except ImportError as exc:
            raise DependencyMissing(
                f"kokoro not installed ({exc}). Run: pip install -r requirements.txt"
            ) from exc
        torch.set_num_threads(self.settings.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self._KPipeline = KPipeline
        self._torch = torch

    def _get_pipeline(self, language):
        if language not in KOKORO_LANG:
            raise ValidationError(f"Unsupported Kokoro language: {language}")
        if language not in self._pipelines:
            self._pipelines[language] = self._KPipeline(
                lang_code=KOKORO_LANG[language],
                repo_id=self.spec.repo_id,
                device="cpu",
            )
        return self._pipelines[language]

    def unload(self):
        self._pipelines.clear()
        self._KPipeline = None
        self._torch = None

    def synthesize(self, text, output_path, language="en", voice=None, speed=1.0):
        pipeline = self._get_pipeline(language)
        voice = voice or DEFAULT_VOICE[language]
        if voice not in KOKORO_VOICES[language]:
            raise ValidationError(f"Unsupported voice '{voice}' for language '{language}'")
        if not 0.5 <= speed <= 2.0:
            raise ValidationError("Speed must be between 0.5 and 2.0")
        started = time.monotonic()
        audio_parts = []
        chunks = _split_text(text)
        for idx, chunk in enumerate(chunks):
            try:
                for item in pipeline(chunk, voice=voice, speed=speed, split_pattern=r"\n+"):
                    audio_parts.append(_coerce_audio(item))
            except ModelError:
                raise
            except Exception as exc:
                raise ModelError(f"Kokoro failed on chunk {idx + 1}: {exc}") from exc
        if not audio_parts:
            raise ModelError("Kokoro produced no audio")
        waveform = np.concatenate(audio_parts)
        _write_wav(output_path, waveform, 24000)
        duration = _verify_wav(output_path)
        elapsed = time.monotonic() - started
        return {"path": output_path, "duration": duration, "language": language,
                "voice": voice, "elapsed": elapsed, "sample_rate": 24000}


class MMSTTSBackend:
    def __init__(self, settings, spec):
        self.settings = settings
        self.spec = spec
        self._model = None
        self._tokenizer = None
        self._torch = None

    def load(self):
        try:
            import torch
            from transformers import AutoTokenizer, VitsModel
        except ImportError as exc:
            raise DependencyMissing(
                f"transformers/torch missing ({exc}). Run: pip install transformers"
            ) from exc
        torch.set_num_threads(self.settings.cpu_threads)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.repo_id)
            self._model = VitsModel.from_pretrained(self.spec.repo_id).to("cpu").eval()
            self._torch = torch
        except Exception as exc:
            raise ModelError(f"MMS-TTS load failed: {exc}") from exc

    def unload(self):
        self._model = None
        self._tokenizer = None
        self._torch = None

    def synthesize(self, text, output_path, language="hi", voice=None, speed=1.0):
        if self._model is None or self._tokenizer is None:
            raise ModelError("MMS-TTS not loaded")
        cleaned = text.strip()
        if not cleaned:
            raise ValidationError("Text is empty")
        started = time.monotonic()
        try:
            inputs = self._tokenizer(cleaned, return_tensors="pt")
            with self._torch.inference_mode():
                waveform = self._model(**inputs).waveform[0].cpu().numpy().astype(np.float32)
        except Exception as exc:
            raise ModelError(f"MMS-TTS inference failed: {exc}") from exc
        rate = int(self._model.config.sampling_rate)
        _write_wav(output_path, waveform, rate)
        duration = _verify_wav(output_path)
        return {"path": output_path, "duration": duration, "language": "hi",
                "voice": "default", "elapsed": time.monotonic() - started,
                "sample_rate": rate}


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
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.spec.repo_id, trust_remote_code=False)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.spec.repo_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
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
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
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
        return {"text": text, "elapsed": time.monotonic() - started,
                "prompt_tokens": input_len,
                "completion_tokens": int(generated.shape[-1])}


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
                self.spec.repo_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
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
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[prompt], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt")
            with self._torch.inference_mode():
                generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
            input_ids = inputs["input_ids"]
            trimmed = []
            for i in range(generated.shape[0]):
                prompt_len = int(input_ids[i].shape[-1])
                trimmed.append(generated[i][prompt_len:])
            answer = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        except Exception as exc:
            raise ModelError(f"Vision inference failed: {exc}") from exc
        if not answer:
            raise ModelError("Vision model returned empty answer")
        return {"text": answer, "elapsed": time.monotonic() - started}


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
            raise DependencyMissing(
                "OpenVoice deps missing. Clone myshell-ai/OpenVoice, install melo-tts, "
                "then set OPENVOICE_REPO_DIR and OPENVOICE_CHECKPOINT_DIR."
            ) from exc
        if not self.settings.openvoice_ckpt:
            raise ModelError(
                "OPENVOICE_CHECKPOINT_DIR is not set; voice cloning is disabled.")
        ckpt = Path(self.settings.openvoice_ckpt).expanduser().resolve()
        conv_dir = ckpt / "converter"
        if not (conv_dir / "config.json").exists():
            raise ModelError(f"OpenVoice converter not found at {conv_dir}")
        torch.set_num_threads(self.settings.cpu_threads)
        try:
            self._converter = ToneColorConverter(
                str(conv_dir / "config.json"), device="cpu")
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
            target_se, _ = self._se.get_se(
                str(reference_wav), self._converter, vad=True)
            speakers = self._base_tts.hps.data.spk2id
            spk = speakers.get("EN-Default") or next(iter(speakers.values()))
            self._base_tts.tts_to_file(cleaned, spk, str(temp_base), speed=1.0)
            source_se, _ = self._se.get_se(
                str(temp_base), self._converter, vad=True)
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
    "kokoro": KokoroBackend,
    "mms-tts-hin": MMSTTSBackend,
    "qwen-chat-0.5b": ChatBackend,
    "qwen-chat-1.5b": ChatBackend,
    "granite-2b": ChatBackend,
    "qwen-vl-3b": VisionBackend,
    "openvoice-v2": OpenVoiceBackend,
}
