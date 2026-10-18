from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from models import (
    Category, ModelError, ModelManager, ModelStatus, by_category, get_spec,
)
from utils import probe_duration, system_snapshot, validate_image

logger = logging.getLogger("test_models")


def _write_report(results: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_{int(time.time())}.json"
    path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": system_snapshot(),
        "results": results,
    }, indent=2))
    return path


def _test_tts(manager: ModelManager, key: str, out_dir: Path) -> dict:
    spec = get_spec(key)
    started = time.time()
    try:
        backend = manager.get(key)
        language = "hi" if "hi" in spec.languages else (spec.languages[0] if spec.languages else "en")
        text = "नमस्ते दुनिया।" if language == "hi" else "Hello world. This is a test."
        out_path = out_dir / f"{key}_sample.wav"
        result = backend.synthesize(text, out_path, language=language)
        if not out_path.exists():
            raise ModelError("Output WAV missing")
        duration = probe_duration(out_path)
        if duration < 0.5:
            raise ModelError(f"Output too short: {duration:.2f}s")
        return {
            "model": key, "category": "tts",
            "status": ModelStatus.PASS.value,
            "wall_seconds": round(time.time() - started, 2),
            "audio_seconds": round(duration, 2),
            "artifact": str(out_path),
            "language": language,
        }
    except Exception as exc:
        logger.exception("TTS test failed for %s", key)
        return {
            "model": key, "category": "tts",
            "status": ModelStatus.FAIL.value,
            "error": str(exc),
            "trace": traceback.format_exc()[-800:],
            "wall_seconds": round(time.time() - started, 2),
        }
    finally:
        manager.unload()


def _test_chat(manager: ModelManager, key: str, out_dir: Path) -> dict:
    started = time.time()
    try:
        backend = manager.get(key)
        result = backend.generate(
            [{"role": "user", "content": "Reply with one short sentence about CPUs."}],
            max_new_tokens=64)
        if not result["text"].strip():
            raise ModelError("Empty response")
        return {
            "model": key, "category": "chat",
            "status": ModelStatus.PASS.value,
            "wall_seconds": round(time.time() - started, 2),
            "text": result["text"][:300],
        }
    except Exception as exc:
        logger.exception("Chat test failed for %s", key)
        return {
            "model": key, "category": "chat",
            "status": ModelStatus.FAIL.value,
            "error": str(exc),
            "trace": traceback.format_exc()[-800:],
            "wall_seconds": round(time.time() - started, 2),
        }
    finally:
        manager.unload()


def _test_vision(manager: ModelManager, key: str, out_dir: Path) -> dict:
    from PIL import Image, ImageDraw
    started = time.time()
    try:
        backend = manager.get(key)
        img = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([64, 64, 192, 192], fill="blue")
        img_path = out_dir / "sample_vision.jpg"
        img.save(img_path, "JPEG", quality=90)
        validate_image(img_path, 10)
        result = backend.describe(img_path, "What is in this image?")
        if not result["text"].strip():
            raise ModelError("Empty response")
        return {
            "model": key, "category": "vision",
            "status": ModelStatus.PASS.value,
            "wall_seconds": round(time.time() - started, 2),
            "text": result["text"][:300],
        }
    except Exception as exc:
        logger.exception("Vision test failed for %s", key)
        return {
            "model": key, "category": "vision",
            "status": ModelStatus.FAIL.value,
            "error": str(exc),
            "trace": traceback.format_exc()[-800:],
            "wall_seconds": round(time.time() - started, 2),
        }
    finally:
        manager.unload()


def _test_voice_cloning(manager: ModelManager, key: str, out_dir: Path) -> dict:
    return {
        "model": key, "category": "voice_cloning",
        "status": ModelStatus.SKIPPED.value,
        "reason": "Requires manual reference audio and OpenVoice checkpoint setup. "
                  "Not testable in automation.",
    }


TESTS = {
    "tts": _test_tts,
    "chat": _test_chat,
    "vision": _test_vision,
    "voice_cloning": _test_voice_cloning,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", help="tts | chat | vision | voice_cloning")
    parser.add_argument("--model", help="specific model key")
    parser.add_argument("--all-small", action="store_true")
    parser.add_argument("--audit", action="store_true",
                        help="print registry metadata only, download nothing")
    parser.add_argument("--out", default="runtime/reports")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.audit:
        from models import REGISTRY
        for spec in REGISTRY.values():
            print(json.dumps(spec.public(), indent=2))
        return 0

    settings = Settings.load()
    snap = system_snapshot(settings.data_dir)
    logger.info("System: %s", json.dumps(snap, indent=2))
    if snap["disk_free_gb"] < 5:
        logger.error("Not enough disk space: %.1f GB free", snap["disk_free_gb"])
        return 2
    if snap["ram_total_gb"] < 2:
        logger.error("Not enough RAM: %.1f GB total", snap["ram_total_gb"])
        return 2

    manager = ModelManager(settings)
    results: list[dict] = []
    plan: list[tuple[str, str]] = []

    if args.model:
        spec = get_spec(args.model)
        plan.append((spec.category.value, spec.key))
    elif args.all_small:
        plan = [("tts", "mms-tts-hin"), ("tts", "kokoro"),
                ("chat", "qwen-chat-0.5b")]
    elif args.category:
        for spec in by_category(args.category):
            if spec.status in {ModelStatus.NEEDS_VERIFICATION,
                                ModelStatus.TOO_LARGE,
                                ModelStatus.TOO_SLOW,
                                ModelStatus.SKIPPED}:
                results.append({
                    "model": spec.key, "category": spec.category.value,
                    "status": spec.status.value, "reason": spec.notes,
                })
                continue
            plan.append((spec.category.value, spec.key))
    else:
        parser.print_help()
        return 1

    for cat, key in plan:
        logger.info("=== Testing %s (%s) ===", key, cat)
        fn = TESTS.get(cat)
        if fn is None:
            results.append({"model": key, "category": cat,
                            "status": ModelStatus.SKIPPED.value,
                            "reason": "no test function"})
            continue
        results.append(fn(manager, key, out_dir))

    report = _write_report(results, out_dir)
    logger.info("Report written: %s", report)
    failed = [r for r in results if r["status"] == ModelStatus.FAIL.value]
    print(json.dumps({"report": str(report), "total": len(results),
                      "failed": len(failed)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
