from __future__ import annotations

import logging
from pathlib import Path

from telegram import (
    CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup,
    Message, Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes,
    MessageHandler, filters,
)

from config import ConfigError, Settings, require_token
from models import (
    Category, ModelManager, ModelStatus, by_category, get_spec,
)
from utils import (
    Job, TaskQueue, deliver_voice, resize_image, system_snapshot,
    to_wav_16k, unique_name, validate_audio, validate_image, validate_text,
)

logger = logging.getLogger(__name__)

CATEGORY_LABELS = {
    Category.TTS: "🗣️ Text to Speech",
    Category.CHAT: "💬 Chat AI",
    Category.VISION: "🧠 Vision AI",
    Category.VOICE_CLONING: "🎭 Voice Cloning",
}

STATUS_ICON = {
    ModelStatus.PASS: "✅",
    ModelStatus.UNTESTED: "⚪",
    ModelStatus.FAIL: "❌",
    ModelStatus.NEEDS_VERIFICATION: "⚠️",
    ModelStatus.TOO_LARGE: "🚫",
    ModelStatus.TOO_SLOW: "🐌",
    ModelStatus.SKIPPED: "·",
}


class BotState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models = ModelManager(settings)
        self.queue = TaskQueue(settings.queue_size)
        self.history: dict[int, list[dict]] = {}
        self.selection: dict[int, dict] = {}
        self.bot = None


def _menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💬 Chat AI", callback_data="cat:chat"),
         InlineKeyboardButton("🧠 Vision AI", callback_data="cat:vision")],
        [InlineKeyboardButton("🗣️ Text to Speech", callback_data="cat:tts"),
         InlineKeyboardButton("🎭 Voice Cloning", callback_data="cat:voice_cloning")],
        [InlineKeyboardButton("📊 System Status", callback_data="cmd:status"),
         InlineKeyboardButton("⚙️ Settings", callback_data="cmd:settings")],
        [InlineKeyboardButton("❌ Cancel Task", callback_data="cmd:cancel")],
    ])


def _category_menu(cat: Category) -> InlineKeyboardMarkup:
    rows = []
    for spec in by_category(cat):
        icon = STATUS_ICON.get(spec.status, "·")
        rows.append([InlineKeyboardButton(
            f"{icon} {spec.display_name}",
            callback_data=f"sel:{cat.value}:{spec.key}",
        )])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="cmd:menu")])
    return InlineKeyboardMarkup(rows)


def _back(target: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data=target)]])


def owner_only(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        state: BotState = context.application.bot_data["state"]
        if state.settings.owner_chat_id is not None:
            user_id = update.effective_user.id if update.effective_user else None
            if user_id != state.settings.owner_chat_id:
                if update.message:
                    await update.message.reply_text("🔒 This is a private bot. You are not authorized.")
                elif update.callback_query:
                    await update.callback_query.answer("🔒 Unauthorized", show_alert=True)
                return
        return await func(update, context)
    return wrapper


@owner_only
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*CPU AI Bot*\n\nRuns entirely on CPU. No GPU, no paid APIs.\n\n"
        "Pick a category below. Use /help for commands and /status for system info.",
        reply_markup=_menu(), parse_mode=ParseMode.MARKDOWN)


@owner_only
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*Commands*\n/start — main menu\n/help — this message\n/status — CPU, RAM and active model\n"
        "/models — model menu\n/cancel — cancel the current task\n/reset — clear chat history and session\n"
        "/about — about the project\n\n*How it works*\n1. Pick a category.\n2. Pick a model.\n3. Send text / photo / voice.\n\n"
        "*TTS format:* `lang:voice text`. Plain text uses defaults.\n"
        "*Voice cloning:* send a 2–30s voice sample, then /confirm_voice, then the text.",
        parse_mode=ParseMode.MARKDOWN)


@owner_only
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: BotState = context.application.bot_data["state"]
    snap = system_snapshot(state.settings.data_dir)
    await update.message.reply_text(
        f"*System*\nPlatform: `{snap['platform']}`\nPython: `{snap['python']}`\nCPUs: `{snap['cpu_count']}`\n"
        f"RAM free: `{snap['ram_free_gb']} / {snap['ram_total_gb']} GB`\nDisk free: `{snap['disk_free_gb']} GB`\n"
        f"PyTorch: `{snap['torch']}`\nCUDA: `{snap['cuda']}`\nffmpeg: `{snap['ffmpeg']}`\n\n"
        f"*Bot*\nActive model: `{state.models.active_key or 'none'}`\nQueue: `{state.queue.size}` jobs\n"
        f"Max upload: `{state.settings.max_upload_mb} MB`",
        parse_mode=ParseMode.MARKDOWN)


@owner_only
async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("*Pick a category*", reply_markup=_menu(), parse_mode=ParseMode.MARKDOWN)


@owner_only
async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: BotState = context.application.bot_data["state"]
    sel = state.selection.get(update.effective_user.id)
    if sel and sel.get("cancel_event"):
        sel["cancel_event"].set()
        await update.message.reply_text("Cancelling…")
    else:
        await update.message.reply_text("No active task to cancel.")


@owner_only
async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: BotState = context.application.bot_data["state"]
    uid = update.effective_user.id
    state.history.pop(uid, None)
    state.selection.pop(uid, None)
    await update.message.reply_text("Session cleared.")


@owner_only
async def cmd_about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*CPU AI Bot*\n\nA Telegram bot that runs AI models on CPU only. No GPU and no paid APIs.\n\n"
        "Models are downloaded on first use and unloaded after each task — only one heavy model is in memory at a time.\n\n"
        "The bot is hosted on GitHub Actions: it is online only while the workflow is running.",
        parse_mode=ParseMode.MARKDOWN)


@owner_only
async def cmd_confirm_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: BotState = context.application.bot_data["state"]
    sel = state.selection.get(update.effective_user.id)
    if not sel or not sel.get("reference_path"):
        await update.message.reply_text("Send a voice sample first (2–30s).")
        return
    sel["confirmed"] = True
    await update.message.reply_text("Permission noted. Now send the text you want spoken in the cloned voice.")


@owner_only
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query: CallbackQuery = update.callback_query
    await query.answer()
    data = query.data or ""
    state: BotState = context.application.bot_data["state"]
    uid = query.from_user.id

    if data == "cmd:menu":
        await query.message.edit_text("*Main menu*", reply_markup=_menu(), parse_mode=ParseMode.MARKDOWN)
        return
    if data == "cmd:status":
        snap = system_snapshot(state.settings.data_dir)
        await query.message.edit_text(
            f"*Status*\nRAM free: `{snap['ram_free_gb']} GB`\nDisk free: `{snap['disk_free_gb']} GB`\n"
            f"Active model: `{state.models.active_key or 'none'}`\nQueue: `{state.queue.size}`",
            parse_mode=ParseMode.MARKDOWN, reply_markup=_back("cmd:menu"))
        return
    if data == "cmd:settings":
        await query.message.edit_text(
            f"*Settings*\nMax upload: `{state.settings.max_upload_mb} MB`\nCPU threads: `{state.settings.cpu_threads}`\n"
            f"Max input: `{state.settings.max_input_chars}` chars\nQueue size: `{state.settings.queue_size}`",
            parse_mode=ParseMode.MARKDOWN, reply_markup=_back("cmd:menu"))
        return
    if data == "cmd:cancel":
        sel = state.selection.get(uid)
        if sel and sel.get("cancel_event"):
            sel["cancel_event"].set()
            await query.message.edit_text("Cancelling…", reply_markup=_back("cmd:menu"))
        else:
            await query.message.edit_text("Nothing to cancel.", reply_markup=_back("cmd:menu"))
        return
    if data.startswith("cat:"):
        cat = Category(data.split(":", 1)[1])
        await query.message.edit_text(
            f"*{CATEGORY_LABELS[cat]}*\nSelect a model:",
            reply_markup=_category_menu(cat), parse_mode=ParseMode.MARKDOWN)
        return
    if data.startswith("sel:"):
        _, cat_value, key = data.split(":", 2)
        spec = get_spec(key)
        if spec.status in {ModelStatus.NEEDS_VERIFICATION, ModelStatus.TOO_LARGE, ModelStatus.TOO_SLOW}:
            await query.message.edit_text(
                f"⚠️ *{spec.display_name}* is not available.\n\nStatus: `{spec.status.value}`\nNotes: {spec.notes or 'no additional notes'}",
                parse_mode=ParseMode.MARKDOWN, reply_markup=_back(f"cat:{cat_value}"))
            return
        state.selection[uid] = {"category": cat_value, "model_key": key, "cancel_event": None}
        await query.message.edit_text(
            f"*{spec.display_name}* selected.\n\nLicense: `{spec.license}`\nLanguages: `{', '.join(spec.languages) or 'unknown'}`\n"
            f"Memory: {spec.memory_notes or 'n/a'}\nDownload: ~{spec.download_size_mb or '?'} MB\n\n{_prompt_for(cat_value)}",
            parse_mode=ParseMode.MARKDOWN, reply_markup=_back("cmd:menu"))
        return


def _prompt_for(cat_value: str) -> str:
    return {
        "tts": "Send me text. Format `lang:voice text` or just plain text.",
        "chat": "Send any message. /reset clears history.",
        "vision": "Send a photo with an optional question in the caption.",
        "voice_cloning": "Send a 2–30s voice sample, then /confirm_voice, then the text.",
    }.get(cat_value, "Send your input.")


def _parse_tts_input(text: str, spec) -> tuple[str, str | None, str]:
    default_lang = spec.languages[0] if spec.languages else "en"
    parts = text.split(" ", 1)
    if len(parts) == 2 and ":" in parts[0]:
        lang, _, voice = parts[0].partition(":")
        if lang in spec.languages:
            return lang, voice or None, parts[1]
    return default_lang, None, text


@owner_only
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: BotState = context.application.bot_data["state"]
    uid = update.effective_user.id
    chat_id = update.effective_chat.id
    sel = state.selection.get(uid)
    if not sel:
        await update.message.reply_text("Pick a model from /models first.")
        return
    cat = sel["category"]
    key = sel["model_key"]

    if cat == "tts":
        try:
            text = validate_text(update.message.text or "", state.settings.max_input_chars)
        except Exception as exc:
            await update.message.reply_text(f"⚠️ {exc}")
            return
        language, voice, body = _parse_tts_input(text, get_spec(key))
        job = Job(user_id=uid, chat_id=chat_id, kind="tts",
                  payload={"model_key": key, "text": body, "language": language, "voice": voice},
                  timeout=state.settings.job_timeout)
        await _submit(update, state, job, uid)

    elif cat == "chat":
        try:
            text = validate_text(update.message.text or "", state.settings.max_input_chars)
        except Exception as exc:
            await update.message.reply_text(f"⚠️ {exc}")
            return
        job = Job(user_id=uid, chat_id=chat_id, kind="chat",
                  payload={"model_key": key, "text": text},
                  timeout=state.settings.job_timeout)
        await _submit(update, state, job, uid)

    elif cat == "vision":
        if not update.message.photo:
            await update.message.reply_text("Send a photo for Vision AI.")
            return
        photo = update.message.photo[-1]
        f = await photo.get_file()
        tmp = state.settings.data_dir / "in"
        tmp.mkdir(parents=True, exist_ok=True)
        raw = tmp / unique_name("img", ".jpg")
        await f.download_to_drive(str(raw))
        try:
            validate_image(raw, state.settings.max_photo_mb)
            resized = tmp / unique_name("rs", ".jpg")
            resize_image(raw, resized)
            raw.unlink(missing_ok=True)
        except Exception as exc:
            raw.unlink(missing_ok=True)
            await update.message.reply_text(f"⚠️ {exc}")
            return
        question = (update.message.caption or "").strip() or "Describe this image."
        job = Job(user_id=uid, chat_id=chat_id, kind="vision",
                  payload={"model_key": key, "image_path": str(resized), "question": question},
                  timeout=state.settings.job_timeout)
        await _submit(update, state, job, uid)

    elif cat == "voice_cloning":
        if update.message.voice or update.message.audio:
            media = update.message.voice or update.message.audio
            f = await media.get_file()
            tmp = state.settings.data_dir / "in"
            tmp.mkdir(parents=True, exist_ok=True)
            raw = tmp / unique_name("ref", ".ogg")
            await f.download_to_drive(str(raw))
            try:
                duration = validate_audio(raw, state.settings.max_reference_audio_mb, state.settings.max_reference_audio_seconds)
            except Exception as exc:
                raw.unlink(missing_ok=True)
                await update.message.reply_text(f"⚠️ {exc}")
                return
            wav = raw.with_suffix(".wav")
            try:
                to_wav_16k(raw, wav)
            finally:
                raw.unlink(missing_ok=True)
            sel["reference_path"] = str(wav)
            sel["confirmed"] = False
            await update.message.reply_text(f"✅ Reference accepted ({duration:.1f}s). Run /confirm_voice to confirm permission, then send the text.")
            return
        if update.message.text:
            if not sel.get("reference_path"):
                await update.message.reply_text("Send a voice sample first (2–30s).")
                return
            if not sel.get("confirmed"):
                await update.message.reply_text("Please run /confirm_voice first to confirm you have permission.")
                return
            try:
                text = validate_text(update.message.text, state.settings.max_input_chars)
            except Exception as exc:
                await update.message.reply_text(f"⚠️ {exc}")
                return
            job = Job(user_id=uid, chat_id=chat_id, kind="voice_clone",
                      payload={"model_key": key, "text": text, "reference_path": sel["reference_path"]},
                      timeout=state.settings.job_timeout)
            await _submit(update, state, job, uid)


async def _submit(update: Update, state: BotState, job: Job, uid: int):
    sel = state.selection.setdefault(uid, {})
    sel["cancel_event"] = job.cancel_event
    ok = await state.queue.submit(job)
    if not ok:
        await update.message.reply_text("⚠️ Queue is full. Try again in a moment.")
        return
    await update.message.reply_text(f"⏳ Queued ({state.queue.size} in queue). Please wait…")


async def _do_tts(job: Job, state: BotState):
    bot = state.bot
    spec = get_spec(job.payload["model_key"])
    status = await bot.send_message(job.chat_id, f"🎙️ Loading {spec.display_name}…")
    out_path = None
    try:
        backend = state.models.get(spec.key)
        await status.edit_text("🎙️ Generating audio…")
        out_dir = state.settings.data_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / unique_name("tts", ".wav")
        kwargs = {
            "language": job.payload.get("language", "hi"),
            "voice": job.payload.get("voice"),
        }
        if spec.key == "indic-parler-tts":
            kwargs["description"] = job.payload.get("description") or "A clear Hindi voice, moderate pace, no background noise."
        
        result = backend.synthesize(job.payload["text"], out_path, **kwargs)
        await status.edit_text("📤 Sending audio…")
        await deliver_voice(bot, job.chat_id, out_path, spec.display_name, state.settings, status)
        await status.edit_text(f"✅ Done. {result['duration']:.1f}s audio in {result['elapsed']:.1f}s.")
    except Exception as exc:
        await status.edit_text(f"❌ TTS failed: {exc}")
    finally:
        if out_path and out_path.exists():
            out_path.unlink(missing_ok=True)


async def _do_chat(job: Job, state: BotState):
    bot = state.bot
    spec = get_spec(job.payload["model_key"])
    status = await bot.send_message(job.chat_id, f"💬 Loading {spec.display_name}…")
    try:
        backend = state.models.get(spec.key)
        await status.edit_text("💭 Thinking…")
        history = state.history.setdefault(job.user_id, [])
        history.append({"role": "user", "content": job.payload["text"]})
        result = backend.generate(history[-6:], max_new_tokens=job.payload.get("max_new_tokens", 200))
        history.append({"role": "assistant", "content": result["text"]})
        state.history[job.user_id] = history[-12:]
        await status.edit_text(
            f"{result['text']}\n\n_({result['completion_tokens']} tokens in {result['elapsed']:.1f}s)_",
            parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:
        await status.edit_text(f"❌ Chat failed: {exc}")


async def _do_vision(job: Job, state: BotState):
    bot = state.bot
    spec = get_spec(job.payload["model_key"])
    status = await bot.send_message(job.chat_id, f"🧠 Loading {spec.display_name}…")
    try:
        backend = state.models.get(spec.key)
        await status.edit_text("👁️ Analyzing image…")
        result = backend.describe(job.payload["image_path"], job.payload["question"])
        await status.edit_text(f"{result['text']}\n\n_({result['elapsed']:.1f}s)_", parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:
        await status.edit_text(f"❌ Vision failed: {exc}")


async def _do_voice_clone(job: Job, state: BotState):
    bot = state.bot
    spec = get_spec(job.payload["model_key"])
    status = await bot.send_message(job.chat_id, f"🎭 Loading {spec.display_name}…")
    out_path = None
    try:
        backend = state.models.get(spec.key)
        await status.edit_text("🎭 Generating cloned voice…")
        out_dir = state.settings.data_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / unique_name("clone", ".wav")
        backend.clone(job.payload["text"], job.payload["reference_path"], out_path)
        await status.edit_text("📤 Sending cloned audio…")
        await deliver_voice(bot, job.chat_id, out_path, "Cloned voice", state.settings, status)
        await status.edit_text("✅ Done.")
    except Exception as exc:
        await status.edit_text(f"❌ Voice cloning failed: {exc}")
    finally:
        if out_path and out_path.exists():
            out_path.unlink(missing_ok=True)


async def _on_startup(app: Application):
    state: BotState = app.bot_data["state"]
    state.bot = app.bot
    state.queue.register("tts", lambda job: _do_tts(job, state))
    state.queue.register("chat", lambda job: _do_chat(job, state))
    state.queue.register("vision", lambda job: _do_vision(job, state))
    state.queue.register("voice_clone", lambda job: _do_voice_clone(job, state))
    await state.queue.start()
    logger.info("Bot started: cpu_threads=%s queue=%s", state.settings.cpu_threads, state.settings.queue_size)
    if state.settings.owner_chat_id:
        try:
            await app.bot.send_message(
                chat_id=state.settings.owner_chat_id,
                text="✅ *Bot is online!*\n\nServer ready hai. `/start` bhejo shuru karne ke liye.",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as exc:
            logger.warning("Could not send startup message to owner: %s", exc)


async def _on_shutdown(app: Application):
    state: BotState = app.bot_data["state"]
    await state.queue.stop()
    state.models.unload()


def build_app(settings: Settings) -> Application:
    token = require_token(settings)
    app = Application.builder().token(token).concurrent_updates(True).build()
    app.bot_data["state"] = BotState(settings)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("confirm_voice", cmd_confirm_voice))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO, on_message))
    app.post_init = _on_startup
    app.post_shutdown = _on_shutdown
    return app


def main():
    settings = Settings.load()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "telegram.ext.Application", "telegram.Bot", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        app = build_app(settings)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        raise SystemExit(2)
    logger.info("Polling for updates…")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True, close_loop=False)


if __name__ == "__main__":
    main()
