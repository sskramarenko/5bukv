# -*- coding: utf-8 -*-
"""
Телеграм-бот — решатель игры «Пять букв» (русский Wordle).

Что умеет:
  • сразу советует, чем бить первым («норка»)
  • после каждого отклика пересчитывает живых кандидатов и лучшие ходы
  • /solve — сброс между словами
  • работает только в разрешённых чатах (ALLOWED_CHATS)
  • платный режим: три слова бесплатно, дальше код разблокировки

Запуск: BOT_TOKEN=... python bot.py
"""

import asyncio
import json
import logging
import math
import os
import random
import re
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramNetworkError
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, ErrorEvent, Message

VERSION = "2026-11-17 · 14:40"

# ---------------------------------------------------------------- настройки

MAX_ATTEMPTS = 6             # попыток в игре
OPENER = "норка"             # чем бить первым ходом
TOP_MOVES = 3                # сколько лучших ходов показывать
SHOW_CANDIDATES = 5          # сколько вероятных кандидатов в строке
LIST_LIMIT = 30              # /list
JUNK_ZIPF = 2.0              # ниже этого слово считаем мусором словаря
JUNK_TOLERANCE = 1.15        # на сколько хуже остаток допустим, чтобы взять чистое слово
SCORE_SET_CAP = 900          # если живых больше — считаем по самым вероятным
SLOW_THRESHOLD = 400         # выше этого числа живых сначала пишем «считаю…»

# Бот отвечает только в этих чатах. Пусто — работает везде (режим продажи).
ALLOWED_CHATS = {int(x) for x in os.getenv("ALLOWED_CHATS", "").replace(" ", "").split(",")
                 if x.lstrip("-").isdigit()}
ADMIN_IDS = {int(x) for x in os.getenv("ADMIN_IDS", "").replace(" ", "").split(",")
             if x.isdigit()}

# Платный режим
PAYWALL = os.getenv("PAYWALL", "0") == "1"
FREE_WORDS = int(os.getenv("FREE_WORDS", "3") or 3)
PAY_TEXT = os.getenv("PAY_TEXT", "").strip()
SEED_CODES = [c.strip().upper() for c in os.getenv("UNLOCK_CODES", "").split(",") if c.strip()]

# Хостинг стирает диск при пересборке — копию памяти держим в телеграме.
STORAGE_CHAT = int(os.getenv("STORAGE_CHAT", "0") or 0) or (min(ADMIN_IDS) if ADMIN_IDS else 0)
BACKUP_EVERY_SEC = 120

MSK = timezone(timedelta(hours=3))
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "solver_data.json"
DICT_FILE = BASE_DIR / "dict_bot.txt"
MOVE_LOG = BASE_DIR / "moves.log"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("solver")


def now_msk() -> datetime:
    return datetime.now(MSK)

# ---------------------------------------------------------------- словарь

ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
LETTER_CODE = {ch: i for i, ch in enumerate(ALPHABET)}
POW3 = np.array([81, 27, 9, 3, 1], dtype=np.int16)


def normalize(word: str) -> str:
    return word.strip().lower().replace("ё", "е")


def load_dictionary() -> tuple[list[str], np.ndarray]:
    words, zipfs = [], []
    if not DICT_FILE.exists():
        log.error("Нет файла словаря %s — бот не сможет работать", DICT_FILE.name)
        return [], np.zeros(0)
    for line in DICT_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        word = normalize(parts[0])
        if len(word) != 5 or any(ch not in LETTER_CODE for ch in word):
            continue
        try:
            z = float(parts[1]) if len(parts) > 1 else 0.0
        except ValueError:
            z = 0.0
        words.append(word)
        zipfs.append(z)
    return words, np.array(zipfs, dtype=np.float64)


WORDS, ZIPF = load_dictionary()
N_WORDS = len(WORDS)
INDEX = {w: i for i, w in enumerate(WORDS)}

# коды букв и количество каждой буквы в слове — для векторного отклика
LET = np.zeros((N_WORDS, 5), dtype=np.uint8)
CNT = np.zeros((N_WORDS, len(ALPHABET)), dtype=np.uint8)
for i, w in enumerate(WORDS):
    for k, ch in enumerate(w):
        c = LETTER_CODE[ch]
        LET[i, k] = c
        CNT[i, c] += 1


def weight_of(zipf: np.ndarray | float):
    """Вес = вероятность, что слово загадано. Откалибровано на реальных ответах."""
    return 3.4 / (1 + np.exp(-4 * (np.asarray(zipf, dtype=np.float64) - 2.7)))


WEIGHT = weight_of(ZIPF)
CLEAN = ZIPF >= JUNK_ZIPF          # приличное обиходное слово
USABLE = ZIPF > 0                  # есть в частотном корпусе; нулевые — чаще обрезки

# ---------------------------------------------------------------- отклик

def feedback(guess: str, answer: str) -> str:
    """Эталонная реализация: строка из пяти цифр 0/1/2."""
    res = ["0"] * 5
    used = [False] * 5
    for i in range(5):
        if guess[i] == answer[i]:
            res[i] = "2"
            used[i] = True
    for i in range(5):
        if res[i] == "2":
            continue
        for j in range(5):
            if not used[j] and guess[i] == answer[j]:
                res[i] = "1"
                used[j] = True
                break
    return "".join(res)


def code_of(pattern: str) -> int:
    return sum(int(d) * p for d, p in zip(pattern, (81, 27, 9, 3, 1)))


def pattern_of(code: int) -> str:
    out = []
    for p in (81, 27, 9, 3, 1):
        out.append(str(code // p))
        code %= p
    return "".join(out)


def fb_matrix(guess_idx: np.ndarray, answer_idx: np.ndarray, chunk: int = 96) -> np.ndarray:
    """Матрица кодов отклика (len(guess) × len(answer)), значения 0..242.

    Считаем на лету: полная матрица «весь словарь × весь словарь» не нужна,
    потому что первый ход
    у нас фиксирован, а дальше живых кандидатов уже мало.
    """
    gi = np.asarray(guess_idx)
    ai = np.asarray(answer_idx)
    m, n = len(gi), len(ai)
    out = np.empty((m, n), dtype=np.uint8)
    if m == 0 or n == 0:
        return out
    A = LET[ai]                       # n,5
    CA = CNT[ai]                      # n,32

    for s in range(0, m, chunk):
        g = LET[gi[s:s + chunk]]                              # b,5
        green = g[:, None, :] == A[None, :, :]                # b,n,5
        digits = np.where(green, 2, 0).astype(np.int16)        # b,n,5
        for c in np.unique(g):
            gc = g == c                                        # b,5
            not_green = gc[:, None, :] & ~green                # b,n,5
            if not not_green.any():
                continue
            green_c = (green & gc[:, None, :]).sum(axis=2)     # b,n
            avail = CA[None, :, c].astype(np.int16) - green_c  # b,n
            rank = np.cumsum(not_green, axis=2)                # b,n,5 (1-based)
            digits[not_green & (rank <= avail[:, :, None])] = 1
        out[s:s + chunk] = (digits * POW3).sum(axis=2).astype(np.uint8)
    return out

# ---------------------------------------------------------------- решатель

@dataclass
class Game:
    moves: list[tuple[str, str]] = field(default_factory=list)   # (слово, отклик)
    alive: np.ndarray = field(default_factory=lambda: np.arange(N_WORDS))
    billed: bool = False          # слово уже списано из квоты


def recompute(game: Game) -> None:
    """Пересчёт живых кандидатов с нуля по всем ходам — так не копятся ошибки."""
    alive = np.arange(N_WORDS)
    for word, resp in game.moves:
        if not alive.size:
            break
        gi = INDEX.get(word)
        if gi is None:
            # слова нет в словаре — считаем отклик напрямую
            keep = [a for a in alive if feedback(word, WORDS[a]) == resp]
            alive = np.array(keep, dtype=np.int64)
            continue
        codes = fb_matrix(np.array([gi]), alive)[0]
        alive = alive[codes == code_of(resp)]
    game.alive = alive


def best_moves(game: Game) -> list[dict]:
    """Оценка всех слов словаря как возможного хода. Меньше score — лучше."""
    alive = game.alive
    if alive.size == 0:
        return []

    # при огромном числе живых считаем по самым вероятным: вес редких ≈ 0
    score_set = alive
    if alive.size > SCORE_SET_CAP:
        order = np.argsort(-WEIGHT[alive])[:SCORE_SET_CAP]
        score_set = alive[np.sort(order)]

    w = WEIGHT[score_set]
    total = float(w.sum())
    if total <= 0:
        w = np.ones_like(w)
        total = float(w.sum())

    all_idx = np.arange(N_WORDS)
    codes = fb_matrix(all_idx, score_set)                     # N × A
    flat = codes.astype(np.int64) + (all_idx[:, None] * 243)
    M = np.bincount(flat.ravel(),
                    weights=np.tile(w, N_WORDS),
                    minlength=N_WORDS * 243).reshape(N_WORDS, 243)

    remaining = (M ** 2).sum(axis=1) / total                  # ожидаемый остаток
    groups = (M > 0).sum(axis=1)

    alive_mask = np.zeros(N_WORDS, dtype=bool)
    alive_mask[score_set] = True
    p_hit = np.where(alive_mask, WEIGHT / total, 0.0)
    p_hit = np.clip(p_hit, 0.0, 1.0)

    score = remaining * (1 - p_hit) - p_hit * 0.6

    # сортировка: score, затем лучшее деление (меньше остаток, больше групп)
    order = np.lexsort((-groups, remaining, score))

    # Фильтр мусора, два слоя.
    # 1. Слова с нулевой частотой — это почти всегда обрезки словаря («бедст»,
    #    «второ», «пробо»). Предлагать их бессмысленно: игра их не примет.
    #    Из списка живых кандидатов они не исчезают, только из советов.
    usable_order = order[USABLE[order]]
    if usable_order.size:
        order = usable_order
    # 2. Среди оставшихся предпочитаем обиходное слово, если его остаток
    #    хуже не более чем на 15 % (правило из ТЗ).
    limit = float(remaining[order[0]]) * JUNK_TOLERANCE
    clean_order = order[CLEAN[order]]
    if clean_order.size and float(remaining[clean_order[0]]) <= limit:
        order = clean_order

    out = []
    for i in order[:TOP_MOVES]:
        out.append({
            "word": WORDS[i],
            "remaining": float(remaining[i]),
            "p_hit": float(p_hit[i]),
            "groups": int(groups[i]),
            "is_candidate": bool(alive_mask[i]),
            "clean": bool(CLEAN[i]),
        })
    return out


def likely(game: Game, limit: int = SHOW_CANDIDATES) -> list[tuple[str, float]]:
    alive = game.alive
    order = np.argsort(-WEIGHT[alive])[:limit]
    return [(WORDS[alive[i]], float(ZIPF[alive[i]])) for i in order]

# ---------------------------------------------------------------- память

DEFAULT_DATA = {
    "users": {},      # user_id -> {"words": n, "paid": bool, "name": str}
    "codes": {},      # код -> None | user_id
    "stats": {"words": 0, "moves": 0},
}

data: dict = json.loads(json.dumps(DEFAULT_DATA))
games: dict[int, Game] = {}
DIRTY = {"flag": False}


def save_data() -> None:
    DIRTY["flag"] = True
    try:
        DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        log.warning("Не сохранить данные: %s", e)


def load_data() -> None:
    global data
    if DATA_FILE.exists():
        try:
            stored = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            data = {**json.loads(json.dumps(DEFAULT_DATA)), **stored}
            log.info("Данные загружены: пользователей %s, кодов %s",
                     len(data["users"]), len(data["codes"]))
        except Exception as e:
            log.warning("Файл данных битый (%s) — начинаю с чистого", e)
    for code in SEED_CODES:
        data["codes"].setdefault(code, None)


def data_bytes() -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8")


async def backup_to_telegram(bot: Bot) -> bool:
    if not STORAGE_CHAT:
        return False
    try:
        paid = sum(1 for u in data["users"].values() if u.get("paid"))
        sent = await bot.send_document(
            STORAGE_CHAT,
            BufferedInputFile(data_bytes(), filename="solver_data.json"),
            caption=(f"💾 Память решателя · {now_msk():%d.%m.%Y %H:%M}\n"
                     f"пользователей {len(data['users'])}, оплативших {paid}, "
                     f"кодов {len(data['codes'])}"),
            disable_notification=True,
        )
        try:
            await bot.unpin_all_chat_messages(STORAGE_CHAT)
        except Exception:
            pass
        await bot.pin_chat_message(STORAGE_CHAT, sent.message_id, disable_notification=True)
        DIRTY["flag"] = False
        log.info("Копия данных сохранена в телеграм")
        return True
    except Exception as e:
        log.warning("Копию в телеграм не отправить: %s", e)
        return False


async def restore_from_telegram(bot: Bot) -> bool:
    global data
    if not STORAGE_CHAT or DATA_FILE.exists():
        return False
    try:
        chat = await bot.get_chat(STORAGE_CHAT)
        msg = chat.pinned_message
        if not msg or not msg.document:
            log.info("Копии в телеграме нет — начинаю с чистого листа")
            return False
        file = await bot.get_file(msg.document.file_id)
        buf = await bot.download_file(file.file_path)
        stored = json.loads(buf.read().decode("utf-8"))
        data = {**json.loads(json.dumps(DEFAULT_DATA)), **stored}
        paid = sum(1 for u in data["users"].values() if u.get("paid"))
        log.info("Данные восстановлены из телеграма: пользователей %s, оплативших %s",
                 len(data["users"]), paid)
        DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return True
    except Exception as e:
        log.warning("Не восстановить из телеграма: %s", e)
        return False


async def backup_loop(bot: Bot) -> None:
    while True:
        await asyncio.sleep(BACKUP_EVERY_SEC)
        if DIRTY["flag"]:
            await backup_to_telegram(bot)

# ---------------------------------------------------------------- доступ и оплата

router = Router()


def is_admin(user_id: int) -> bool:
    return not ADMIN_IDS or user_id in ADMIN_IDS


def user_rec(message: Message) -> dict:
    uid = str(message.from_user.id)
    rec = data["users"].setdefault(uid, {"words": 0, "paid": False, "name": ""})
    name = message.from_user.username or message.from_user.first_name or ""
    if name and rec.get("name") != name:
        rec["name"] = name
        save_data()
    return rec


def pay_message() -> str:
    if PAY_TEXT:
        return PAY_TEXT
    return ("Бесплатные слова закончились.\n\n"
            "Чтобы продолжить, нужен код разблокировки — напишите мне "
            "и я пришлю реквизиты.\n\n"
            "Код вводится так: <code>/code ВАШ-КОД</code>")


def quota_ok(message: Message) -> tuple[bool, str]:
    """Можно ли начать новое слово."""
    if not PAYWALL or is_admin(message.from_user.id):
        return True, ""
    rec = user_rec(message)
    if rec.get("paid"):
        return True, ""
    if rec.get("words", 0) < FREE_WORDS:
        return True, ""
    return False, pay_message()


def bill_word(message: Message, game: Game) -> None:
    """Списываем слово из квоты — один раз на слово."""
    if game.billed:
        return
    game.billed = True
    data["stats"]["words"] = data["stats"].get("words", 0) + 1
    if not PAYWALL or is_admin(message.from_user.id):
        save_data()
        return
    rec = user_rec(message)
    if not rec.get("paid"):
        rec["words"] = rec.get("words", 0) + 1
    save_data()


def quota_tail(message: Message) -> str:
    if not PAYWALL or is_admin(message.from_user.id):
        return ""
    rec = user_rec(message)
    if rec.get("paid"):
        return ""
    left = max(0, FREE_WORDS - rec.get("words", 0))
    return f"\n\n<i>Бесплатных слов осталось: {left}</i>" if left else ""


async def access_guard(handler, event, ctx):
    message = event if isinstance(event, Message) else getattr(event, "message", None)
    if message is None:
        return await handler(event, ctx)
    user = event.from_user
    if user and user.is_bot:
        return None
    text = message.text or ""
    if text.startswith("/chatid") or not ALLOWED_CHATS:
        return await handler(event, ctx)
    if message.chat.type in ("group", "supergroup"):
        if message.chat.id not in ALLOWED_CHATS:
            log.warning("Чужой чат %s (%s) — игнорирую", message.chat.id, message.chat.title)
            return None
    elif not (user and (user.id in ALLOWED_CHATS or is_admin(user.id))):
        log.warning("Чужая личка от %s — игнорирую", user.id if user else "?")
        return None
    return await handler(event, ctx)

# ---------------------------------------------------------------- тексты

HELP = (
    "🎯 <b>Решатель «Пяти букв»</b>\n\n"
    "Пишите мне слово и отклик пятью цифрами:\n"
    "<code>0</code> — буквы нет · <code>1</code> — есть, но не здесь · "
    "<code>2</code> — на своём месте\n\n"
    "<code>норка 00210</code>\n"
    "Н — нет, О — нет, Р — на месте, К — есть не здесь, А — нет.\n\n"
    "Можно слитно (<code>норка00210</code>), можно несколько ходов сразу — "
    "с новой строки или через <code>;</code>\n\n"
    "<b>Команды</b>\n"
    "/solve — новое слово (сброс)\n"
    "/hint — подсказать ход прямо сейчас\n"
    "/state — что известно и сколько кандидатов\n"
    "/list — живые кандидаты\n"
    "/undo — отменить последний ход\n"
    "/rules — эта справка\n\n"
    "<i>В общем чате я откликаюсь только на точный формат «слово и пять цифр», "
    "чтобы не мешать игровому боту.</i>"
)

FORMAT_HINT = ("Формат: слово и пять цифр.\n<code>норка 00210</code>\n\n"
               "Цифры: 0 — нет буквы, 1 — есть не здесь, 2 — на месте.")

# ---------------------------------------------------------------- вывод

async def render_answer(game: Game, message: Message, warnings: list[str]) -> str:
    alive = game.alive
    used = len(game.moves)
    left = MAX_ATTEMPTS - used
    head = "\n".join(f"⚠️ {w}" for w in warnings)
    if head:
        head += "\n\n"

    if alive.size == 0:
        return (head + "❌ <b>Ограничения противоречивы</b> — ни одно слово словаря "
                "не подходит под все отклики.\n\nПроверьте записанные цифры: "
                "/state, отменить последний ход — /undo.")

    if alive.size == 1:
        return head + f"✅ <b>Это {WORDS[alive[0]].upper()}</b>\n\nНовое слово — /solve"

    if alive.size == 2:
        a, b = WORDS[alive[0]], WORDS[alive[1]]
        return (head + f"Осталось два: <b>{a.upper()}</b> и <b>{b.upper()}</b>\n\n"
                f"→ Бей <b>{a.upper()}</b>")

    lines = [head + f"Живых кандидатов: <b>{alive.size}</b>  ·  попытка {used + 1} из {MAX_ATTEMPTS}"]

    probable = likely(game)
    if probable:
        lines.append("\nВероятные: " + " ".join(f"{w}({z:.1f})" for w, z in probable))

    # numpy считает секунды — уводим в поток, чтобы бот не подвисал для других
    moves = await asyncio.to_thread(best_moves, game)
    if moves:
        lines.append("\nЛучшие ходы:")
        for m in moves:
            tail = "  ← кандидат" if m["is_candidate"] else ""
            lines.append(f"  <code>{m['word']}</code>   остаток {m['remaining']:.1f}   "
                         f"шанс попасть {m['p_hit'] * 100:.0f}%{tail}")
        if alive.size > left and left <= 2:
            lines.append("\n<i>Кандидатов больше, чем попыток — бейте разведчика, "
                         "а не перебирайте по одному.</i>")
        lines.append(f"\n→ Бей <b>{moves[0]['word'].upper()}</b>")

    return "\n".join(lines) + quota_tail(message)

# ---------------------------------------------------------------- команды

@router.message(Command("myid"))
async def cmd_chatid(message: Message):
    await message.answer(
        f"id этого чата: <code>{message.chat.id}</code>\n"
        f"ваш id: <code>{message.from_user.id}</code>\n\n"
        f"Первое — в переменную ALLOWED_CHATS, второе — в ADMIN_IDS."
    )


@router.message(CommandStart(), F.chat.type == "private")
@router.message(Command("rules", "формат", "справка"))
async def cmd_help(message: Message):
    user_rec(message)
    await message.answer(HELP + "\n\n" + f"Первым ходом бей <b>{OPENER.upper()}</b>.")


@router.message(Command("solve", "решать", "слово"))
async def cmd_new(message: Message):
    ok, why = quota_ok(message)
    if not ok:
        await message.answer(why)
        return
    games[message.chat.id] = Game()
    await message.answer(
        f"🆕 <b>Новое слово.</b> Начинаем с чистого листа.\n\n"
        f"Кандидатов: {N_WORDS}\n\n"
        f"→ Бей <b>{OPENER.upper()}</b>\n"
        f"<i>Лучшее первое слово: ставит Р на третью позицию и А на пятую — "
        f"самые частые буквы в этих местах.</i>\n\n"
        f"Потом пришлите отклик: <code>{OPENER} 00210</code>"
        + quota_tail(message)
    )


@router.message(Command("hint", "ход", "подскажи"))
async def cmd_hint(message: Message):
    """Пересчитать и подсказать ход, не вводя новый отклик."""
    game = games.get(message.chat.id)
    if not game or not game.moves:
        await message.answer(f"Ходов пока нет.\n\n→ Бей <b>{OPENER.upper()}</b>")
        return
    waiting = None
    if game.alive.size > SLOW_THRESHOLD:
        waiting = await message.answer(f"⏳ Считаю, кандидатов пока {game.alive.size}…")
    text = await render_answer(game, message, [])
    if waiting:
        try:
            await waiting.edit_text(text)
            return
        except Exception:
            pass
    await message.answer(text)


@router.message(Command("state", "состояние"))
async def cmd_state(message: Message):
    game = games.get(message.chat.id)
    if not game or not game.moves:
        await message.answer(f"Ходов пока нет. Кандидатов {N_WORDS}.\n\n"
                             f"→ Бей <b>{OPENER.upper()}</b>")
        return
    lines = ["📋 <b>Что известно</b>\n"]
    for i, (w, r) in enumerate(game.moves, 1):
        lines.append(f"{i}. <code>{w} {r}</code>")
    lines.append(f"\nЖивых кандидатов: <b>{game.alive.size}</b> · "
                 f"попытка {len(game.moves) + 1} из {MAX_ATTEMPTS}")
    await message.answer("\n".join(lines))


@router.message(Command("list", "список"))
async def cmd_list(message: Message):
    game = games.get(message.chat.id)
    if not game:
        await message.answer("Ходов пока нет — /solve и вперёд.")
        return
    alive = game.alive
    if alive.size == 0:
        await message.answer("Кандидатов не осталось — ограничения противоречивы. /undo")
        return
    order = np.argsort(-WEIGHT[alive])[:LIST_LIMIT]
    rows = [f"{WORDS[alive[i]]} ({ZIPF[alive[i]]:.2f})" for i in order]
    tail = f"\n\n…и ещё {alive.size - len(rows)}" if alive.size > len(rows) else ""
    await message.answer(f"🔎 <b>Живые кандидаты ({alive.size})</b>\n\n"
                         + "\n".join(rows) + tail)


@router.message(Command("undo", "отмена"))
async def cmd_undo(message: Message):
    game = games.get(message.chat.id)
    if not game or not game.moves:
        await message.answer("Отменять нечего.")
        return
    word, resp = game.moves.pop()
    await asyncio.to_thread(recompute, game)
    await message.answer(f"↩️ Убрал ход <code>{word} {resp}</code>\n\n"
                         + await render_answer(game, message, []))

# ------------------------------------------------- оплата

@router.message(Command("pay", "оплата"))
async def cmd_pay(message: Message):
    rec = user_rec(message)
    if rec.get("paid"):
        await message.answer("У вас уже полный доступ 👍")
        return
    left = max(0, FREE_WORDS - rec.get("words", 0)) if PAYWALL else "без ограничений"
    await message.answer(f"Бесплатных слов осталось: <b>{left}</b>\n\n" + pay_message())


@router.message(Command("code", "код"))
async def cmd_code(message: Message):
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Напишите так: <code>/code ВАШ-КОД</code>")
        return
    code = parts[1].strip().upper()
    rec = user_rec(message)
    if rec.get("paid"):
        await message.answer("У вас и так полный доступ 👍")
        return
    if code not in data["codes"]:
        await message.answer("Такого кода нет. Проверьте раскладку и дефисы.")
        return
    if data["codes"][code] is not None:
        await message.answer("Этот код уже использован.")
        return
    data["codes"][code] = message.from_user.id
    rec["paid"] = True
    rec["paid_at"] = now_msk().strftime("%d.%m.%Y %H:%M")
    save_data()
    log.info("Код %s активирован пользователем %s", code, message.from_user.id)
    await message.answer("✅ <b>Доступ открыт.</b> Ограничений больше нет.\n\n"
                         "Новое слово — /solve")


@router.message(Command("codes"))
async def cmd_codes(message: Message):
    if not is_admin(message.from_user.id):
        return
    parts = (message.text or "").split()
    try:
        count = min(20, max(1, int(parts[1]))) if len(parts) > 1 else 5
    except ValueError:
        count = 5
    alphabet = string.ascii_uppercase.replace("O", "").replace("I", "") + "23456789"
    fresh = []
    while len(fresh) < count:
        code = "-".join("".join(random.choice(alphabet) for _ in range(4)) for _ in range(2))
        if code not in data["codes"]:
            data["codes"][code] = None
            fresh.append(code)
    save_data()
    free = sum(1 for v in data["codes"].values() if v is None)
    await message.answer("🎟 <b>Новые коды</b>\n\n"
                         + "\n".join(f"<code>{c}</code>" for c in fresh)
                         + f"\n\nВсего неиспользованных: {free}")


@router.message(Command("grant"))
async def cmd_grant(message: Message):
    if not is_admin(message.from_user.id):
        return
    parts = (message.text or "").split()
    if len(parts) < 2 or not parts[1].lstrip("-").isdigit():
        await message.answer("Напишите так: <code>/grant 123456789</code> (id из /chatid)")
        return
    uid = parts[1]
    rec = data["users"].setdefault(uid, {"words": 0, "paid": False, "name": ""})
    rec["paid"] = True
    rec["paid_at"] = now_msk().strftime("%d.%m.%Y %H:%M")
    save_data()
    await message.answer(f"✅ Доступ открыт для <code>{uid}</code>")


@router.message(Command("revoke"))
async def cmd_revoke(message: Message):
    if not is_admin(message.from_user.id):
        return
    parts = (message.text or "").split()
    if len(parts) < 2:
        await message.answer("Напишите так: <code>/revoke 123456789</code>")
        return
    rec = data["users"].get(parts[1])
    if not rec:
        await message.answer("Такого пользователя не вижу.")
        return
    rec["paid"] = False
    save_data()
    await message.answer(f"Доступ закрыт для <code>{parts[1]}</code>")


@router.message(Command("stat", "stats"))
async def cmd_stat(message: Message):
    if not is_admin(message.from_user.id):
        return
    users = data["users"]
    paid = [u for u in users.values() if u.get("paid")]
    free_codes = sum(1 for v in data["codes"].values() if v is None)
    lines = [f"📊 <b>Статистика</b>\n",
             f"Пользователей: {len(users)} · оплативших: {len(paid)}",
             f"Слов разобрано: {data['stats'].get('words', 0)} · "
             f"ходов: {data['stats'].get('moves', 0)}",
             f"Кодов свободных: {free_codes} из {len(data['codes'])}",
             f"Платный режим: {'включён' if PAYWALL else 'выключен'} "
             f"(бесплатно {FREE_WORDS})"]
    top = sorted(users.items(), key=lambda kv: -kv[1].get("words", 0))[:10]
    if top:
        lines.append("\nАктивные:")
        for uid, u in top:
            mark = "💰" if u.get("paid") else "▫️"
            lines.append(f"{mark} {u.get('name') or uid} — слов {u.get('words', 0)}")
    await message.answer("\n".join(lines))


@router.message(Command("sinfo"))
async def cmd_version(message: Message):
    await message.answer(
        f"🔖 <b>Версия: {VERSION}</b>\n\n"
        f"Слов в словаре: {N_WORDS}\n"
        f"Платный режим: {'включён' if PAYWALL else 'выключен'}\n"
        f"Разрешённые чаты: {'все' if not ALLOWED_CHATS else len(ALLOWED_CHATS)}"
    )

# ---------------------------------------------------------------- разбор ходов

# слово берём целиком: «норкаа 00210» — это не «оркаа», а ошибка длины
MOVE_RE = re.compile(r"(?<![а-яё])([а-яё]{5})(?![а-яё])\s*([012]{5})(?![0-9])", re.IGNORECASE)
LOOSE_RE = re.compile(r"(?<![а-яё])([а-яё]{2,10})(?![а-яё])\s*([0-9]{1,10})(?![0-9])", re.IGNORECASE)


def split_moves(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[\n;]+", text or "") if p.strip()]


@router.message(F.text.regexp(LOOSE_RE))
async def handle_move(message: Message):
    # В общем чате рядом работает игровой бот, туда постоянно пишут слова и
    # цифры. Поэтому в группе отвечаем, только если формат точный: слово и
    # ровно пять цифр из 0/1/2. Всё остальное молча пропускаем.
    in_group = message.chat.type in ("group", "supergroup")
    if in_group and not MOVE_RE.search(message.text or ""):
        return

    chunks = split_moves(message.text)
    parsed: list[tuple[str, str]] = []
    problems: list[str] = []

    for chunk in chunks:
        m = MOVE_RE.search(chunk)
        if m:
            parsed.append((normalize(m.group(1)), m.group(2)))
            continue
        loose = LOOSE_RE.search(chunk)
        if not loose:
            continue
        word, digits = normalize(loose.group(1)), loose.group(2)
        if len(word) != 5:
            problems.append(f"«{word}» — не пять букв")
        elif len(digits) != 5:
            problems.append(f"«{chunk.strip()}» — нужно ровно пять цифр, а их {len(digits)}")
        else:
            problems.append(f"«{chunk.strip()}» — в цифрах только 0, 1 и 2")

    if not parsed:
        if problems and not in_group:
            await message.answer("⚠️ " + "\n⚠️ ".join(problems[:4]) + "\n\n" + FORMAT_HINT)
        return

    game = games.get(message.chat.id)
    if game is None:
        game = games[message.chat.id] = Game()

    if not game.moves:
        ok, why = quota_ok(message)
        if not ok:
            await message.answer(why)
            return

    warnings = [] if in_group else list(problems)
    for word, resp in parsed:
        if len(game.moves) >= MAX_ATTEMPTS:
            warnings.append(f"Попытки закончились — «{word}» не учёл. /solve для нового слова")
            break
        if word not in INDEX:
            warnings.append(f"«{word}» нет в моём словаре — ход учёл, но в кандидаты не беру")
        game.moves.append((word, resp))

    bill_word(message, game)
    data["stats"]["moves"] = data["stats"].get("moves", 0) + len(parsed)
    save_data()

    await asyncio.to_thread(recompute, game)

    try:
        with MOVE_LOG.open("a", encoding="utf-8") as f:
            for word, resp in parsed:
                f.write(f"{now_msk():%Y-%m-%d %H:%M}\t{message.chat.id}\t"
                        f"{word}\t{resp}\t{game.alive.size}\n")
    except Exception:
        pass

    # при большом числе кандидатов расчёт идёт несколько секунд — предупреждаем
    waiting = None
    if game.alive.size > SLOW_THRESHOLD:
        waiting = await message.answer(f"⏳ Считаю, кандидатов пока {game.alive.size}…")

    text = await render_answer(game, message, warnings)
    if waiting:
        try:
            await waiting.edit_text(text)
            return
        except Exception:
            pass
    await message.answer(text)


@router.message(F.text.regexp(r"^[а-яё]{5}$", flags=re.IGNORECASE),
                F.chat.type == "private")
async def handle_bare_word(message: Message):
    await message.answer("Это только слово — нужен ещё отклик.\n\n" + FORMAT_HINT)

# ---------------------------------------------------------------- запуск

def get_token() -> str:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        f = BASE_DIR / "token.txt"
        if f.exists():
            token = f.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("Не найден токен: задайте BOT_TOKEN или положите token.txt.")
    return token


async def on_error(event: ErrorEvent) -> None:
    if isinstance(event.exception, TelegramNetworkError):
        log.error("Нет связи с Telegram — сообщение не доставлено.")
    else:
        log.exception("Ошибка при обработке: %s", event.exception)


async def main() -> None:
    if not N_WORDS:
        raise SystemExit("Словарь пуст — положите dict_bot.txt рядом с bot.py.")
    load_data()
    bot = Bot(token=get_token(),
              default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await restore_from_telegram(bot)

    dp = Dispatcher()
    router.message.outer_middleware(access_guard)
    dp.include_router(router)
    dp.errors.register(on_error)

    log.info("ВЕРСИЯ %s", VERSION)
    log.info("Слов в словаре: %s · первый ход: %s", N_WORDS, OPENER)
    log.info("Разрешённые чаты: %s · руководители: %s",
             ALLOWED_CHATS or "(все — задайте ALLOWED_CHATS)",
             ADMIN_IDS or "(все — задайте ADMIN_IDS)")
    log.info("Платный режим: %s (бесплатно слов: %s) · хранилище копий: %s",
             "включён" if PAYWALL else "выключен", FREE_WORDS, STORAGE_CHAT or "нет")

    asyncio.create_task(backup_loop(bot))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
