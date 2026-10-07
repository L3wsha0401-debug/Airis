"""
Airis Semantic Memory Extraction and Background Worker.
Analyzes user inputs and assistant responses asynchronously to extract and maintain
long-term user preferences and companion identity facts.
"""

from __future__ import annotations
import re
import json
import queue
import logging
import threading
import time
from typing import Optional
import ollama
from colorama import Fore

from airis.config import load_config
from airis.core.locks import ollama_lock
from airis.memory.chroma import insert_fact, get_default_collection
from airis.memory.logger import dump_memory_log

_cfg = load_config()
MODEL_NAME = _cfg.ollama.model

FACT_CANDIDATE_PATTERN = re.compile(
    r'\b(я|меня|мне|мной|мною|мой|моя|мое|моё|мои|моих|моем|моём|моей|у меня|люблю|обожаю|ненавижу|купил|собрал|учусь|работаю|живу|зовут|катаюсь|играю|предпочитаю)\b',
    re.IGNORECASE
)

GARBAGE_FACT_WORDS = [
    'кибертерроризм', 'хакерств', 'ванную комнату', 'расходные материалы',
    'токен', 'модуль', 'алгоритм', 'процессор', 'пуш-донат',
    'превосходств', 'вредоносн', 'террорист', 'ключ для доступа',
    'приветствует меня', 'просит о помощ', 'пытается понять себя',
    'имя пользователя', 'имя собеседника',
]

GARBAGE_SELF_WORDS = [
    'расходные материалы', 'экономия ресурсов', 'пользователи пк',
    'модуль', 'алгоритм', 'процессор', 'хардвар', 'архитектур',
    'анализ', 'оптимизаци', 'функциональн', 'эффективност',
    'вредоносн', 'корень из числа', 'зашифрованн',
    'базы данных', 'проверяет базы',
]


def extract_fact_from_user(user_text: str):
    """Analyzes user inputs for persistent personal facts."""
    col = get_default_collection()
    if col is None:
        return

    text = user_text.strip()
    if len(text) < 15:
        return

    if not FACT_CANDIDATE_PATTERN.search(text):
        return

    prompt = f"""Проанализируй РЕПЛИКУ ПОЛЬЗОВАТЕЛЯ в диалоге с ИИ-компаньоном Айрис.
Определи, есть ли в ней устойчивый факт о пользователе: его железо, авто, софт, ОС, проекты, привычки, предпочтения.

ВАЖНО:
- Приветствия, эмоции и обращения к Айрис — это НЕ факты.
- Не путай факты об Айрис с фактами о пользователе.
- Если это имя пользователя — верни has_fact: false.
- Категорически запрещено придумывать факты.

Сообщение: "{text}"
Ответь СТРОГО в формате JSON без разметки:
{{"has_fact": true, "fact": "<краткий факт о пользователе в 3-7 словах>"}}
или
{{"has_fact": false, "fact": ""}}"""

    try:
        with ollama_lock:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "temperature": 0.1,
                    "num_predict": 80,
                },
            )

        content = response["message"]["content"].strip()
        if "```" in content:
            content = re.sub(r'```(?:json)?', '', content).strip('` \n')
        data = json.loads(content)

        if not data.get("has_fact") or not data.get("fact"):
            return

        fact = data["fact"].strip()
        if not (4 <= len(fact) <= 300):
            return

        fact_lower = fact.lower()
        if any(w in fact_lower for w in GARBAGE_FACT_WORDS):
            logging.info(f"[ПАМЯТЬ ОТБРОШЕНА (МУСОР)]: {fact}")
            return

        inserted = insert_fact(fact, entity="user")
        if inserted:
            logging.info(f"[ПАМЯТЬ СОХРАНЕНА (USER)]: {fact}")
        else:
            logging.info(f"[ПАМЯТЬ ДУБЛИКАТ (USER)]: {fact}")

    except Exception as e:
        logging.error(f"[MEMORY] Error extracting user fact: {e}")


def extract_fact_about_self(iris_text: str):
    """Analyzes assistant utterances to preserve enduring personality traits."""
    col = get_default_collection()
    if col is None or len(iris_text.strip()) < 25:
        return

    prompt = f"""Проанализируй РЕПЛИКУ ИИ-КОМПАНЬОНА АЙРИС (девушка).
Определи, выразила ли она устойчивый, искренний факт или предпочтение О СЕБЕ САМОЙ (ее вкусы, предпочтения, увлечения).

ВАЖНО:
- НЕ сохраняй временные шутки, абсурдные приколы или сарказм (например, если Айрис в шутку говорит глупость).
- Пропускай банальные и контекстные фразы ("я тут", "я ответила", "я слушаю").
- Игнорируй обращения к пользователю.
- Записывай факт строго от 3-го лица в женском роде ("Айрис предпочитает...", "Айрис нравится...", "Айрис считает...").
- Не придумывай ничего от себя.

Реплика Айрис: "{iris_text}"

Ответь СТРОГО в формате JSON без разметки:
{{"has_fact": true, "fact": "<краткий факт от 3-го лица в 3-7 словах>"}}
или
{{"has_fact": false, "fact": ""}}"""

    try:
        with ollama_lock:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "temperature": 0.1,
                    "num_predict": 80,
                },
            )

        content = response["message"]["content"].strip()
        if "```" in content:
            content = re.sub(r'```(?:json)?', '', content).strip('` \n')
        data = json.loads(content)

        if not data.get("has_fact") or not data.get("fact"):
            return

        fact = data["fact"].strip()
        if not (5 <= len(fact) <= 300):
            return

        fact_lower = fact.lower()
        if any(w in fact_lower for w in GARBAGE_SELF_WORDS):
            logging.info(f"[ЛИЧНОСТЬ ОТБРОШЕНА (МУСОР)]: {fact}")
            return

        inserted = insert_fact(fact, entity="self")
        if inserted:
            logging.info(f"[ЛИЧНОСТЬ АЙРИС ОБНОВЛЕНА (SELF)]: {fact}")
        else:
            logging.info(f"[ЛИЧНОСТЬ ДУБЛИКАТ (SELF)]: {fact}")

    except Exception as e:
        logging.error(f"[MEMORY] Error extracting self fact: {e}")


# Background memory worker queue and thread
memory_queue = queue.Queue()
memory_stop_event = threading.Event()


def memory_worker():
    """Background worker draining facts to extract."""
    while not memory_stop_event.is_set():
        try:
            item = memory_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            if isinstance(item, tuple) and len(item) == 2:
                target, text = item
            elif isinstance(item, str):
                target, text = "user", item
            else:
                target, text = None, None

            if target == "user" and text:
                extract_fact_from_user(text)
            elif target == "self" and text:
                extract_fact_about_self(text)
        except Exception as e:
            logging.error(f"[MEMORY WORKER] Error processing {item}: {e}")
        finally:
            memory_queue.task_done()


memory_thread = threading.Thread(target=memory_worker, daemon=True)
memory_thread.start()


def shutdown_memory(timeout: float = 2.0, logs_dir: Optional[str] = None):
    """Graceful shutdown of memory worker thread and flush of pending items."""
    print(Fore.YELLOW + "Сохранение фоновых воспоминаний...")
    end_time = time.time() + timeout
    while not memory_queue.empty() and time.time() < end_time:
        time.sleep(0.05)
    memory_stop_event.set()
    if memory_thread and memory_thread.is_alive():
        memory_thread.join(timeout=timeout)

    dump_memory_log(logs_dir=logs_dir)
