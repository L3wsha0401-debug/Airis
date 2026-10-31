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
from airis.memory.chroma import (
    insert_fact,
    get_default_collection,
    query_candidate_facts,
    update_fact,
    delete_fact_by_id,
    CORE_USER_NAME_ID,
)
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
    """Analyzes user inputs for persistent personal facts with context-aware conflict resolution."""
    col = get_default_collection()
    if col is None:
        return

    text = user_text.strip()
    if len(text) < 15:
        return

    if not FACT_CANDIDATE_PATTERN.search(text):
        return

    # Phase 1: Retrieve top-5 candidate facts from ChromaDB under chroma_lock
    candidate_facts = query_candidate_facts(text, entity="user", n_results=5, collection_obj=col)

    if candidate_facts:
        cand_lines = [
            f'- ID: "{c["id"]}", Факт: "{c["fact"]}"'
            for c in candidate_facts
        ]
        candidates_block = "\n".join(cand_lines)
    else:
        candidates_block = "(сохраненных фактов о пользователе пока нет)"

    prompt = f"""Проанализируй РЕПЛИКУ ПОЛЬЗОВАТЕЛЯ в диалоге с ИИ-компаньоном Айрис и сопоставь её с известными фактами из базы.

СУЩЕСТВУЮЩИЕ ФАКТЫ О ПОЛЬЗОВАТЕЛЕ В БАЗЕ:
{candidates_block}

РЕПЛИКА ПОЛЬЗОВАТЕЛЯ: "{text}"

Определи действие (action) в отношении долгосрочной памяти:
1. "ignore" — в реплике нет устойчивого факта, либо факт уже в точности известен из списка выше (дубликат), либо это имя пользователя.
2. "insert" — новый устойчивый факт о пользователе (железо, авто, софт, ОС, привычки, предпочтения), который не противоречит существующим фактам.
3. "replace" — пользователь изменил предпочтение или сообщил факт, прямо противоречащий одному из существующих фактов (например, сменил ОС, продал машину, начал пить чай вместо кофе). Укажи replace_id старого факта и новый факт.
4. "delete" — пользователь явно аннулировал или отказался от ранее сохраненного факта без замены (например, "больше не курю", "я продал мотоцикл"). Укажи replace_id удаляемого факта.

ПРАВИЛА:
- Факт формулируй кратко (3-7 слов) от третьего лица: "Пользователь перешел на Windows 11".
- Если action="insert": replace_id=null.
- Если action="replace": replace_id="<точный ID из списка выше>", fact="<новый краткий факт>".
- Если action="delete": replace_id="<точный ID из списка выше>", fact="".
- Если action="ignore": replace_id=null, fact="".

Ответь СТРОГО в формате JSON без разметки:
{{"action": "ignore"|"insert"|"replace"|"delete", "fact": "<краткий факт>", "replace_id": "<ID или null>"}}"""

    try:
        # Phase 2: Single LLM reasoning call under ollama_lock
        with ollama_lock:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "temperature": 0.1,
                    "num_predict": 100,
                },
            )

        content = response["message"]["content"].strip()
        if "```" in content:
            content = re.sub(r'```(?:json)?', '', content).strip('` \n')
        data = json.loads(content)

        # Backward compatibility normalization
        raw_action = data.get("action")
        fact = (data.get("fact") or "").strip()
        replace_id = data.get("replace_id")
        if replace_id is not None:
            replace_id = str(replace_id).strip()
            if replace_id.lower() in ("null", "none", ""):
                replace_id = None

        if raw_action:
            action = str(raw_action).lower().strip()
        else:
            # Legacy test responses format: {"has_fact": true, "fact": "..."}
            if data.get("has_fact") and fact:
                action = "insert"
            else:
                action = "ignore"

        if action == "ignore":
            return

        # Phase 3: Execute memory operation
        if action == "delete":
            if replace_id and replace_id != CORE_USER_NAME_ID:
                deleted = delete_fact_by_id(replace_id, collection_obj=col)
                if deleted:
                    logging.info(f"[ПАМЯТЬ АННУЛИРОВАНА (USER)]: id={replace_id}")
            return

        # For insert and replace, validate fact content
        if not (4 <= len(fact) <= 300):
            return

        fact_lower = fact.lower()
        if any(w in fact_lower for w in GARBAGE_FACT_WORDS):
            logging.info(f"[ПАМЯТЬ ОТБРОШЕНА (МУСОР)]: {fact}")
            return

        if action == "replace" and replace_id and replace_id != CORE_USER_NAME_ID:
            updated = update_fact(replace_id, fact, entity="user", collection_obj=col)
            if updated:
                logging.info(f"[ПАМЯТЬ ЗАМЕНЕНА (USER)]: id={replace_id} -> {fact}")
                return

        # Fallback for action == "insert" or failed replace: insert as new fact
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
                # Self-fact extraction disabled in queue processor to prevent lock contention
                pass
        except Exception as e:
            logging.error(f"[MEMORY WORKER] Error processing {item}: {e}")
        finally:
            memory_queue.task_done()


memory_thread = threading.Thread(target=memory_worker, daemon=True)
memory_thread.start()


def shutdown_memory(timeout: float = 8.0, logs_dir: Optional[str] = None):
    """Graceful shutdown of memory worker thread and flush of pending items."""
    print(Fore.YELLOW + "Сохранение фоновых воспоминаний...")
    end_time = time.time() + timeout
    while getattr(memory_queue, "unfinished_tasks", 0) > 0 and time.time() < end_time and (memory_thread.is_alive() if memory_thread else False):
        time.sleep(0.05)
    memory_stop_event.set()
    if memory_thread and memory_thread.is_alive():
        remaining = max(0.1, end_time - time.time())
        memory_thread.join(timeout=remaining)

    while not memory_queue.empty():
        try:
            memory_queue.get_nowait()
            memory_queue.task_done()
        except Exception:
            break

    dump_memory_log(logs_dir=logs_dir)
