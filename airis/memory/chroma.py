"""
Airis ChromaDB Vector Memory Module.
Provides multilingual semantic memory with Ollama bge-m3 embeddings (1024-d cosine),
legacy collection migration with backup, deduplication, recall, identity tracking, and memory management.
"""

from __future__ import annotations
import os
import re
import json
import uuid
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple
import chromadb
import chromadb.utils.embedding_functions as ef
from colorama import Fore

from airis.config import load_config, BASE_DIR, LOGS_DIR
from airis.core.locks import chroma_lock

_cfg = load_config()
CHROMA_DIR = os.path.join(BASE_DIR, _cfg.memory.chroma_dir)
OLLAMA_EMBED_URL = _cfg.ollama.url
EMBEDDING_MODEL_NAME = _cfg.ollama.embedding_model
MEMORY_RESULTS = _cfg.memory.results_count
MEMORY_MAX_DISTANCE = _cfg.memory.max_distance

CORE_USER_NAME_ID = "core:user_name"

# Patterns for identity detection
NAME_COMMAND_PATTERN = re.compile(
    r"^(?:/name|!имя)\s+([A-Za-zА-Яа-яЁё0-9_\-]{2,30})$", re.I | re.U
)
NAME_FALLBACK_PATTERN = re.compile(
    r"(?:меня зовут|мо[её]\s+имя|называй меня)\s+[\"']?([A-Za-zА-Яа-яЁё0-9_\-]{2,30})[\"']?", re.I | re.U
)


class ChromaInitResult(tuple):
    """
    Polymorphic tuple (client, collection) supporting tuple unpacking
    and direct forwarding of collection attributes and methods.
    """
    def __new__(cls, client, col):
        return super().__new__(cls, (client, col))

    @property
    def client(self):
        return self[0]

    @property
    def collection(self):
        return self[1]

    def count(self, *args, **kwargs):
        if not args and not kwargs:
            return self[1].count()
        return super().count(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self[1], name)


def initialize_and_migrate_chroma(
    chroma_dir: Optional[str] = None,
    embedding_fn=None,
    backup_dir: Optional[str] = None,
    db_path: Optional[str] = None
) -> ChromaInitResult:
    """
    Initializes ChromaDB PersistentClient and migrates legacy collection to
    bge-m3 1024-d cosine collection while preserving all pre-existing facts.
    Guarantees two-pass idempotency: subsequent runs on an already-migrated
    collection execute a clean no-op without re-migrating or losing data.
    """
    target_dir = chroma_dir or db_path or CHROMA_DIR
    if embedding_fn is None:
        embedding_fn = ef.OllamaEmbeddingFunction(
            url=OLLAMA_EMBED_URL,
            model_name=EMBEDDING_MODEL_NAME
        )
    target_backup_dir = backup_dir or LOGS_DIR

    client = chromadb.PersistentClient(path=target_dir)
    existing_collections = [c.name for c in client.list_collections()]
    migrated_data = None

    if "iris_memory" in existing_collections:
        old_col = client.get_collection("iris_memory")
        meta = old_col.metadata or {}

        # Check for legacy schema: space, embedding function name, vector dimension
        is_legacy = False
        if meta.get("hnsw:space") != "cosine":
            is_legacy = True
        else:
            config_json = getattr(old_col, "configuration_json", None) or {}
            ef_info = config_json.get("embedding_function") if isinstance(config_json, dict) else {}
            if not isinstance(ef_info, dict):
                ef_info = {}

            ef_name = ef_info.get("name")
            ef_model = ef_info.get("config", {}).get("model_name") if isinstance(ef_info.get("config"), dict) else None

            if ef_name != "ollama" or ef_model != "bge-m3":
                is_legacy = True
            elif old_col.count() > 0:
                sample = old_col.get(limit=1, include=["embeddings"])
                embs = sample.get("embeddings")
                if embs is not None and len(embs) > 0 and len(embs[0]) != 1024:
                    is_legacy = True

        if is_legacy:
            logging.info("[MEMORY] Legacy memory collection detected. Migrating to bge-m3 (1024-d)...")
            migrated_data = old_col.get(include=["documents", "metadatas"])

            # 1. Primary backup in target db directory
            os.makedirs(target_dir, exist_ok=True)
            backup_path = os.path.join(target_dir, "migration_backup.json")
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(migrated_data, f, ensure_ascii=False, indent=2)

            # 2. Additional backup in logs directory
            if target_backup_dir:
                os.makedirs(target_backup_dir, exist_ok=True)
                logs_backup_path = os.path.join(target_backup_dir, "iris_memory_backup.json")
                with open(logs_backup_path, "w", encoding="utf-8") as f:
                    json.dump(migrated_data, f, ensure_ascii=False, indent=2)

            client.delete_collection("iris_memory")

    if "iris_memory" not in [c.name for c in client.list_collections()]:
        col = client.create_collection(
            name="iris_memory",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        if migrated_data and migrated_data.get("ids"):
            col.add(
                ids=migrated_data["ids"],
                documents=migrated_data["documents"],
                metadatas=migrated_data["metadatas"]
            )
            logging.info(f"[MEMORY] Successfully migrated {len(migrated_data['ids'])} memories to bge-m3.")
    else:
        col = client.get_collection("iris_memory", embedding_function=embedding_fn)

    return ChromaInitResult(client, col)


def recall_memory(*args, **kwargs) -> List[str]:
    """
    Retrieves matching semantic facts from collection where entity matches and distance <= threshold.
    Supports both direct calling convention:
        recall_memory(query, entity="user") -> uses module collection and chroma_lock
    and test suite convention:
        recall_memory(collection, query, entity="user", threshold=0.35, lock=None)
        recall_memory(collection, user_text, entity="user", max_distance=0.35, n_results=3, lock=None)
    """
    if len(args) >= 1 and isinstance(args[0], str):
        col = collection
        target_text = args[0]
        entity = args[1] if len(args) > 1 else kwargs.get("entity", "user")
        max_dist = args[2] if len(args) > 2 else kwargs.get("threshold", kwargs.get("max_distance", MEMORY_MAX_DISTANCE))
        n_res = args[3] if len(args) > 3 else kwargs.get("n_results", MEMORY_RESULTS)
        use_lock = args[4] if len(args) > 4 else kwargs.get("lock", chroma_lock)
    elif len(args) >= 1:
        col = args[0]
        target_text = args[1] if len(args) > 1 else kwargs.get("user_text", kwargs.get("query", ""))
        entity = args[2] if len(args) > 2 else kwargs.get("entity", "user")
        max_dist = args[3] if len(args) > 3 else kwargs.get("threshold", kwargs.get("max_distance", MEMORY_MAX_DISTANCE))
        n_res = args[4] if len(args) > 4 else kwargs.get("n_results", MEMORY_RESULTS)
        use_lock = args[5] if len(args) > 5 else kwargs.get("lock", None)
    else:
        col = kwargs.get("collection", collection)
        target_text = kwargs.get("user_text", kwargs.get("query", ""))
        entity = kwargs.get("entity", "user")
        max_dist = kwargs.get("threshold", kwargs.get("max_distance", MEMORY_MAX_DISTANCE))
        n_res = kwargs.get("n_results", MEMORY_RESULTS)
        use_lock = kwargs.get("lock", chroma_lock if col is collection else None)

    if col is None or not isinstance(target_text, str) or not target_text.strip():
        return []

    def _query():
        try:
            res = col.query(
                query_texts=[target_text],
                n_results=n_res,
                where={"entity": entity}
            )
            docs = res.get("documents", [[]])[0]
            distances = res.get("distances", [[]])[0]
            return [doc for doc, dist in zip(docs, distances) if dist <= max_dist]
        except Exception as e:
            logging.error(f"[MEMORY] Retrieval error ({entity}): {e}")
            return []

    if use_lock:
        with use_lock:
            return _query()
    return _query()


def insert_fact(*args, **kwargs) -> bool:
    """
    Deduplicates (rejects if distance <= threshold) and inserts new fact document.
    Supports both direct calling convention:
        insert_fact(fact, entity="user") -> uses module collection and chroma_lock
    and test suite convention:
        insert_fact(collection, fact, entity="user", threshold=0.25, lock=None)
        insert_fact(collection, fact, entity="user", dedup_threshold=0.25, lock=None)
    """
    if len(args) >= 1 and isinstance(args[0], str):
        col = collection
        target_fact = args[0]
        entity = args[1] if len(args) > 1 else kwargs.get("entity", "user")
        threshold = args[2] if len(args) > 2 else kwargs.get("threshold", kwargs.get("dedup_threshold", 0.25))
        use_lock = args[3] if len(args) > 3 else kwargs.get("lock", chroma_lock)
    elif len(args) >= 1:
        col = args[0]
        target_fact = args[1] if len(args) > 1 else kwargs.get("fact", "")
        entity = args[2] if len(args) > 2 else kwargs.get("entity", "user")
        threshold = args[3] if len(args) > 3 else kwargs.get("threshold", kwargs.get("dedup_threshold", 0.25))
        use_lock = args[4] if len(args) > 4 else kwargs.get("lock", None)
    else:
        col = kwargs.get("collection", collection)
        target_fact = kwargs.get("fact", "")
        entity = kwargs.get("entity", "user")
        threshold = kwargs.get("threshold", kwargs.get("dedup_threshold", 0.25))
        use_lock = kwargs.get("lock", chroma_lock if col is collection else None)

    if col is None or not isinstance(target_fact, str) or not target_fact.strip():
        return False

    def _insert():
        try:
            res = col.query(
                query_texts=[target_fact],
                n_results=1,
                where={"entity": entity}
            )
            distances = res.get("distances", [[]])[0]
            if distances and len(distances) > 0 and distances[0] <= threshold:
                return False  # Duplicate detected
            col.add(
                ids=[str(uuid.uuid4())],
                documents=[target_fact],
                metadatas=[{"entity": entity}]
            )
            return True
        except Exception as e:
            logging.error(f"[MEMORY] Error adding fact ({entity}): {e}")
            return False

    if use_lock:
        with use_lock:
            return _insert()
    return _insert()


def get_core_fact(key: str, collection_obj=None, lock: Optional[threading.Lock] = None) -> Optional[str]:
    """Retrieves a point system fact by key id under chroma_lock."""
    col = collection_obj if collection_obj is not None else collection
    if col is None:
        return None
    use_lock = lock if lock is not None else chroma_lock

    def _get():
        try:
            result = col.get(ids=[key])
            docs = result.get("documents") or []
            if docs:
                return docs[0]
        except Exception:
            pass
        return None

    if use_lock:
        with use_lock:
            return _get()
    return _get()


def get_current_user_name(collection_obj=None, lock: Optional[threading.Lock] = None) -> str:
    """Retrieves the active user name (defaults to l3wsha)."""
    fact = get_core_fact(CORE_USER_NAME_ID, collection_obj=collection_obj, lock=lock)
    if fact:
        if ":" in fact:
            return fact.split(":")[-1].strip()
        return fact.strip()
    return _cfg.personality.user_name_default


def maybe_store_identity_fact(user_text: str, collection_obj=None, lock: Optional[threading.Lock] = None) -> bool:
    """Changes user name on explicit command or natural statement under chroma_lock."""
    col = collection_obj if collection_obj is not None else collection
    if col is None:
        return False

    clean_text = user_text.strip()
    match = NAME_COMMAND_PATTERN.match(clean_text)

    if not match:
        match = NAME_FALLBACK_PATTERN.search(clean_text)
        if not match:
            return False

    name = match.group(1).strip(" .,!?\"'")

    stop_words = {"устал", "дома", "тут", "здесь", "человек", "парень", "девушка", "все", "никто", "ник"}
    if not (1 < len(name) <= 30) or name.lower() in stop_words:
        return False

    use_lock = lock if lock is not None else chroma_lock

    def _upsert():
        try:
            col.upsert(
                ids=[CORE_USER_NAME_ID],
                documents=[f"Имя собеседника: {name}"],
            )
            logging.info(f"[ПАМЯТЬ] Имя собеседника обновлено: {name}")
            print(Fore.GREEN + f"  ✓ Имя обновлено: {name}\n")
            return True
        except Exception as e:
            logging.error(f"[MEMORY] Ошибка записи имени: {e}")
            return False

    if use_lock:
        with use_lock:
            return _upsert()
    return _upsert()


def list_memories(
    collection_obj=None,
    entity: Optional[str] = None,
    lock: Optional[threading.Lock] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves all stored facts from ChromaDB collection formatted as dictionaries.
    Used by /memories command and management interfaces.
    """
    col = collection_obj if collection_obj is not None else collection
    if col is None:
        return []

    use_lock = lock if lock is not None else chroma_lock
    with use_lock:
        try:
            where_clause = {"entity": entity} if entity else None
            data = col.get(where=where_clause) if where_clause else col.get()
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            ids = data.get("ids") or []

            results = []
            for doc_id, doc, meta in zip(ids, docs, metas):
                if not doc:
                    continue
                ent = meta.get("entity", "unknown") if isinstance(meta, dict) else "unknown"
                results.append({
                    "id": doc_id,
                    "document": doc,
                    "entity": ent
                })
            return results
        except Exception as e:
            logging.error(f"[MEMORY] Error listing memories: {e}")
            return []


def list_stored_memories(
    collection_obj=None,
    lock: Optional[threading.Lock] = None
) -> Dict[str, Any]:
    """
    Retrieves and categorizes stored memories into user_name, user_facts, self_facts, total.
    """
    items = list_memories(collection_obj=collection_obj, lock=lock)
    user_name = None
    user_facts = []
    self_facts = []

    for item in items:
        doc_id = item["id"]
        doc = item["document"]
        ent = item["entity"]

        if doc_id == CORE_USER_NAME_ID or str(doc).startswith("Имя собеседника:"):
            user_name = doc.split(":")[-1].strip() if ":" in doc else doc
            continue

        if ent == "self":
            self_facts.append((doc_id, doc))
        else:
            user_facts.append((doc_id, doc))

    return {
        "user_name": user_name or "l3wsha",
        "user_facts": user_facts,
        "self_facts": self_facts,
        "total": len(user_facts) + len(self_facts) + (1 if user_name else 0)
    }


def forget_memory(
    query_or_collection: Any,
    collection_or_query: Any = None,
    threshold: float = 0.45,
    collection_obj: Any = None,
    lock: Optional[threading.Lock] = None
) -> Tuple[bool, str]:
    """
    Locates and removes matching memory from ChromaDB vector collection.
    Supports polymorphic signatures:
      - forget_memory(query, collection_obj=None, threshold=0.45)
      - forget_memory(collection, query, threshold=0.45)
    Guards core user identity against deletion.
    """
    if isinstance(query_or_collection, str):
        query = query_or_collection
        col = collection_or_query if collection_or_query is not None else (collection_obj or collection)
    else:
        col = query_or_collection if query_or_collection is not None else (collection_obj or collection)
        query = collection_or_query

    if not query or not str(query).strip():
        return False, "Поисковый запрос для удаления не может быть пустым."

    query_str = str(query).strip()

    if col is None:
        return False, "Коллекция ChromaDB не инициализирована."

    use_lock = lock if lock is not None else chroma_lock
    with use_lock:
        try:
            # 0. Explicit user identity query guard (safe extraction under lock)
            q_lower = query_str.lower().strip()
            core_protected_tokens = {
                CORE_USER_NAME_ID.lower(),
                "имя",
                "имя собеседника",
                "собеседника:",
                "собеседник",
            }
            active_user_name = None
            try:
                name_res = col.get(ids=[CORE_USER_NAME_ID])
                if name_res and name_res.get("documents"):
                    doc_str = str(name_res["documents"][0])
                    if ":" in doc_str:
                        active_user_name = doc_str.split(":", 1)[1].strip()
                    else:
                        active_user_name = doc_str.strip()
            except Exception:
                pass

            if not active_user_name and _cfg and hasattr(_cfg, "personality"):
                active_user_name = _cfg.personality.user_name_default

            if active_user_name:
                core_protected_tokens.add(active_user_name.lower())
                core_protected_tokens.add(f"имя собеседника: {active_user_name}".lower())

            if q_lower in core_protected_tokens or q_lower.startswith("имя собеседника:"):
                return False, "Нельзя удалить имя собеседника через /forget. Используйте /name для смены имени."

            # 1. Semantic query with top 3 candidates
            res = col.query(
                query_texts=[query_str],
                n_results=3
            )
            docs = res.get("documents", [[]])[0] if res.get("documents") else []
            ids = res.get("ids", [[]])[0] if res.get("ids") else []
            distances = res.get("distances", [[]])[0] if res.get("distances") else []

            # Check semantic candidates within threshold
            for cand_id, cand_doc, raw_dist in zip(ids, docs, distances):
                try:
                    cand_dist = float(raw_dist) if raw_dist is not None else 1.0
                except (ValueError, TypeError):
                    cand_dist = 1.0

                if cand_dist <= threshold:
                    if cand_id == CORE_USER_NAME_ID or str(cand_doc).startswith("Имя собеседника:"):
                        return False, "Нельзя удалить имя собеседника через /forget. Используйте /name для смены имени."
                    col.delete(ids=[cand_id])
                    logging.info(f"[MEMORY] Deleted memory id={cand_id}: {cand_doc} (dist={cand_dist:.3f})")
                    return True, f"Удалено воспоминание: \"{cand_doc}\" (сходство: {1 - cand_dist:.2f})"

            # 2. Fallback: exact case-insensitive substring match in all documents (excluding user identity)
            all_data = col.get()
            all_ids = all_data.get("ids", []) or []
            all_docs = all_data.get("documents", []) or []
            for cand_id, cand_doc in zip(all_ids, all_docs):
                if not cand_doc:
                    continue
                if cand_id == CORE_USER_NAME_ID or str(cand_doc).startswith("Имя собеседника:"):
                    continue
                if query_str.lower() in str(cand_doc).lower():
                    col.delete(ids=[cand_id])
                    logging.info(f"[MEMORY] Deleted memory by substring id={cand_id}: {cand_doc}")
                    return True, f"Удалено точное совпадение: \"{cand_doc}\""

            # 3. Informative error message when no match found
            closest_doc = None
            closest_dist = 1.0
            for cand_id, cand_doc, raw_dist in zip(ids, docs, distances):
                if cand_id != CORE_USER_NAME_ID and not str(cand_doc).startswith("Имя собеседника:"):
                    closest_doc = cand_doc
                    try:
                        closest_dist = float(raw_dist) if raw_dist is not None else 1.0
                    except (ValueError, TypeError):
                        closest_dist = 1.0
                    break

            if closest_doc is not None:
                return False, f"Ближайшее воспоминание \"{closest_doc}\" слишком далеко (дистанция {closest_dist:.3f} > {threshold:.2f})."
            else:
                return False, f"Воспоминаний по запросу '{query_str}' не найдено."

        except Exception as e:
            logging.error(f"[MEMORY] Error forgetting memory: {e}")
            return False, f"Ошибка при удалении: {e}"


# Global instances initialized at module load
try:
    chroma_embedding_fn = ef.OllamaEmbeddingFunction(
        url=OLLAMA_EMBED_URL,
        model_name=EMBEDDING_MODEL_NAME
    )
    chroma_client, collection = initialize_and_migrate_chroma(
        chroma_dir=CHROMA_DIR,
        embedding_fn=chroma_embedding_fn,
        backup_dir=LOGS_DIR
    )
except Exception as e:
    logging.error(f"[MEMORY] Global ChromaDB initialization error: {e}")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_collection("iris_memory")
    except Exception:
        chroma_embedding_fn = None
        chroma_client = None
        collection = None


def get_default_collection():
    """Returns the default initialized Chroma collection."""
    return collection
