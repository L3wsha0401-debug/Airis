"""
Airis Memory Package.
Provides ChromaDB vector storage, semantic recall, session persistence,
background fact extraction, and memory logging.
"""

from airis.memory.chroma import (
    ChromaInitResult,
    initialize_and_migrate_chroma,
    recall_memory,
    insert_fact,
    get_core_fact,
    get_current_user_name,
    maybe_store_identity_fact,
    list_memories,
    list_stored_memories,
    forget_memory,
    CORE_USER_NAME_ID,
    NAME_COMMAND_PATTERN,
    NAME_FALLBACK_PATTERN,
    chroma_embedding_fn,
    chroma_client,
    collection,
    get_default_collection,
)
from airis.memory.session import (
    save_session_history,
    load_session_history,
    conversation_history,
)
from airis.memory.extractor import (
    FACT_CANDIDATE_PATTERN,
    extract_fact_from_user,
    extract_fact_about_self,
    memory_queue,
    memory_stop_event,
    memory_worker,
    memory_thread,
    shutdown_memory,
)
from airis.memory.logger import (
    dump_memory_log,
    dump_all_memories_to_log,
    log_chat_interaction,
)

__all__ = [
    "ChromaInitResult",
    "initialize_and_migrate_chroma",
    "recall_memory",
    "insert_fact",
    "get_core_fact",
    "get_current_user_name",
    "maybe_store_identity_fact",
    "list_memories",
    "list_stored_memories",
    "forget_memory",
    "CORE_USER_NAME_ID",
    "NAME_COMMAND_PATTERN",
    "NAME_FALLBACK_PATTERN",
    "chroma_embedding_fn",
    "chroma_client",
    "collection",
    "get_default_collection",
    "save_session_history",
    "load_session_history",
    "conversation_history",
    "FACT_CANDIDATE_PATTERN",
    "extract_fact_from_user",
    "extract_fact_about_self",
    "memory_queue",
    "memory_stop_event",
    "memory_worker",
    "memory_thread",
    "shutdown_memory",
    "dump_memory_log",
    "dump_all_memories_to_log",
    "log_chat_interaction",
]
