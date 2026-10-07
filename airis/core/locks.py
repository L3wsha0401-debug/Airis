"""
Airis Core Concurrency Locks.
Provides process-wide synchronization primitives to ensure non-blocking,
deadlock-free concurrency between foreground UI, LLM streaming,
background memory extraction, and atomic session persistence.
"""

import threading

# Lock for serializing requests to the Ollama local inference daemon
ollama_lock = threading.Lock()

# Alias for backward compatibility and R4 contract compliance
ollama_generation_lock = ollama_lock

# Mutual exclusion lock for all ChromaDB vector storage reads and writes
chroma_lock = threading.Lock()

# Mutual exclusion lock for conversation session history disk I/O and mutations
session_history_lock = threading.Lock()

__all__ = [
    "ollama_lock",
    "ollama_generation_lock",
    "chroma_lock",
    "session_history_lock",
]
