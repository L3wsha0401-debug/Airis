# Project: Airis Companion Assistant Upgrade

## Architecture
Airis is an AI companion assistant integrating an LLM streaming chat loop (Ollama), real-time sentence-by-sentence text-to-speech synthesis (Silero V4 RU), semantic long-term memory (ChromaDB + Ollama `bge-m3`), and short-term dialogue continuity.

```
                      +----------------------------------+
                      |        User Terminal (CLI)       |
                      +----------------------------------+
                                        │
                                        ▼
                      +----------------------------------+
                      |  Main Orchestration Loop (chat)  |
                      +----------------------------------+
                             │           │          │
         ┌───────────────────┘           │          └───────────────────┐
         ▼                               ▼                              ▼
+-----------------+            +-------------------+          +-------------------+
|  Session Turn   |            |   Memory Engine   |          |  Streaming Voice  |
|  Manager (R3)   |            |    2.0 (R2)       |          |    Engine (R1)    |
| - History RAM   |            | - ChromaDB 1024-d |          | - SentenceChunker |
| - Atomic JSON   |            | - Ollama bge-m3   |          | - Text Normalizer |
| - Signal Hooks  |            | - Fact Migration  |          | - Synth Worker    |
| - Lifecycle     |            | - chroma_lock     |          | - 48kHz Resample  |
+-----------------+            +-------------------+          | - Playback Worker |
                                                              +-------------------+
```

## Feature Inventory
Every feature from the Survey phase is mapped to an implementation milestone:
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Test Harness & Infrastructure | Multi-tier test suite (`test_airis_suite.py`) testing Tiers 1-4 | M1 | survey |
| 2 | Sentence-by-Sentence Stream Splitter | Real-time sentence segmenter splitting on `[.!?\n]` | M2 | survey |
| 3 | Text Sanitizer & Normalizer | Strips emojis, tags; expands numbers, abbreviations, latin words | M2 | survey |
| 4 | Asynchronous Synthesis Worker | Background thread consuming text sentences and invoking Silero | M2 | survey |
| 5 | Fixed 48 kHz Playback with Resampling | Software resamples 1.25x pitch audio to 48 kHz hardware stream | M2 | survey |
| 6 | Audio Queue Drain / Flush | Graceful drain of pending audio queue on turn completion / shutdown | M2 | survey |
| 7 | Multilingual Embedding Function | Ollama `bge-m3` embedding function (1,024-d) | M3 | survey |
| 8 | ChromaDB 2.0 Collection Adapter | 1,024-d collection configured with `bge-m3` and cosine space | M3 | survey |
| 9 | Legacy Memory Migration Engine | Backs up 17 facts to JSON, migrates into 1,024-d collection | M3 | survey |
| 10 | Semantic Recall with Cosine Threshold | Queries ChromaDB with `distance <= 0.35` for Russian text | M3 | survey |
| 11 | Deduplicated Fact Insertion | Verifies candidate fact distance > 0.25 before insertion | M3 | survey |
| 12 | Background User Fact Extraction | LLM JSON prompt detecting persistent user facts | M3 | survey |
| 13 | Background Self Fact Extraction | LLM JSON prompt detecting persistent self identity facts | M3 | survey |
| 14 | Core User Identity Persistence | Explicit `/name` command storing user name in `core:user_name` | M3 | survey |
| 15 | Memory Log Dump | Formats all stored facts to `Logs/iris_memory.log` on shutdown | M3 | survey |
| 16 | In-Memory Turn Manager | Sliding window of recent turns (`MAX_HISTORY=12`) | M4 | survey |
| 17 | Atomic Disk Serializer | Serializes history to `Logs/session_history.json` via `os.replace` | M4 | survey |
| 18 | Disk Deserializer & Validator | Reads and validates `Logs/session_history.json` on startup | M4 | survey |
| 19 | Dynamic System Prompt Builder | Combines base persona, user name, and semantic memories | M4 | survey |
| 20 | Process Lifecycle Signal Handler | Captures `SIGINT` (Ctrl+C), `SIGTERM`, `exit` for orderly exit | M4 | survey |
| 21 | Orderly Teardown Coordinator | Flushes TTS, stops memory thread, saves session, closes Chroma | M4 | survey |
| 22 | Multi-tier Concurrency Locks | `chroma_lock`, `ollama_generation_lock`, `session_history_lock` | M5 | survey |
| 23 | Streaming Terminal Filter | Clean display chunks to stdout while buffering hidden tags | M5 | survey |
| 24 | Final E2E Pass & Adversarial Hardening | 100% E2E test suite pass + Tier 5 adversarial stress testing | M6 | survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | E2E Test Harness & Suite | Requirements-driven test suite (`test_airis_suite.py`) covering Tiers 1-4 | none | DONE (Gate PASS) |
| M2 | Streaming TTS & Audio Safety (R1) | SentenceChunker, text normalizer, 48kHz software resampling, async playback | M1 | DONE (Gate PASS) |
| M3 | Multilingual Memory 2.0 (R2) | Ollama `bge-m3` embedding, ChromaDB 1024-d migration (17 facts), Russian recall | M1 | DONE (Gate PASS) |
| M4 | Session History & Continuity (R3) | `session_history.json` atomic serialization, turn restore, signal teardown | M1 | DONE (Gate PASS) |
| M5 | Concurrency & Lock Integrity (R4) | Three-tier lock hierarchy, non-blocking Ollama priority, queue thread safety | M2, M3, M4 | IN_PROGRESS |
| M6 | Final E2E Pass & Hardening | Pass 100% E2E test suite + Tier 5 adversarial stress testing | M1, M2, M3, M4, M5 | PLANNED |

## Interface Contracts

### VoiceEngine (TTS) ↔ Chat Loop
- `VoiceEngine.feed_sentence(sentence: str) -> None`: Enqueues sentence string for asynchronous normalization and synthesis.
- `VoiceEngine.flush() -> None`: Signals completion of response stream and waits for audio queue playback.
- `clean_and_normalize_tts_text(text: str) -> str`: Normalizes numbers, abbreviations, transliterates Latin names (`l3wsha` -> `Лёша`), filters emojis/symbols, returns empty string if no Cyrillic phonemes.
- Audio output guarantee: `sounddevice.play(audio_tensor, samplerate=48000)` strictly at 48 kHz.

### MemoryEngine ↔ Chat Loop
- `initialize_and_migrate_chroma(chroma_dir: str) -> tuple[PersistentClient, Collection]`: Migrates legacy collection to `bge-m3` (1024-d), preserving existing 17 facts.
- `recall_memory(user_text: str, entity: str = "user") -> list[str]`: Queries with `where={"entity": entity}` and `distance <= 0.35` under `chroma_lock`.
- `insert_fact(fact: str, entity: str) -> bool`: Deduplicates (`distance <= 0.25`) and inserts document with `chroma_lock`.

### SessionManager ↔ Process Lifecycle
- `load_session_history(file_path: str, max_turns: int = 12) -> list[dict]`: Deserializes and validates recent turns from disk.
- `save_session_history(file_path: str, history: list[dict]) -> None`: Atomically saves history to temp file and replaces target file.
- `graceful_shutdown() -> None`: Coordinates ordered teardown: audio queue flush, memory queue drain, session save, Chroma lock release.

## Code Layout
- `C:\LLM\Airis\Airis_V0.3.py`: Primary application source code.
- `C:\LLM\Airis\tests\test_airis_suite.py`: E2E test harness and test cases (Tiers 1-4).
- `C:\LLM\Airis\Logs\session_history.json`: Persisted short-term conversation context.
- `C:\LLM\Airis\Logs\iris_memory.log`: Memory export dump.
- `C:\LLM\Airis\chroma_db\`: Persistent vector database (ChromaDB).
- `C:\LLM\Airis\PROJECT.md`: Project index and architectural specification.
- `C:\LLM\Airis\TEST_INFRA.md`: Test harness architecture and coverage matrix.
- `C:\LLM\Airis\TEST_READY.md`: Test readiness signal and runner instructions.
