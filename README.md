# Airis

**English** | [Русский](README.ru.md)

A local voice AI companion. Runs fully offline: LLM via Ollama, long-term memory in ChromaDB, streaming speech synthesis (Silero TTS) and speech recognition (faster-whisper).

## Features

- **LLM chat** via [Ollama](https://ollama.com) (`gemma2:9b` by default) with streaming output to the terminal.
- **Streaming TTS** — speech is synthesized sentence by sentence while the model is still generating. Uses Silero with the `baya` voice, resampled to 48 kHz.
- **Voice input (STT)** — faster-whisper, push-to-talk mode (`space` key), VAD and barge-in (you can interrupt the assistant).
- **Long-term memory** — facts about the user and about Airis itself are extracted from the conversation automatically and stored in ChromaDB. Embeddings use `bge-m3` (1024-d) with multilingual support, cosine similarity search and fact deduplication.
- **Sessions** — conversation history (last 12 turns) is saved atomically to disk and restored on startup.
- **Graceful shutdown** — on `Ctrl+C`, `SIGTERM` or the exit command, the TTS queue is flushed and the session and memory are saved.

## Requirements

- Windows (the project was developed and tested on Windows)
- Python 3.10+
- [Ollama](https://ollama.com) with the models pulled:
  ```bash
  ollama pull gemma2:9b
  ollama pull bge-m3
  ```
- Silero TTS model (`model.pt`), with its path set in `config.yaml`
- A microphone and an audio output device

## Installation

```bash
git clone https://github.com/L3wsha0401-debug/Airis.git
cd Airis

python -m venv venv
venv\Scripts\activate

pip install torch numpy pyyaml colorama sounddevice ollama chromadb faster-whisper
```

> The dependency list is derived from the imports in the code. For reproducible installs, pin versions with `pip freeze > requirements.txt`.

## Configuration

All settings live in [`config.yaml`](config.yaml):

| Section | What it controls |
|---------|------------------|
| `ollama` | server URL, model, embedding model, generation parameters |
| `tts` | path to `model.pt`, voice, pitch shift, sample rate, mute |
| `stt` | Whisper model, device, input mode (`ptt`), hotkey, VAD |
| `memory` | ChromaDB directory, distance threshold, deduplication, history size |
| `logging` | log directory and files |
| `personality` | default user name, emoji limit |

**You must change** `tts.model_path` — the config contains the author's local path (`C:\LLM\Instruments\model.pt`).

## Usage

Make sure Ollama is running, then:

```bash
python run.py
```

To quit, use the exit command in the chat or press `Ctrl+C`.

## Chat commands

| Command | Action |
|---------|--------|
| `/help` | list commands |
| `/mute` | toggle speech output |
| `/memories` | show stored facts |
| `/forget` | delete a fact from memory |
| `/name <name>` | remember the user's name |

## Project structure

```
airis/
├── config.py        # loading and validation of config.yaml
├── core/            # commands, lifecycle, locks, prompts, streaming output
├── memory/          # ChromaDB, fact extraction, session history, memory logs
├── stt/             # recording, VAD, push-to-talk, barge-in, Whisper
└── tts/             # sentence chunking, text normalization, synthesis, resampling
run.py               # entry point
config.yaml          # configuration
PROJECT.md           # detailed architecture and milestones
LICENSE              # MIT license
```

## Data that is not tracked in the repository

These files are created at runtime and listed in `.gitignore`:

- `Model/` — model weights (`*.gguf`, `*.pt`, `*.onnx`)
- `chroma_db/` — vector memory database
- `Logs/` — logs, session history, memory dump
- `*.session` — Telegram sessions (private data)

## License

MIT, see [LICENSE](LICENSE).