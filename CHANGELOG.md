# Changelog / История изменений

## [v0.5] — 2026-10-31

### English
- **STT model upgraded to `large-v3-turbo` (`int8`, CPU)**:
  - Default Whisper model changed from `small` to `large-v3-turbo` (`int8`, 6 CPU threads, `beam_size: 1`, `initial_prompt: "Айрис, Лёша."`) for significantly better Russian inflection, proper noun, and sentence accuracy without GPU VRAM contention.
- **Breath, cough & hallucination suppression (`airis/stt/engine.py`)**:
  - Enabled built-in Silero VAD (`vad_filter=True`, `min_speech_duration_ms: 250`, `min_silence_duration_ms: 300`) and `condition_on_previous_text=False`.
  - Added segment pruning by duration (`< 150` ms), confidence (`no_speech_prob` / `avg_logprob`), and a 40+ phrase Russian hallucination/cough/interjection filter (`HALLUCINATION_PHRASES` + `LAUGHTER_NOISE_PATTERN`).
- **Push-to-Talk post-roll buffering (`airis/stt/ptt.py`)**:
  - Added a configurable `ptt_post_roll_ms` window (`250` ms by default) after releasing the PTT hotkey so trailing word endings are never clipped; re-pressing the key within the post-roll window seamlessly continues the same recording.
- **CPU thread balancing (`airis/tts/engine.py`)**:
  - Bounded Silero TTS PyTorch threads (`torch.set_num_threads(4)`) to prevent CPU contention with CTranslate2 STT.
- **Context-aware memory & session summaries (`airis/memory/`, `airis/core/prompts.py`)**:
  - Added single-call LLM fact conflict resolution (`insert` / `replace` / `delete` / `ignore`), contextual recall using the previous assistant reply, and automatic Russian session summary generation on shutdown (`last_session_summary`).

### Русский
- **Переход STT на модель `large-v3-turbo` (`int8`, CPU)**:
  - Модель распознавания речи по умолчанию заменена с `small` на `large-v3-turbo` (`int8`, 6 потоков CPU, `beam_size: 1`, `initial_prompt: "Айрис, Лёша."`) — исправлены окончания русских слов, имена собственные и пропуск слов в длинных фразах без нагрузки на VRAM видеокарты.
- **Подавление вдохов, кашля и галлюцинаций Whisper (`airis/stt/engine.py`)**:
  - Включён встроенный Silero VAD (`vad_filter=True`, `min_speech_duration_ms: 250`, `min_silence_duration_ms: 300`) и `condition_on_previous_text=False`.
  - Добавлена отбраковка сегментов по длительности (`< 150` мс), порогам уверенности (`no_speech_prob` / `avg_logprob`) и списку из 40+ русских галлюцинаций, междометий и кашля/смеха (`HALLUCINATION_PHRASES` + `LAUGHTER_NOISE_PATTERN`).
- **Post-roll буферизация для Push-to-Talk (`airis/stt/ptt.py`)**:
  - Добавлено окно дозаписи `ptt_post_roll_ms` (по умолчанию `250` мс) после отпускания клавиши PTT, чтобы не съедались концы слов; повторное нажатие клавиши во время post-roll бесшовно продолжает текущую запись.
- **Балансировка потоков CPU (`airis/tts/engine.py`)**:
  - Ограничено число потоков PyTorch для Silero TTS (`torch.set_num_threads(4)`), чтобы синтез и распознавание речи не конкурировали за ядра процессора.
- **Контекстная память и саммари сессий (`airis/memory/`, `airis/core/prompts.py`)**:
  - Добавлено разрешение конфликтов фактов в памяти за один вызов LLM (`insert` / `replace` / `delete` / `ignore`), поиск по памяти с учётом предыдущей реплики ассистента и автоматическое сохранение краткого саммари диалога при выходе (`last_session_summary`).

---

## [v0.4] — 2026-10-11
- Added Live2D avatar integration via VTube Studio WebSocket API (`airis/avatar/`) with real-time RMS audio lip-sync and emotion hotkey triggers.
- Added autonomous idle chatter coordinator (`airis/core/idle.py`).
- Moved personal system and idle prompts out of source code into local `system_prompt.txt` / `idle_prompt.txt` (gitignored) with `system_prompt.example.txt`.
