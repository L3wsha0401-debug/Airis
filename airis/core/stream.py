"""
Airis Stream Filtering and Console UI Utilities.
Provides real-time token filtering (hiding reasoning tags, stopping on end markers),
artifact post-processing, and terminal rendering.
"""

import re
from typing import Callable, Iterable, Iterator, Optional, Set
from colorama import init, Fore, Style
from airis.core.prompts import EMOTION_CATALOG, DEFAULT_EMOTION

init(autoreset=True)


def print_header():
    """Prints the welcome banner and command help in terminal."""
    print(Fore.CYAN + Style.BRIGHT + "=" * 55)
    print(Fore.MAGENTA + Style.BRIGHT + "  🌸 Айрис — Диалоговый режим (V0.4 Модульная) 🌸")
    print(Fore.CYAN + Style.BRIGHT + "=" * 55)
    print(Fore.YELLOW + "  Введите 'exit' или 'выход' для завершения диалога.\n")
    print(Fore.YELLOW + "  Команда '/name Имя' — сменить имя пользователя.\n")


def print_user_input_prompt(user_name: str) -> str:
    """Prompts the user for terminal input with color styling."""
    return input(Fore.YELLOW + Style.BRIGHT + f"{user_name} ➔ " + Fore.RESET)


def sanitize_text(text: str) -> str:
    """
    Cleans XML/pseudo-tags and trailing role identifiers from text.
    """
    # Cut off unclosed/remaining XML tags
    text = re.sub(r'</?[A-Za-z_]+>.*', '', text)
    # Cut off accidental user/assistant role suffixes (preceded by newline or with colon)
    text = re.sub(r'(?:\n\s*(?:user|assistant)\b.*$|\b(?:user|assistant)\s*:.*$)', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove unicode replacement characters (corrupted tokens)
    text = text.replace('\ufffd', '')
    return text


def clean_stream_filter(chunk_stream):
    """
    Generator filter for clean real-time console streaming.
    Hides <think>...</think>, <tools>, and breaks cleanly upon encountering stop markers.
    """
    buffer = ""
    in_hidden_block = False
    hidden_tags = [("<think>", "</think>"), ("<tools>", "</tools>"), ("<tool_call>", "</tool_call>")]
    stop_markers = [
        "<start_of_turn>", "<end_of_turn>",
        "user", "assistant",
        "L3wsha:", "Айрис:",
        "<|", "</",
        "?>", "\n\n"
    ]

    for chunk in chunk_stream:
        token = chunk['message']['content']
        buffer += token

        # Check for start of hidden thought/reasoning tags
        if not in_hidden_block:
            for open_tag, _ in hidden_tags:
                if open_tag in buffer:
                    in_hidden_block = True
                    break

        if in_hidden_block:
            closed = False
            for _, close_tag in hidden_tags:
                if close_tag in buffer:
                    buffer = buffer.split(close_tag, 1)[1]
                    in_hidden_block = False
                    closed = True
                    break
            if not closed:
                continue

        # Check for stop markers
        stop_found = False
        for marker in stop_markers:
            if marker.lower() in buffer.lower():
                idx = buffer.lower().find(marker.lower())
                safe_chunk = buffer[:idx]
                buffer = ""
                if safe_chunk:
                    cleaned = sanitize_text(safe_chunk)
                    if cleaned:
                        yield cleaned
                stop_found = True
                break

        if stop_found:
            break

        # Flush safe portion of buffer (keep trailing 12 chars in case of partial tag)
        if len(buffer) > 12:
            safe_part = buffer[:-12]
            buffer = buffer[-12:]
            cleaned_safe = sanitize_text(safe_part)
            if cleaned_safe:
                yield cleaned_safe

    # Flush remainder of buffer
    if not in_hidden_block and buffer:
        remaining = re.sub(r'<(think|tools|tool_call)>.*', '', buffer, flags=re.DOTALL | re.IGNORECASE)
        for marker in stop_markers:
            if marker.lower() in remaining.lower():
                idx = remaining.lower().find(marker.lower())
                remaining = remaining[:idx]
        remaining = sanitize_text(remaining)
        if remaining:
            yield remaining


def clean_artifacts(text: str) -> str:
    """
    Final post-processing of AI response text before saving to history.
    """
    # 1. Truncate at system role markers or fragments
    text = re.split(
        r'(</?tool_call>|<\/?think>|<\/?tools>|<start_of_turn>|<end_of_turn>|<\||(?:\n\s*(?:user|assistant|L3wsha:|Айрис:)\b)|(?:\b(?:user|assistant|L3wsha:|Айрис:):))',
        text,
        flags=re.IGNORECASE
    )[0]

    # 2. Clean XML/service tags
    text = re.sub(r'<(tools|schema|think)>.*?(</\1>|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?(tools|schema|think|xml|tool_call)[^>]*>', '', text, flags=re.IGNORECASE)

    # 3. Sanitize characters
    text = sanitize_text(text)

    # 4. Defense-in-depth: strip trailing cookie/sweet emojis and sign-off clutter
    text = re.sub(r'[\s\u2000-\u200f\U0001F36A\U0001F36B\U0001F36C\U0001F36D]+$', '', text)

    return text.strip()


EMOTION_BADGES = {
    "neutral": {"emoji": "😐", "color": Fore.CYAN, "label": "neutral"},
    "happy": {"emoji": "😊", "color": Fore.GREEN + Style.BRIGHT, "label": "happy"},
    "smug": {"emoji": "😏", "color": Fore.MAGENTA + Style.BRIGHT, "label": "smug"},
    "surprised": {"emoji": "😲", "color": Fore.YELLOW + Style.BRIGHT, "label": "surprised"},
    "angry": {"emoji": "💢", "color": Fore.RED + Style.BRIGHT, "label": "angry"},
    "blush": {"emoji": "😳", "color": Fore.LIGHTMAGENTA_EX + Style.BRIGHT, "label": "blush"},
    "bored": {"emoji": "🥱", "color": Fore.WHITE, "label": "bored"},
    "thinking": {"emoji": "🤔", "color": Fore.BLUE + Style.BRIGHT, "label": "thinking"},
}


def format_emotion_badge(emotion: str) -> str:
    """Renders formatted ANSI color badge for console display."""
    key = str(emotion).strip().lower()
    meta = EMOTION_BADGES.get(key, EMOTION_BADGES[DEFAULT_EMOTION])
    return f"{meta['color']}[{meta['emoji']} {meta['label']}]{Style.RESET_ALL} "


class EmotionStreamParser:
    """
    Streaming parser that intercepts the initial tokens of an LLM generation,
    extracts the leading emotion tag ([neutral], [happy], etc.),
    triggers an on_emotion callback, and yields text chunks stripped of the leading tag.
    """
    def __init__(
        self,
        valid_emotions: Optional[Iterable[str]] = None,
        default_emotion: str = DEFAULT_EMOTION,
        on_emotion: Optional[Callable[[str], None]] = None,
        max_tag_length: int = 35,
    ):
        if valid_emotions is not None:
            self.valid_emotions = {str(e).strip().lower() for e in valid_emotions}
        else:
            self.valid_emotions = {e.lower() for e in EMOTION_CATALOG}
        self.default_emotion = default_emotion
        self.on_emotion = on_emotion
        self.max_tag_length = max_tag_length
        self.detected_emotion: Optional[str] = None
        self._tag_resolved: bool = False
        self._buffer: str = ""
        self._strip_leading_space: bool = False

    def reset(self) -> None:
        """Resets the parser state for reuse across turns."""
        self.detected_emotion = None
        self._tag_resolved = False
        self._buffer = ""
        self._strip_leading_space = False

    def _resolve(self, emotion: str) -> None:
        if not self._tag_resolved:
            self.detected_emotion = emotion
            self._tag_resolved = True
            if self.on_emotion:
                try:
                    self.on_emotion(emotion)
                except Exception:
                    pass

    def process_stream(self, chunk_generator: Iterable[str]) -> Iterator[str]:
        for chunk in chunk_generator:
            if not chunk:
                continue

            if self._tag_resolved:
                if self._strip_leading_space:
                    chunk = chunk.lstrip()
                    if not chunk:
                        continue
                    self._strip_leading_space = False
                yield chunk
                continue

            self._buffer += chunk
            stripped_leading = self._buffer.lstrip()

            # Case 1: First non-whitespace character is NOT '[' -> No tag present
            if stripped_leading and not stripped_leading.startswith('['):
                self._resolve(self.default_emotion)
                to_yield = self._buffer
                self._buffer = ""
                yield to_yield
                continue

            # Case 2: Buffer starts with '[' -> Look for closing ']'
            if stripped_leading.startswith('['):
                bracket_end = stripped_leading.find(']')
                if bracket_end != -1:
                    raw_tag = stripped_leading[1:bracket_end].strip().lower()
                    emotion = raw_tag if raw_tag in self.valid_emotions else self.default_emotion
                    self._resolve(emotion)

                    # Text following the closing bracket
                    remainder = stripped_leading[bracket_end + 1:]
                    clean_remainder = remainder.lstrip()
                    self._buffer = ""
                    if clean_remainder:
                        self._strip_leading_space = False
                        yield clean_remainder
                    else:
                        self._strip_leading_space = True
                    continue

                # Bracket not yet closed: check length guard
                if len(self._buffer) > self.max_tag_length:
                    # Exceeded maximum tag length without closing bracket
                    self._resolve(self.default_emotion)
                    to_yield = self._buffer
                    self._buffer = ""
                    yield to_yield
                    continue

        # Stream exhausted
        if not self._tag_resolved:
            self._resolve(self.default_emotion)
            if self._buffer:
                to_yield = self._buffer
                self._buffer = ""
                yield to_yield

    def __call__(self, chunk_generator: Iterable[str]) -> Iterator[str]:
        return self.process_stream(chunk_generator)


__all__ = [
    "print_header",
    "print_user_input_prompt",
    "sanitize_text",
    "clean_stream_filter",
    "clean_artifacts",
    "EMOTION_BADGES",
    "format_emotion_badge",
    "EmotionStreamParser",
]

