"""
Airis Stream Filtering and Console UI Utilities.
Provides real-time token filtering (hiding reasoning tags, stopping on end markers),
artifact post-processing, and terminal rendering.
"""

import re
from colorama import init, Fore, Style

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
    # Cut off accidental user/assistant role suffixes
    text = re.sub(r'\b(user|assistant)\b.*$', '', text, flags=re.IGNORECASE)
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
        r'(</?tool_call>|<\/?think>|<\/?tools>|<start_of_turn>|<end_of_turn>|<\||\.?\s*(?:user|assistant|L3wsha:|Айрис:))',
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
