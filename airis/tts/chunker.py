"""
Airis Streaming Sentence Chunker.
Buffers streamed LLM tokens and segments complete sentences for real-time TTS synthesis.
"""

import re
from typing import List


class SentenceChunker:
    """
    Streaming sentence segmenter buffering partial tokens and emitting complete
    sentences on [.!?] and newlines, protecting decimals and common abbreviations.
    """
    def __init__(self):
        self._buffer = ""

    def feed(self, chunk: str) -> List[str]:
        if not chunk:
            return []
        self._buffer += chunk
        sentences = []

        while True:
            match = None
            m_obj = None
            for m in re.finditer(r'([.!?]+(?:\s+|\Z)|[\n\r]+)', self._buffer):
                end_pos = m.end()
                punct = m.group(1)
                start_punct = m.start()

                # Protect decimal numbers (e.g. 3.14 or 2.5)
                if 0 < start_punct < len(self._buffer) - 1:
                    prev_char = self._buffer[start_punct - 1]
                    next_char = self._buffer[start_punct + 1] if start_punct + 1 < len(self._buffer) else ""
                    if prev_char.isdigit() and (next_char.isdigit() or (punct.startswith(".") and next_char.isdigit())):
                        continue

                # Protect if buffer ends in a digit followed by period (e.g. 'Версия 3.')
                if start_punct > 0 and self._buffer[start_punct - 1].isdigit() and end_pos == len(self._buffer):
                    break

                # Protect common abbreviations (и т.д., т.е., руб.)
                prefix = self._buffer[:end_pos]
                if re.search(r'(?:\bи\s+т\.д|\bт\.е|\bруб)\.$', prefix.strip()):
                    rest = self._buffer[end_pos:].lstrip()
                    if rest:
                        if not rest[0].isupper():
                            continue
                    else:
                        # Abbreviation at the very end of buffer, wait for more tokens
                        break

                match = (m.start(), end_pos)
                m_obj = m
                break

            if match and m_obj:
                sent = self._buffer[:match[0] + len(m_obj.group(1).rstrip())].strip()
                self._buffer = self._buffer[match[1]:]
                if sent:
                    sentences.append(sent)
            else:
                break

        return sentences

    def flush(self) -> List[str]:
        rem = self._buffer.strip()
        self._buffer = ""
        return [rem] if rem else []

    def reset(self) -> None:
        self._buffer = ""
