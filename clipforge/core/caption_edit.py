"""Manual caption editing (Edit step).

Pure functions over clip-relative :class:`Word` lists: edit a caption
line's text, delete a word, delete a whole line. Times are preserved —
an edited line keeps its original time span (new words split the span
evenly), so the render needs no re-timing.
"""

from __future__ import annotations

from .captions import group_words
from .transcription import Word

__all__ = [
    "delete_line",
    "delete_word",
    "edit_line_text",
    "line_spans",
    "line_texts",
]


def line_texts(words: list[Word]) -> list[str]:
    """Caption line texts for a word list (same grouping as the render)."""
    return [text for _start, _end, text in line_spans(words)]


def line_spans(words: list[Word]) -> list[tuple[float, float, str]]:
    """Caption lines as (start, end, text) spans, clip-relative.

    Same grouping as the render (:func:`group_words`), so the Edit
    step's corrected-captions list shows exactly what will burn in.
    """
    return [
        (line[0].start, line[-1].end, " ".join(w.text for w in line))
        for line in group_words(words)
    ]


def edit_line_text(words: list[Word], line_index: int, new_text: str) -> list[Word]:
    """Replace one caption line's text, keeping its time span.

    The new words split the line's original span evenly (at least 0.05 s
    per word). An empty ``new_text`` deletes the line.
    """
    lines = group_words(words)
    if line_index < 0 or line_index >= len(lines):
        raise IndexError(f"no caption line {line_index}")
    line = lines[line_index]
    new_words = new_text.split()
    if not new_words:
        return delete_line(words, line_index)
    start = line[0].start
    end = max(line[-1].end, start + 0.05 * len(new_words))
    span = end - start
    step = span / len(new_words)
    rebuilt = [
        Word(text=text, start=start + i * step, end=start + (i + 1) * step)
        for i, text in enumerate(new_words)
    ]
    out: list[Word] = []
    for i, group in enumerate(lines):
        out.extend(rebuilt if i == line_index else group)
    return out


def delete_word(words: list[Word], word_index: int) -> list[Word]:
    """Delete one word (flat index across all lines)."""
    if word_index < 0 or word_index >= len(words):
        raise IndexError(f"no word {word_index}")
    return [w for i, w in enumerate(words) if i != word_index]


def delete_line(words: list[Word], line_index: int) -> list[Word]:
    """Delete a whole caption line."""
    lines = group_words(words)
    if line_index < 0 or line_index >= len(lines):
        raise IndexError(f"no caption line {line_index}")
    doomed = set(map(id, lines[line_index]))
    return [w for w in words if id(w) not in doomed]
