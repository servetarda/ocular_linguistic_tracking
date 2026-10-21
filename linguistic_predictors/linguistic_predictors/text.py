"""Text units and where they sit in the original text.

A unit is a maximal run of letters (a word) or a single punctuation mark;
each unit becomes one row of output. Digits, quotes and dashes are not units
but stay in the text the language model reads. Keeping the original text and
each unit's character span is what lets GPT-2 read the text as written
(punctuation attached, "überraschend." not "überraschend .").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

UNIT_RE = re.compile(r"[^\W\d_]+|[.,!?;:]")
PUNCT_UNITS = frozenset(".,!?;:")
SENTENCE_END = frozenset(".!?")


def is_word_unit(unit: str) -> bool:
    return bool(unit) and unit[0].isalpha()


def normalize_text(text: str) -> str:
    """Whitespace clean-up that does not change what a reader sees: CRLF ->
    LF, runs of spaces/tabs -> one space, no spaces around line breaks, at
    most one blank line in a row, no leading/trailing whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detokenize(words: Sequence[str]) -> str:
    """Natural text from a unit list: one space between words, punctuation
    attached to the word before it."""
    out: List[str] = []
    for w in words:
        if out and w not in PUNCT_UNITS:
            out.append(" ")
        out.append(w)
    return "".join(out)


@dataclass
class TextLayout:
    text: str                        # the text the language model reads
    words: List[str]                 # units: letter runs and single punctuation marks
    spans: List[Tuple[int, int]]     # character span of each unit in `text`

    @classmethod
    def from_text(cls, text: str) -> "TextLayout":
        text = normalize_text(text)
        ms = list(UNIT_RE.finditer(text))
        return cls(text, [m.group() for m in ms], [m.span() for m in ms])

    @classmethod
    def from_words(cls, words: Sequence[str]) -> "TextLayout":
        parts: List[str] = []
        spans: List[Tuple[int, int]] = []
        pos = 0
        for w in words:
            if parts and w not in PUNCT_UNITS:
                parts.append(" ")
                pos += 1
            spans.append((pos, pos + len(w)))
            parts.append(w)
            pos += len(w)
        return cls("".join(parts), list(words), spans)


def sentence_ids(words: Sequence[str]) -> List[int]:
    """Sentence index of each unit; a sentence-final mark belongs to the
    sentence it closes."""
    ids: List[int] = []
    s = 0
    for w in words:
        ids.append(s)
        if w and w[-1] in SENTENCE_END:
            s += 1
    return ids


def sentence_bounds(ids: Sequence[int], i: int) -> Tuple[int, int]:
    """[lo, hi) unit range of the sentence containing unit i."""
    lo = i
    while lo > 0 and ids[lo - 1] == ids[i]:
        lo -= 1
    hi = i + 1
    while hi < len(ids) and ids[hi] == ids[i]:
        hi += 1
    return lo, hi
