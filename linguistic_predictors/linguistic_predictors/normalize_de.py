"""Spelling normalisation for German stimulus texts, applied to what the MODELS
read -- never to the `word` column that output rows are matched on.

Swiss Standard German writes "ss" for every "ß" ("äusserst", "sass",
"geniessen", "regelmässig"). A listener hears no difference, but German
GPT-2, the pronunciation dictionary, SUBTLEX-DE and the embeddings were all
built from Germany-standard text, where these forms are rare or absent:
"äusserst" is not in SUBTLEX-DE at all (0 vs 489 counts for "äußerst"), so
every level would treat an ordinary word as a rarity. The fix is decided
per word by frequency, not by rule, because "ss" is also correct standard
spelling after short vowels ("dass", "musste", "Fitnessstudio"):

    replace "ss" -> "ß" in a word only if that variant is at least
    `min_ratio` times more frequent than the word as written, and attested
    at least `min_count` times, in SUBTLEX-DE (modern orthography:
    "weiß" 57,465 vs "weiss" 0; but "dass" 141,710 vs "daß" 0, "Masse"
    237 vs "Maße" 0 -- so those stay). `wordfreq` cannot arbitrate: it
    case-folds "ß" to "ss", so both spellings get the same number.

The number of words is unchanged (only letters inside a word change), so
normalised and original units stay aligned one-to-one.
"""
from __future__ import annotations

import itertools
import re
from typing import Callable, List, Optional, Tuple

_WORD_RE = re.compile(r"[^\W\d_]+")


def _default_frequency() -> Callable[[str], float]:
    from .frequency_de import SubtlexDE
    return SubtlexDE().count


def _ss_variants(word: str) -> List[str]:
    """Every way of turning some lowercase "ss" occurrences into "ß"."""
    idx = [m.start() for m in re.finditer("ss", word)]
    out = []
    for k in range(1, min(len(idx), 3) + 1):
        for combo in itertools.combinations(idx, k):
            if any(b - a < 2 for a, b in zip(combo, combo[1:])):   # overlapping "sss"
                continue
            w = word
            for pos in sorted(combo, reverse=True):
                w = w[:pos] + "ß" + w[pos + 2:]
            out.append(w)
    return out


def standardize_swiss_spelling(text: str, frequency: Optional[Callable[[str], float]] = None,
                               min_ratio: float = 5.0, min_count: float = 3.0
                               ) -> Tuple[str, List[Tuple[str, str]]]:
    """Return (normalised text, [(original, replacement), ...]). `frequency`
    maps a word to a corpus count (default: SUBTLEX-DE, case-insensitive)."""
    freq = frequency or _default_frequency()
    changes: List[Tuple[str, str]] = []

    def fix(m):
        w = m.group(0)
        if "ss" not in w:
            return w
        f0 = freq(w)
        best, fb = w, f0
        for v in _ss_variants(w):
            fv = freq(v)
            if fv > fb:
                best, fb = v, fv
        if best != w and fb >= min_count and fb >= min_ratio * f0:
            changes.append((w, best))
            return best
        return w

    return _WORD_RE.sub(fix, text), changes


# --------------------------------------------------------------------------- #
# Spoken-word equivalence: what a listener hears as the same word            #
# --------------------------------------------------------------------------- #
_UNITS = {"eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7,
          "acht": 8, "neun": 9}
_UNIT_PREFIX = {"ein": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7,
                "acht": 8, "neun": 9}
_TEENS = {"zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15,
          "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19}
_TENS = {"zwanzig": 20, "dreißig": 30, "vierzig": 40, "fünfzig": 50, "sechzig": 60,
         "siebzig": 70, "achtzig": 80, "neunzig": 90}


def _below_100(w: str) -> Optional[int]:
    if w in _UNITS:
        return _UNITS[w]
    if w in _TEENS:
        return _TEENS[w]
    if w in _TENS:
        return _TENS[w]
    if "und" in w:                                      # sechsundfünfzig
        a, _, b = w.partition("und")
        if a in _UNIT_PREFIX and b in _TENS:
            return _UNIT_PREFIX[a] + _TENS[b]
    return None


def _below_1000(w: str) -> Optional[int]:
    if "hundert" in w:
        a, _, b = w.partition("hundert")
        h = 1 if a in ("", "ein") else _UNIT_PREFIX.get(a)
        rest = 0 if b == "" else _below_100(b)
        return None if h is None or rest is None else 100 * h + rest
    return _below_100(w)


def german_number(word: str) -> Optional[int]:
    """Value of a German cardinal written as ONE word, 0 < n < 10000
    ("hundert" -> 100, "zweihundert" -> 200, "sechsundfünfzig" -> 56);
    None otherwise. "ein"/"eine"/"einen" are articles, not numerals, and
    are never parsed."""
    w = word.lower().replace("ss", "ß") if "dreiss" in word.lower() else word.lower()
    if w in ("ein", "eine", "einen", "einem", "einer", "eines"):
        return None
    if "tausend" in w:
        a, _, b = w.partition("tausend")
        t = 1 if a in ("", "ein") else _below_1000(a)
        rest = 0 if b == "" else _below_1000(b)
        return None if t is None or rest is None or t >= 10 else 1000 * t + rest
    return _below_1000(w)


def spoken_variants(word: str) -> List[str]:
    """Spellings a LISTENER cannot tell apart from `word`: its casings, and
    for a numeral the digit string ("hundert" / "Hundert" / "100"). German
    GPT-2 mostly saw numbers as digits, so P(" hundert") alone badly
    understates how expected the spoken word "hundert" is."""
    out = [word]
    for v in (word.lower(), word[:1].upper() + word[1:].lower() if word else word):
        if v and v not in out:
            out.append(v)
    n = german_number(word)
    if n is not None and str(n) not in out:
        out.append(str(n))
    return out


def standard_spellings(key: str, frequency: Callable[[str], float],
                       min_ratio: float = 5.0, min_count: float = 3.0) -> List[str]:
    """Standard-orthography spellings of a dictionary key written with "ss"
    ("weiss" -> ["weiß"]), by the same frequency rule as
    `standardize_swiss_spelling`; [] if the "ss" spelling is standard
    ("dass", "muss")."""
    if "ss" not in key:
        return []
    f0 = frequency(key)
    best = max(_ss_variants(key), key=frequency, default=None)
    if best is not None and frequency(best) >= min_count and frequency(best) >= min_ratio * f0:
        return [best]
    return []
