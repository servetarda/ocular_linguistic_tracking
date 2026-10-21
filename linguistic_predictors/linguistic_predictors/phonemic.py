"""Phonemic level: pronunciations and the frequency prior.

The cohort computation itself is in `cohort.py`; the hierarchical prior (the
language model's probability of every dictionary word) comes from
`candidates.py`. A `PronouncingDictionary` maps a word to its phoneme list
(the MFA `german_mfa` dictionary in real runs, see `lexicon_mfa.py`).
"""
from __future__ import annotations

from typing import Dict, List, Protocol, Sequence


class PronouncingDictionary(Protocol):
    def phonemes(self, word: str) -> List[str]:
        """Phoneme sequence of `word`, or [] if unknown."""
        ...

    def vocabulary(self) -> Sequence[str]:
        """Every word the dictionary can pronounce (the candidate vocabulary)."""
        ...


class FrequencyPrior:
    """word -> P(word) under a fixed unigram frequency model: the prior of
    the single-level cohort (SUBTLEX-DE counts in real runs)."""

    def __init__(self, freq_by_word: Dict[str, float]):
        total = sum(freq_by_word.values())
        self.prob_by_word = {w: f / total for w, f in freq_by_word.items()} if total > 0 else {}

    def distribution(self) -> Dict[str, float]:
        return dict(self.prob_by_word)


def phoneme_units(words: Sequence[str], pron_dict: PronouncingDictionary) -> Dict[str, List[str]]:
    """{word: phonemes} for every word the dictionary can pronounce."""
    out = {}
    for w in words:
        phons = pron_dict.phonemes(w)
        if phons:
            out[w] = phons
    return out
