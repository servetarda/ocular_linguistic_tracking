"""Syntactic level: the predicted part of speech of the next word.

Heilbron et al. (2022, SI A.3): every candidate in the nucleus of the
next-word distribution is POS-tagged in "the context plus the predicted
word" (the paper: "POS tagging on every potential sentence"); the
probability of a tag is the probability mass of the candidates carrying it,
and POS surprisal is -log P(tag of the presented word) (SI Eq. 6).

Nothing here sees the words that follow. The presented word is tagged
exactly like a candidate -- left context plus the word -- so the tag it is
scored on is the one a listener could assign at that moment, and an
ambiguous word is resolved by the preceding context only (infinitival "zu"
in "ein Bad zu" is tagged as a preposition: the verb has not been heard
yet). Its full-sentence tag is returned separately as the word's label
(content vs function word); it never enters the prediction.
"""
from __future__ import annotations

from typing import Dict, List, Protocol, Tuple

from . import config


class POSTagger(Protocol):
    def tag_word_in_context(self, before: List[str], word: str, after: List[str]) -> str:
        """Universal Dependencies tag of `word` between `before` and `after`."""
        ...

    # Optional, for speed: tag_words_batch(candidates, before, after) -> {word: tag}


def pos_surprisal_entropy(
    before: List[str],
    actual: str,
    after: List[str],
    candidates: Dict[str, float],
    tagger: POSTagger,
    actual_p: float = 0.0,
) -> Tuple[float, float, str]:
    """(POS surprisal, POS entropy, full-sentence POS label of the presented word).

    `candidates`: {candidate word: probability} -- the nucleus of the
    next-word distribution, as predicted (normalised here). POS entropy is
    the entropy of the tag distribution over this nucleus. For the
    surprisal, the presented word is added with `actual_p` if it is not in
    the nucleus, so that P(its tag) >= P(the word) always holds.
    """
    label = tagger.tag_word_in_context(before, actual, after)
    total = sum(candidates.values())
    if total <= 0:
        return float("nan"), float("nan"), label
    words = list(candidates) + ([actual] if actual not in candidates else [])
    batch = getattr(tagger, "tag_words_batch", None)
    if batch is not None:
        tags = batch(words, before, [])
    else:
        tags = {w: tagger.tag_word_in_context(before, w, []) for w in words}
    dist: Dict[str, float] = {}
    for w, p in candidates.items():
        dist[tags[w]] = dist.get(tags[w], 0.0) + p / total
    entropy = config.entropy(dist.values())
    if actual not in candidates:
        extra = actual_p / total
        dist = {t: p / (1.0 + extra) for t, p in dist.items()}
        dist[tags[actual]] = dist.get(tags[actual], 0.0) + extra / (1.0 + extra)
    return config.surprisal(dist.get(tags[actual], 0.0)), entropy, label
