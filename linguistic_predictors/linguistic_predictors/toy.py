"""Tiny offline stand-ins for every model and resource, so the whole pipeline
can run (and be tested) with no downloads: `run_story.py --backend toy`.

None of the numbers are linguistically meaningful. The language model is a
Laplace-smoothed bigram model fit on 20 hand-written German sentences; the
pronunciations are approximate X-SAMPA; the POS tags are a lookup; the
"embeddings" are deterministic pseudo-random vectors.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .lexical import LanguageModel
from .phonemic import FrequencyPrior
from .syllabic import DE_SAMPA_VOWELS

TOY_CORPUS_DE = [
    "die katze saß auf der matte .",
    "der hund rannte zu dem park .",
    "ein großer hund rannte schnell unter dem baum .",
    "eine kleine katze saß langsam auf der matte .",
    "der junge sah einen vogel .",
    "das mädchen sah einen mann .",
    "er sah das haus .",
    "sie sah den baum .",
    "der vogel flog über das haus .",
    "der mann war glücklich .",
    "die frau war traurig .",
    "er schrieb eine geschichte .",
    "sie las das buch .",
    "der junge las ein buch .",
    "das mädchen schrieb eine geschichte .",
    "es war ein großes haus .",
    "sein hund rannte zu dem park .",
    "ihre katze saß auf der matte .",
    "der hund sah die katze .",
    "die katze sah den hund .",
]

# approximate X-SAMPA; long vowels with ":", diphthongs aI/aU/OY as one symbol
TOY_PHONEMES_DE: Dict[str, List[str]] = {
    "die": ["d", "i:"], "der": ["d", "e:", "6"], "das": ["d", "a", "s"],
    "dem": ["d", "e:", "m"], "den": ["d", "e:", "n"],
    "ein": ["aI", "n"], "eine": ["aI", "n", "@"], "einen": ["aI", "n", "@", "n"],
    "katze": ["k", "a", "ts", "@"], "hund": ["h", "U", "n", "t"],
    "matte": ["m", "a", "t", "@"], "park": ["p", "a:", "6", "k"],
    "baum": ["b", "aU", "m"], "vogel": ["f", "o:", "g", "@", "l"],
    "mädchen": ["m", "E:", "t", "C", "@", "n"], "mann": ["m", "a", "n"],
    "frau": ["f", "R", "aU"], "buch": ["b", "u:", "x"],
    "geschichte": ["g", "@", "S", "I", "C", "t", "@"], "haus": ["h", "aU", "s"],
    "saß": ["z", "a:", "s"], "auf": ["aU", "f"], "rannte": ["R", "a", "n", "t", "@"],
    "zu": ["ts", "u:"], "großer": ["g", "R", "o:", "s", "6"], "großes": ["g", "R", "o:", "s", "@", "s"],
    "schnell": ["S", "n", "E", "l"], "unter": ["U", "n", "t", "6"],
    "kleine": ["k", "l", "aI", "n", "@"], "langsam": ["l", "a", "N", "z", "a:", "m"],
    "junge": ["j", "U", "N", "@"], "sah": ["z", "a:"],
    "er": ["e:", "6"], "sie": ["z", "i:"], "es": ["E", "s"],
    "flog": ["f", "l", "o:", "k"], "über": ["y:", "b", "6"], "war": ["v", "a:", "6"],
    "glücklich": ["g", "l", "Y", "k", "l", "I", "C"],
    "traurig": ["t", "R", "aU", "R", "I", "C"], "schrieb": ["S", "R", "i:", "p"],
    "las": ["l", "a:", "s"], "sein": ["z", "aI", "n"], "ihre": ["i:", "R", "@"],
    "und": ["U", "n", "t"], "aber": ["a:", "b", "6"], "mit": ["m", "I", "t"],
}

# Universal Dependencies tags, as spaCy's `.pos_`
TOY_POS_DE: Dict[str, str] = {
    "die": "DET", "der": "DET", "das": "DET", "dem": "DET", "den": "DET",
    "ein": "DET", "eine": "DET", "einen": "DET", "sein": "DET", "ihre": "DET",
    "katze": "NOUN", "hund": "NOUN", "matte": "NOUN", "park": "NOUN",
    "baum": "NOUN", "vogel": "NOUN", "mädchen": "NOUN", "mann": "NOUN",
    "frau": "NOUN", "buch": "NOUN", "geschichte": "NOUN", "haus": "NOUN", "junge": "NOUN",
    "saß": "VERB", "rannte": "VERB", "sah": "VERB", "flog": "VERB",
    "war": "VERB", "schrieb": "VERB", "las": "VERB",
    "großer": "ADJ", "großes": "ADJ", "kleine": "ADJ", "glücklich": "ADJ", "traurig": "ADJ",
    "schnell": "ADV", "langsam": "ADV",
    "auf": "ADP", "zu": "ADP", "unter": "ADP", "über": "ADP", "mit": "ADP",
    "und": "CCONJ", "aber": "CCONJ",
    "er": "PRON", "sie": "PRON", "es": "PRON",
    ".": "PUNCT", ",": "PUNCT",
}


class ToyLanguageModel(LanguageModel):
    """Laplace-smoothed bigram model: P(next | prev) = (count + 1) / (count(prev) + V).
    Uncased (all words lowercased)."""

    def __init__(self, corpus: List[str] = TOY_CORPUS_DE):
        bigrams: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        starts: Dict[str, int] = defaultdict(int)
        seen = set()
        for sent in corpus:
            words = [w.lower() for w in sent.split()]
            seen.update(words)
            starts[words[0]] += 1
            for w1, w2 in zip(words[:-1], words[1:]):
                bigrams[w1][w2] += 1
        self.bigrams, self.starts, self.vocab = bigrams, starts, sorted(seen)

    def next_word_distribution(self, context_words) -> Dict[str, float]:
        counts = self.bigrams.get(context_words[-1].lower(), {}) if context_words else self.starts
        total = sum(counts.values()) + len(self.vocab)
        return {w: (counts.get(w, 0) + 1) / total for w in self.vocab}

    def word_logprob(self, context_words, word: str) -> float:
        return float(np.log(self.next_word_distribution(context_words).get(word.lower(), 1e-12)))


class ToyPronDict:
    """`phonemic.PronouncingDictionary` over a word -> phonemes dict."""

    def __init__(self, phonemes: Dict[str, List[str]] = TOY_PHONEMES_DE):
        self._phonemes = phonemes

    def phonemes(self, word: str) -> List[str]:
        return list(self._phonemes.get(word.lower(), []))

    def vocabulary(self) -> List[str]:
        return sorted(self._phonemes)


class ToyPOSTagger:
    """Context-independent tag lookup (`syntactic.POSTagger`)."""

    def __init__(self, tags: Dict[str, str] = TOY_POS_DE):
        self._tags = tags

    def tag_word_in_context(self, before, word, after) -> str:
        return self._tags.get(word.lower(), "X")


class ToyEmbeddings:
    """Deterministic pseudo-random unit vectors (md5-seeded), no semantics."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self._cache: Dict[str, np.ndarray] = {}

    def vector(self, word: str):
        word = word.lower()
        if not word[:1].isalpha():
            return None
        if word not in self._cache:
            rng = np.random.RandomState(int(hashlib.md5(word.encode()).hexdigest()[:8], 16))
            v = rng.normal(size=self.dim)
            self._cache[word] = v / np.linalg.norm(v)
        return self._cache[word]


def toy_frequency_prior(corpus: List[str] = TOY_CORPUS_DE) -> FrequencyPrior:
    counts: Dict[str, int] = defaultdict(int)
    for sent in corpus:
        for w in sent.split():
            if w[:1].isalpha():
                counts[w.lower()] += 1
    return FrequencyPrior(dict(counts))


def toy_components() -> dict:
    """The same component set `components_de.load_components` returns, offline."""
    return dict(lm=ToyLanguageModel(), pron_dict=ToyPronDict(), freq_prior=toy_frequency_prior(),
                pos_tagger=ToyPOSTagger(), embeddings=ToyEmbeddings(), vowel_set=DE_SAMPA_VOWELS,
                subtlex=None)
