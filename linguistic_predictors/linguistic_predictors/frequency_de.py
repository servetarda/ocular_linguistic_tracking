"""German word frequencies from SUBTLEX-DE (Brysbaert, Buchmeier, Conrad,
Jacobs, Bölte & Böhl, 2011, Experimental Psychology 58, 412-424) -- film
subtitles, ~25M words.

Why SUBTLEX rather than `wordfreq`
----------------------------------
Heilbron et al. (2022) took both frequency measures from SUBTLEX: the
unigram-surprisal confound regressor ("-log P(word) based on its frequency
of occurrence in subtlex") and the prior of the single-level phoneme cohort
(SI A.4: "a statistical phonetic dictionary ... that combined frequency
information from SUBTLEX database and pronunciation from the CMU
dictionary"). Subtitle frequencies are also the better model of SPOKEN
German -- the stimuli here are heard -- and SUBTLEX-DE predicts German word
recognition times better than written-corpus norms (Brysbaert et al. 2011).
`wordfreq` blends Wikipedia, news, web and subtitles, i.e. mostly written text.

Unseen words
------------
Counts are Laplace-smoothed, P(w) = (count + 1) / (N + V), the same
pseudocount-of-1 rule the paper applies in its cohort model. A compound
SUBTLEX never saw ("Versicherungsagentur") therefore gets the surprisal of a
word seen less than once in 25M words (~24.6 bits) -- an estimate, not a
missing value -- and `in_corpus()` lets an analysis flag or exclude those.

Data: CC BY-NC-SA 4.0, https://osf.io/py9ba/ ("SUBTLEX-DE cleaned version
with Zipf values.xlsx"). `run_story.py` looks for it under
`data/subtlex_de/`; a TSV cache is written next to the .xlsx on first load.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional

from . import config
from .phonemic import FrequencyPrior

_HERE = Path(__file__).resolve().parent.parent
DEFAULT_LOCATIONS = (
    _HERE / "data" / "subtlex_de" / "subtlex_de.tsv",
    _HERE / "data" / "subtlex_de" / "SUBTLEX-DE_cleaned_with_Zipf.xlsx",
)


def find_subtlex_de(path: Optional[str] = None) -> Optional[Path]:
    cands = [Path(path).expanduser()] if path else list(DEFAULT_LOCATIONS)
    for c in cands:
        if c.exists():
            return c
    return None


class SubtlexDE:
    """Case-insensitive SUBTLEX-DE counts (`CUMfreqcount`: all casings of a
    word summed, e.g. "essen" + "Essen")."""

    def __init__(self, path: Optional[str] = None):
        import pandas as pd
        src = find_subtlex_de(path)
        if src is None:
            raise FileNotFoundError(
                "SUBTLEX-DE not found. Download 'SUBTLEX-DE cleaned version with Zipf values.xlsx' "
                "from https://osf.io/py9ba/ into data/subtlex_de/ (or pass its path).")
        if src.suffix == ".xlsx":
            df = pd.read_excel(src)
            cache = src.with_name("subtlex_de.tsv")
            df[["Word", "WFfreqcount", "CUMfreqcount", "SUBTLEX"]].to_csv(cache, sep="\t", index=False)
        else:
            df = pd.read_csv(src, sep="\t", keep_default_na=False, na_values=[""])
        words = df["Word"].astype(str).str.lower()
        self.counts: Dict[str, int] = dict(zip(words, df["CUMfreqcount"].astype(int)))
        # corpus size implied by the per-million column (~25.0M tokens)
        per_m = df["SUBTLEX"].astype(float)
        ok = per_m > 0
        self.n_tokens = float((df["CUMfreqcount"][ok] / per_m[ok] * 1e6).median())
        self.n_types = len(self.counts)
        self.source = str(src)

    def count(self, word: str) -> int:
        return self.counts.get(word.lower(), 0)

    def count_any_spelling(self, word: str) -> int:
        """Count under the word's standard spelling: for a dictionary key in
        "ss" orthography ("weiss", "strasse") the most frequent of its ß
        variants ("weiß", "straße"). SUBTLEX-DE uses modern orthography."""
        c = self.count(word)
        if "ss" in word:
            from .normalize_de import _ss_variants
            c = max([c] + [self.count(v) for v in _ss_variants(word.lower())])
        return c

    def in_corpus(self, word: str) -> bool:
        return word.lower() in self.counts

    def prob(self, word: str) -> float:
        """Laplace-smoothed P(word)."""
        return (self.count(word) + 1.0) / (self.n_tokens + self.n_types)

    def surprisal(self, word: str) -> float:
        """-log P(word) in `config.LOG_BASE` units (unigram surprisal)."""
        return config.surprisal_from_logprob(math.log(self.prob(word)))

    def zipf(self, word: str) -> float:
        """Zipf scale (log10 frequency per billion words), Laplace-smoothed."""
        return math.log10(self.prob(word) * 1e9)

    def frequency_prior(self, vocabulary: Iterable[str]) -> FrequencyPrior:
        """Prior for the single-level phoneme/syllable cohort: every
        dictionary word with its SUBTLEX count + 1 (the paper's pseudocount),
        so words SUBTLEX never saw still compete instead of vanishing."""
        return FrequencyPrior({w.lower(): self.count_any_spelling(w) + 1.0 for w in vocabulary})
