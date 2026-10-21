"""QA spot-check sheet for a native German speaker (no programming needed):
one row per sampled word with its automatic phoneme transcription,
syllabification and POS tag, the surrounding words with the target in
[brackets], and blank `flag` / `notes` columns.

The reviewer marks rows where the phonemes, syllables or POS tag look wrong
for the word as used in that sentence. Flags that cluster on one pattern
(compounds, one POS confusion, one letter) point to a fixable problem;
scattered flags are the normal error rate of automatic tools.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

import pandas as pd

from .syllabic import DE_SAMPA_VOWELS, syllable_strings


def build_qa_sample(
    words: Sequence[str],
    pos_tags: Sequence[str],
    pron_dict,
    vowel_set=DE_SAMPA_VOWELS,
    n: Optional[int] = 50,
    seed: int = 0,
    context_window: int = 3,
) -> pd.DataFrame:
    """Build the QA spot-check table.

    `words` and `pos_tags` must be the same length and in the same order as
    the `word`/`pos` columns `pipeline.LinguisticPredictorPipeline.process()`
    returns (i.e. just pass `df["word"].tolist()` and `df["pos"].tolist()`).

    `n=None` (or `n >= len(words)`) includes every word position; otherwise a
    random sample of `n` positions is drawn (seeded on `seed`, so re-running
    with the same `n`/`seed` reproduces the same sample -- useful for
    reviewing again after fixing something the first pass flagged).

    Phonemes/syllables are recomputed directly from `pron_dict` here (cheap
    for just the sampled words) rather than reusing the main pipeline's
    internal cohort-model computations, which don't carry a plain per-word
    transcription string.
    """
    if len(words) != len(pos_tags):
        raise ValueError(
            f"words and pos_tags must be the same length, got {len(words)} vs {len(pos_tags)}"
        )
    n_total = len(words)
    indices = list(range(n_total))
    if n is not None and n < n_total:
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, n))

    rows = []
    for i in indices:
        word = words[i]
        phons = pron_dict.phonemes(word.lower())
        syls = syllable_strings(phons, vowel_set) if phons else []
        lo = max(0, i - context_window)
        hi = min(n_total, i + context_window + 1)
        context = " ".join(
            w if j != i else f"[{w}]" for j, w in enumerate(words[lo:hi], start=lo)
        )
        rows.append(dict(
            position=i,
            word=word,
            phonemes=" ".join(phons) if phons else "(NO PRONUNCIATION FOUND -- flag this)",
            syllables="-".join(syls) if syls else "",
            pos=pos_tags[i],
            context=context,
            flag="",   # reviewer fills in "x" (or anything) if something looks wrong
            notes="",  # reviewer's free-text comment, optional
        ))
    return pd.DataFrame(rows)
