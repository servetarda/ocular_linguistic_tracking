"""Long-format tables for mTRF: one row per phoneme / per syllable.

`words.csv` holds each word's phoneme and syllable values as lists; these
functions explode them into one row per unit, with the phoneme or syllable
itself attached (looked up the same way the pipeline did). Onset times are
added by `textgrid_align.py`.
"""
from __future__ import annotations

import pandas as pd

from .syllabic import DE_SAMPA_VOWELS, syllable_strings


def flatten_phoneme_level(df: pd.DataFrame, pron_dict) -> pd.DataFrame:
    """Explode `pipeline.process()`'s per-word phoneme lists into one row
    per (word, phoneme-position-within-word).

    `pron_dict` is used only to attach the actual phoneme SYMBOL at each
    position (e.g. "d", "i:") -- purely for readability/QA -- by
    recomputing `pron_dict.phonemes(word)` per word (cheap: this is a
    dictionary lookup, not a model call). This is always the same
    transcription the pipeline itself used internally, since both call the
    same `pron_dict.phonemes()`.

    Words with no pronunciation at all (`phon_surprisal_hier == []`, e.g. an
    out-of-vocabulary word your dictionary/G2P fallback couldn't resolve --
    see `qa_export.summarize_qa_coverage`) contribute no rows here; check
    coverage separately rather than assuming every input word appears.
    """
    rows = []
    for _, r in df.iterrows():
        surprisal_hier = r["phon_surprisal_hier"]
        if not surprisal_hier:
            continue
        entropy_hier = r["phon_entropy_hier"]
        surprisal_freq = r["phon_surprisal_freq"]
        entropy_freq = r["phon_entropy_freq"]
        phons = pron_dict.phonemes(str(r.get("word_model", r["word"])).lower())
        n = len(surprisal_hier)
        for i in range(n):
            rows.append(dict(
                word_position=r["position"],
                word=r["word"],
                phon_index=i,
                n_phonemes=n,
                phoneme=phons[i] if i < len(phons) else None,
                phon_surprisal_hier=surprisal_hier[i],
                phon_entropy_hier=entropy_hier[i],
                phon_surprisal_freq=surprisal_freq[i],
                phon_entropy_freq=entropy_freq[i],
            ))
    return pd.DataFrame(rows)


def flatten_syllable_level(df: pd.DataFrame, pron_dict, vowel_set=DE_SAMPA_VOWELS) -> pd.DataFrame:
    """Same as `flatten_phoneme_level`, at the syllable level. `syllable`
    is the syllable's own phoneme string (e.g. "d-i:"), recomputed from
    `pron_dict` + `syllabic.syllable_strings` the same way the pipeline did
    internally."""
    rows = []
    for _, r in df.iterrows():
        surprisal_hier = r["syll_surprisal_hier"]
        if not surprisal_hier:
            continue
        entropy_hier = r["syll_entropy_hier"]
        surprisal_freq = r["syll_surprisal_freq"]
        entropy_freq = r["syll_entropy_freq"]
        phons = pron_dict.phonemes(str(r.get("word_model", r["word"])).lower())
        syls = syllable_strings(phons, vowel_set) if phons else []
        n = len(surprisal_hier)
        for i in range(n):
            rows.append(dict(
                word_position=r["position"],
                word=r["word"],
                syll_index=i,
                n_syllables=n,
                syllable=syls[i] if i < len(syls) else None,
                syll_surprisal_hier=surprisal_hier[i],
                syll_entropy_hier=entropy_hier[i],
                syll_surprisal_freq=surprisal_freq[i],
                syll_entropy_freq=entropy_freq[i],
            ))
    return pd.DataFrame(rows)
