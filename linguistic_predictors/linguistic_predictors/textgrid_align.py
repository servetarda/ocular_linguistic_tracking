"""Merge forced-alignment times (a Praat TextGrid, e.g. from `mfa align`)
onto the per-phoneme and per-syllable tables from `mtrf_export.py`.

The pipeline itself only sees text; onset/offset times come from aligning
the audio with MFA and passing the TextGrid to `run_story.py --textgrid`.

Matching is positional: the n-th spoken word of the text (punctuation
skipped) is the n-th non-empty word interval of the TextGrid, and each
word's phonemes must be the SAME phones, in the same order, as its aligned
phone intervals -- equal counts are not enough, since a different
pronunciation of equal length would otherwise pass. Phone symbols are
compared after `normalize_phone` (MFA's `german_mfa` dictionary and acoustic
model share one phone set, so no further equivalences are assumed; pass
`equivalences` for a justified exception). Syllables are timed by cutting
the aligned phones exactly where the pipeline's own syllables (the ones
its predictors were computed on) end. Any mismatch raises by default (`strict=True`), because a silent shift would
corrupt every time point downstream; `strict=False` warns and drops the
mismatched word instead. Only Praat's long text format is supported (what
MFA writes).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .syllabic import DE_SAMPA_VOWELS

Interval = Tuple[float, float, str]

_PUNCT_ONLY_RE = re.compile(r"^[.,!?;:]+$")


def normalize_phone(phone: str, equivalences: Optional[Dict[str, str]] = None) -> str:
    """A phone symbol for comparison: Unicode-normalised (NFC), stripped,
    then mapped through `equivalences` (aligner symbol -> dictionary symbol)."""
    p = unicodedata.normalize("NFC", str(phone)).strip()
    return (equivalences or {}).get(p, p)


def _phones_differ(dict_phones, aligned_phones, equivalences) -> bool:
    return ([normalize_phone(p, equivalences) for p in dict_phones]
            != [normalize_phone(p, equivalences) for p in aligned_phones])

# Marks a forced aligner writes for things that are not speech sounds: MFA
# uses "sil"/"sp"/"spn"/"<eps>", BAS WebMAUS "<p:>". They occupy intervals on
# the phone tier but are not phonemes and must not be counted as such.
NON_SPEECH_LABELS = frozenset({
    "sil", "sp", "spn", "nsn", "<eps>", "<p:>", "<usb>", "<nib>", "",
})

_INTERVAL_RE = re.compile(
    r'intervals\s*\[\d+\]:\s*\n'
    r'\s*xmin\s*=\s*([\d.eE+-]+)\s*\n'
    r'\s*xmax\s*=\s*([\d.eE+-]+)\s*\n'
    r'\s*text\s*=\s*"((?:[^"]|"")*)"',
)
_TIER_NAME_RE = re.compile(r'name\s*=\s*"(.*?)"')


def parse_textgrid(path: str) -> Dict[str, List[Interval]]:
    """Parse a Praat long-text-format TextGrid into {tier_name: [(xmin, xmax, text), ...]}.
    Only IntervalTiers are read (point tiers, if any, are ignored -- MFA's
    word/phone alignment output only uses interval tiers)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if "item [" not in text:
        raise ValueError(
            f"{path!r} doesn't look like a Praat 'long text format' TextGrid (no "
            "'item [n]:' blocks found). This parser only supports that format "
            "(Praat/MFA's default). If your file is 'short text format', re-save it "
            "from Praat as 'Text file' (not 'Short text file')."
        )

    chunks = re.split(r"\n\s*item \[\d+\]:\s*\n", text)[1:]
    tiers: Dict[str, List[Interval]] = {}
    for chunk in chunks:
        name_m = _TIER_NAME_RE.search(chunk)
        if not name_m:
            continue
        name = name_m.group(1)
        intervals = [
            (float(m.group(1)), float(m.group(2)), m.group(3).replace('""', '"'))
            for m in _INTERVAL_RE.finditer(chunk)
        ]
        tiers[name] = intervals
    if not tiers:
        raise ValueError(f"{path!r} parsed with no named tiers found -- check the file isn't truncated/corrupt.")
    return tiers


def _find_tier(tiers: Dict[str, List[Interval]], keyword: str, explicit_name: Optional[str]) -> str:
    if explicit_name is not None:
        if explicit_name not in tiers:
            raise ValueError(f"tier {explicit_name!r} not found; available tiers: {list(tiers)}")
        return explicit_name
    matches = [name for name in tiers if keyword in name.lower()]
    if not matches:
        raise ValueError(
            f"no tier name containing {keyword!r} found; available tiers: {list(tiers)} "
            f"-- pass the exact name explicitly."
        )
    if len(matches) > 1:
        raise ValueError(f"multiple tiers contain {keyword!r}: {matches} -- pass the exact name explicitly.")
    return matches[0]


def group_phones_by_word(
    tiers: Dict[str, List[Interval]],
    word_tier: Optional[str] = None,
    phone_tier: Optional[str] = None,
) -> List[dict]:
    """Group the phone tier's intervals under each non-empty word-tier
    interval (by time containment), in time order. Returns one dict per
    spoken word:
    `{"word": text, "xmin": ..., "xmax": ..., "phones": [(xmin, xmax, text), ...]}`.

    Non-speech intervals are dropped from BOTH tiers. On the word tier those
    are empty/whitespace marks; on the phone tier they are the aligner's
    silence and noise labels (`NON_SPEECH_LABELS`). This matters: an aligner
    can place a short pause or a noise token INSIDE a word's interval, and
    counting it as a phoneme would make the phone count disagree with the
    dictionary's transcription -- which, with the strict checks downstream,
    throws that word's timings away for no reason.
    """
    word_tier_name = _find_tier(tiers, "word", word_tier)
    phone_tier_name = _find_tier(tiers, "phone", phone_tier)
    words = tiers[word_tier_name]
    phones = tiers[phone_tier_name]

    groups = []
    eps = 1e-6
    for xmin, xmax, wtext in words:
        if not wtext.strip() or wtext.strip().lower() in NON_SPEECH_LABELS:
            continue
        in_word = [
            (pxmin, pxmax, ptext) for pxmin, pxmax, ptext in phones
            if ptext.strip() and ptext.strip().lower() not in NON_SPEECH_LABELS
            and pxmin >= xmin - eps and pxmax <= xmax + eps
        ]
        groups.append({"word": wtext.strip(), "xmin": xmin, "xmax": xmax, "phones": in_word})
    return groups


def _spoken_words(flat_df: pd.DataFrame) -> pd.DataFrame:
    words_df = flat_df[["word_position", "word"]].drop_duplicates().sort_values("word_position")
    return words_df[~words_df["word"].astype(str).str.match(_PUNCT_ONLY_RE)].reset_index(drop=True)


def merge_onsets_into_phoneme_table(
    flat_phon_df: pd.DataFrame,
    word_phone_groups: List[dict],
    strict: bool = True,
    equivalences: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Merge onset/offset times from `group_phones_by_word()` onto
    `mtrf_export.flatten_phoneme_level()`'s output.

    Matches words POSITIONALLY (see module docstring for why this matters):
    the i-th spoken (non-punctuation) word in `flat_phon_df` is matched to
    the i-th entry of `word_phone_groups`. Per word, the pipeline's phoneme
    count must equal the number of aligned phone intervals for that word
    (true whenever MFA aligned against the SAME pronunciation dictionary
    this pipeline used) -- a mismatch means the two are out of sync and
    onsets for that word cannot be trusted.

    With `strict=True` (default), any word-count, word-text, or
    phoneme-count mismatch raises immediately. With `strict=False`, mismatches
    are printed as warnings and that word's rows are dropped from the
    output rather than guessed at.
    """
    spoken = _spoken_words(flat_phon_df)
    if len(spoken) != len(word_phone_groups):
        msg = (
            f"word count mismatch: the processed text has {len(spoken)} spoken "
            f"(non-punctuation) word(s), the TextGrid has {len(word_phone_groups)} aligned "
            "word interval(s). These must match 1:1 -- check you ran forced alignment on "
            "exactly the same words, in the same order, as the text given to run_story.py."
        )
        if strict:
            raise ValueError(msg)
        print(f"[textgrid_align] WARNING: {msg}\nProceeding with the shorter of the two.")

    n = min(len(spoken), len(word_phone_groups))
    out_rows = []
    for i in range(n):
        pos = spoken.iloc[i]["word_position"]
        pipeline_word = str(spoken.iloc[i]["word"])
        group = word_phone_groups[i]

        if pipeline_word.lower() != group["word"].lower():
            msg = (
                f"word text mismatch at spoken-word index {i} (text position {pos}): "
                f"pipeline word {pipeline_word!r} vs TextGrid word {group['word']!r}. "
                "Matching is positional, not by text, so this strongly suggests the two "
                "have drifted out of sync (different tokenization, or an extra/missing "
                "word earlier) -- fix before trusting any onset in this output."
            )
            if strict:
                raise ValueError(msg)
            print(f"[textgrid_align] WARNING: {msg}")

        sub = flat_phon_df[flat_phon_df["word_position"] == pos].sort_values("phon_index")
        phones = group["phones"]
        if len(sub) != len(phones):
            msg = (
                f"phoneme count mismatch for word {pipeline_word!r} (text position {pos}): "
                f"pipeline dictionary gives {len(sub)} phoneme(s), TextGrid aligned "
                f"{len(phones)} phone interval(s). This word's rows will be dropped rather "
                "than guessed at."
            )
            print(f"[textgrid_align] WARNING: {msg}")
            if strict:
                raise ValueError(msg)
            continue
        if _phones_differ(sub["phoneme"], [p[2] for p in phones], equivalences):
            msg = (
                f"pronunciation mismatch for word {pipeline_word!r} (text position {pos}): "
                f"pipeline {' '.join(map(str, sub['phoneme']))} vs TextGrid "
                f"{' '.join(p[2] for p in phones)}. The aligner used a different pronunciation "
                "than the predictors were computed for; this word's rows will be dropped."
            )
            print(f"[textgrid_align] WARNING: {msg}")
            if strict:
                raise ValueError(msg)
            continue

        for (_, row), (pxmin, pxmax, ptext) in zip(sub.iterrows(), phones):
            r = row.to_dict()
            r["onset"] = pxmin
            r["offset"] = pxmax
            r["duration"] = pxmax - pxmin
            r["aligned_phone"] = ptext
            out_rows.append(r)

    if not out_rows:
        return pd.DataFrame(columns=list(flat_phon_df.columns) + ["onset", "offset", "duration", "aligned_phone"])
    return pd.DataFrame(out_rows).sort_values("onset").reset_index(drop=True)


def merge_onsets_into_syllable_table(
    flat_syll_df: pd.DataFrame,
    word_phone_groups: List[dict],
    vowel_set=DE_SAMPA_VOWELS,
    strict: bool = True,
    equivalences: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Same idea as `merge_onsets_into_phoneme_table`, at the syllable
    level: a syllable's onset is its first phone's onset, its offset is its
    last phone's offset. The aligned phones are cut where the pipeline's own
    syllables end (the `syllable` column, e.g. "k-a", "ts-@" -- the
    syllables the predictors were computed on), and the phones of each
    syllable must be the aligned phones themselves, so a syllable's time
    always belongs to the syllable its value describes. `vowel_set` is
    unused (kept for call compatibility).
    """
    spoken = _spoken_words(flat_syll_df)
    if len(spoken) != len(word_phone_groups):
        msg = (
            f"word count mismatch: {len(spoken)} spoken word(s) vs {len(word_phone_groups)} "
            "aligned TextGrid word(s) -- see merge_onsets_into_phoneme_table's docstring."
        )
        if strict:
            raise ValueError(msg)
        print(f"[textgrid_align] WARNING: {msg}\nProceeding with the shorter of the two.")

    n = min(len(spoken), len(word_phone_groups))
    out_rows = []
    for i in range(n):
        pos = spoken.iloc[i]["word_position"]
        pipeline_word = str(spoken.iloc[i]["word"])
        phones = word_phone_groups[i]["phones"]
        sub = flat_syll_df[flat_syll_df["word_position"] == pos].sort_values("syll_index")
        syl_groups = [str(sy).split("-") for sy in sub["syllable"]]
        flat = [p for g in syl_groups for p in g]
        if len(flat) != len(phones) or _phones_differ(flat, [p[2] for p in phones], equivalences):
            msg = (
                f"syllable/phone mismatch for word {pipeline_word!r} (text position {pos}): "
                f"pipeline syllables {'.'.join(map(str, sub['syllable']))} vs TextGrid phones "
                f"{' '.join(p[2] for p in phones)}. This word's rows will be dropped."
            )
            print(f"[textgrid_align] WARNING: {msg}")
            if strict:
                raise ValueError(msg)
            continue

        idx = 0
        for (_, row), syl in zip(sub.iterrows(), syl_groups):
            these_phones = phones[idx: idx + len(syl)]
            idx += len(syl)
            r = row.to_dict()
            r["onset"] = these_phones[0][0]
            r["offset"] = these_phones[-1][1]
            r["duration"] = r["offset"] - r["onset"]
            out_rows.append(r)

    if not out_rows:
        return pd.DataFrame(columns=list(flat_syll_df.columns) + ["onset", "offset", "duration"])
    return pd.DataFrame(out_rows).sort_values("onset").reset_index(drop=True)
