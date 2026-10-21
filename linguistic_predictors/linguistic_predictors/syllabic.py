"""Syllabic level (an extension -- not in Heilbron et al. 2022): syllabification.

Syllables are built from the dictionary pronunciations and fed to the same
cohort model as phonemes (`cohort.py`), one unit per syllable. Rules:

- Nuclei are the vowel phones of the dictionary (`lexicon_mfa.infer_vowel_phones`),
  including syllabic consonants. The r-glide "ɐ" after a monophthong belongs
  to that vowel ("werden" = veːɐ.dən); after a diphthong it is a nucleus
  ("Feuer" = fɔʏ.ɐ).
- Consonants between nuclei go to the next syllable as far as they form a
  legal onset -- a cluster that begins at least 10 dictionary words, judged
  on phonemes not allophones, loanword-only onsets (/s/+C, /ps/, /ks/)
  excluded ("Garten" = ɡaʁ.tʰn̩, "Fenster" = fɛns.tɐ).
- Compound and prefix boundaries (`compounds_de.py`) are hard syllable
  edges ("Stromanschluss" = ʃtʁoːm.an.ʃlʊs).

The onset inventory and compound boundaries travel inside the vowel set
(`PhoneInventory`), so every caller that passes a vowel set gets them.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Sequence

# X-SAMPA vowels, for the toy resources (the real dictionary is IPA and its
# vowels are inferred from the data).
DE_SAMPA_VOWELS: FrozenSet[str] = frozenset({
    "a", "a:", "E", "E:", "e:", "I", "i:", "O", "o:", "U", "u:",
    "Y", "y:", "2", "2:", "@", "6", "aI", "aU", "OY",
})


# German centering off-glide: MFA writes coda /r/ after a vowel as a plain
# "ɐ" (e.g. "werden" /v eː ɐ d ə n/). After a MONOPHTHONG it is part of that
# vowel, NOT a syllable of its own -- "werden" is 2 syllables ("veːɐ.dən"),
# "vier" /f iː ɐ/ is 1. After a DIPHTHONG it is a real nucleus: "Feuer"
# /f ɔʏ ɐ/, "Bauer" /b aw ɐ/, "Eier" /aj ɐ/ are all 2 syllables. And with
# consonants before it, it is always a nucleus ("hundert" /h ʊ n d ɐ t/).
_OFFGLIDES_AFTER_VOWEL: FrozenSet[str] = frozenset({"ɐ", "ɐ̯"})

# Diphthongs as written by MFA's german_mfa (IPA) and in X-SAMPA. An "ɐ" after
# one of these is a syllable of its own, not an off-glide (see above).
_DIPHTHONGS: FrozenSet[str] = frozenset({"aj", "aw", "ɔʏ", "aɪ", "aʊ", "ɔɪ", "aI", "aU", "OY"})


class PhoneInventory(frozenset):
    """A vowel set that can also carry the language's LEGAL syllable onsets.

    It IS a frozenset of nucleus phones, so every function that takes a
    `vowel_set` keeps working unchanged; the extra `legal_onsets` attribute
    (a set of consonant tuples attested word-initially in the lexicon, e.g.
    ("ʃ", "t", "ʁ")) switches `syllabify` from plain maximal onset to
    maximal onset *subject to phonotactic legality*. `MFAPronDict.vowel_phones`
    returns one of these, so the German pipeline gets legal onsets
    automatically wherever the vowel set is threaded through.

    `morph_boundaries` ({phoneme tuple: boundary indices}, see
    `compounds_de.py`) adds compound/prefix boundaries that syllables never
    straddle ("Strom|anschluss" = ʃtʁoːm.an.ʃlʊs, not ʃtʁoː.man.ʃlʊs).
    """

    def __new__(cls, vowels=(), legal_onsets=None, morph_boundaries=None):
        obj = super().__new__(cls, vowels)
        obj.legal_onsets = frozenset(legal_onsets) if legal_onsets is not None else None
        obj.morph_boundaries = dict(morph_boundaries) if morph_boundaries else None
        return obj

    def __or__(self, other):
        return PhoneInventory(frozenset(self) | frozenset(other), self.legal_onsets, self.morph_boundaries)

    def __reduce__(self):
        return (PhoneInventory, (frozenset(self), self.legal_onsets, self.morph_boundaries))


# Word-initial clusters that are loanword-only in German and are NOT used as
# onsets word-internally: /s/ + consonant ("Ski", "Stil", "Sphäre" -- native
# German has /ʃt/, /ʃp/) and obstruent + /s/ ("Psyche", "Xylophon").
# Word-internally German syllabifies "Mus.kel", "Fens.ter", "Kas.per".
_LOAN_ONSET_STARTS = ("s",)
_LOAN_ONSETS = frozenset({("p", "s"), ("k", "s")})


def _onset_key(cluster: Sequence[str]) -> tuple:
    """Onset legality ignores aspiration: MFA writes word-initial /t/ as
    [tʰ], but an intervocalic [t] is the same phoneme and just as legal an
    onset ("Fens.ter")."""
    return tuple({"ɟ": "ɡ", "c": "k"}.get(q, q) for q in (p.replace("ʰ", "") for p in cluster))


def attested_onsets(prons: Sequence[Sequence[str]], vowel_set: FrozenSet[str],
                    min_count: int = 10) -> FrozenSet[tuple]:
    """Consonant clusters that begin a word (before its first nucleus) in at
    least `min_count` lexicon entries -- the standard operationalisation of a
    'legal onset'. The threshold keeps out transcription noise and one-off
    loan clusters (a German lexicon attests e.g. ʃtʁ 719x, pfl 136x, but also
    junk like 'ʃpʁŋ' 2x). Loanword-only clusters (/s/+C, /ps/, /ks/) are
    excluded even when frequent. The empty onset is always legal."""
    counts: Dict[tuple, int] = {}
    for p in prons:
        k = 0
        while k < len(p) and p[k] not in vowel_set:
            k += 1
        if 0 < k < len(p):
            on = _onset_key(p[:k])
            counts[on] = counts.get(on, 0) + 1
    return frozenset(on for on, n in counts.items()
                     if n >= min_count and not (len(on) > 1 and on[0] in _LOAN_ONSET_STARTS)
                     and on not in _LOAN_ONSETS) | {()}


def _nucleus_indices(phonemes: Sequence[str], vowel_set) -> List[int]:
    """Indices of syllable nuclei. An off-glide "ɐ" directly after a
    monophthong is NOT a nucleus (it belongs to that vowel); after a
    diphthong it is."""
    out: List[int] = []
    for i, p in enumerate(phonemes):
        if p not in vowel_set:
            continue
        if (p in _OFFGLIDES_AFTER_VOWEL and out and out[-1] == i - 1
                and phonemes[i - 1] not in _DIPHTHONGS):
            continue
        out.append(i)
    return out


def syllabify(phonemes: Sequence[str], vowel_set: FrozenSet[str] = DE_SAMPA_VOWELS) -> List[List[str]]:
    """Split a phoneme sequence into syllables by the maximal onset principle.

    Between two nuclei, the consonants go to the FOLLOWING syllable's onset
    -- as many as form a legal onset if `vowel_set` carries `legal_onsets`
    (see `PhoneInventory`), otherwise all of them (plain maximal onset, for
    vowel sets without onset information, e.g. the toy resources). The rest
    stay behind as the previous syllable's coda. Legality matters: plain
    maximal onset gives "Garten" = ɡa.ʁtʰn̩ and "hundert" = hʊ.ndɐt; with legal
    onsets these are ɡaʁ.tʰn̩ and hʊn.dɐt. Consonants after the last nucleus
    are the final syllable's coda.

    A non-nucleus off-glide "ɐ" (after a monophthong) always stays with its
    vowel: "werden" = veːɐ.dən, not veː.ɐdən.

    Compound/prefix boundaries carried by the vowel set (`PhoneInventory.
    morph_boundaries`) are hard syllable edges: each part is syllabified on
    its own (a part without a vowel, e.g. a stray linking consonant, joins
    the part before it).
    """
    cuts = (getattr(vowel_set, "morph_boundaries", None) or {}).get(tuple(phonemes))
    if cuts:
        out: List[List[str]] = []
        edges = [0] + [c for c in cuts if 0 < c < len(phonemes)] + [len(phonemes)]
        for a, b in zip(edges, edges[1:]):
            part = list(phonemes[a:b])
            if out and not _nucleus_indices(part, vowel_set):
                out[-1].extend(part)
            else:
                out.extend(_syllabify_part(part, vowel_set))
        return out
    return _syllabify_part(phonemes, vowel_set)


def _syllabify_part(phonemes: Sequence[str], vowel_set: FrozenSet[str]) -> List[List[str]]:
    nucleus_idx = _nucleus_indices(phonemes, vowel_set)
    if not nucleus_idx:
        # no vowel found (e.g. abbreviations); treat whole word as one syllable
        return [list(phonemes)] if phonemes else []

    legal = getattr(vowel_set, "legal_onsets", None)
    boundaries = [0]
    for a, b in zip(nucleus_idx, nucleus_idx[1:]):
        lo = a + 1
        # an off-glide right after nucleus `a` is part of that nucleus
        while lo < b and phonemes[lo] in _OFFGLIDES_AFTER_VOWEL and phonemes[lo] in vowel_set:
            lo += 1
        cut = lo  # plain maximal onset: every consonant to the next syllable
        if legal is not None:
            # longest suffix of the cluster phonemes[lo:b] that is a legal onset
            cut = b
            for s in range(lo, b + 1):
                if _onset_key(phonemes[s:b]) in legal:
                    cut = s
                    break
        boundaries.append(cut)
    boundaries.append(len(phonemes))
    return [list(phonemes[boundaries[i]:boundaries[i + 1]]) for i in range(len(boundaries) - 1)]


def syllable_strings(phonemes: Sequence[str], vowel_set: FrozenSet[str] = DE_SAMPA_VOWELS) -> List[str]:
    """Syllabify then represent each syllable as a single hashable token
    (its phonemes joined by '-'), suitable for use as a cohort "unit"."""
    return ["-".join(syl) for syl in syllabify(phonemes, vowel_set) if syl]


def syllable_units(phon_units: Dict[str, Sequence[str]], vowel_set: FrozenSet[str]) -> Dict[str, List[str]]:
    """{word: syllable strings} for {word: phonemes}."""
    return {w: syllable_strings(p, vowel_set) for w, p in phon_units.items()}
