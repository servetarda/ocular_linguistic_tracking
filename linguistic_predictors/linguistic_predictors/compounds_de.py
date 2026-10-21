"""Compound (and prefix) boundaries for German syllabification.

German does not resyllabify across the parts of a compound or after a
prefix: "Strom|anschluss" is [ʃtʁoːm.ʔan.ʃlʊs], not *[ʃtʁoː.man.ʃlʊs];
"Wellness|oase" keeps its [s] in "Wellness". Onset maximisation alone gets
these wrong whenever the second part starts with a vowel, which is exactly
where German inserts a glottal stop and listeners hear a syllable edge.

A word is split into dictionary words, left part optionally ending in a
linking element ("Arbeit-s-platz", "Versicherung-s-agentur"), when

1. the parts are more frequent than the whole (geometric mean of the parts'
   SUBTLEX counts > the word's count; Koehn & Knight 2003's criterion,
   standard for German compound splitting), and
2. the right part's own pronunciation is exactly the END of the word's
   pronunciation (aspiration ignored) -- so the boundary can be placed in the
   phoneme string, and splits that do not correspond to how the word is
   pronounced ("Gar|ten": "ten" is /tʰɛn/, the word ends in /tʰn̩/) are
   rejected.

Splitting recurses into both parts ("Vor|ruhe|stand"). Prefix boundaries
("ver|ein", "an|lage") come out the same way and are equally real syllable
edges. Spurious splits whose boundary coincides with a syllable edge anyway
("Dort|mund") are harmless.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, FrozenSet, List, Sequence

# German linking elements (Fugenelemente) and how each is PRONOUNCED at the
# end of the left part. A linking "s" is always a voiceless [s]; requiring the
# sound as well as the letters stops e.g. "Fern|seher" from being split as
# "Fern-s|eher" ("eher" = 'rather' is frequent, but the [z] of "Seher" cannot
# be a linking s).
LINKING_ELEMENTS = {
    "s": (("s",),), "es": (("ə", "s"),), "n": (("n",),), "en": (("ə", "n"), ("n̩",)),
    "e": (("ə",),), "er": (("ɐ",), ("ə", "ʁ")), "ens": (("ə", "n", "s"), ("n̩", "s")),
    "nen": (("n", "ə", "n"), ("n", "n̩")),
}


# Positional variants of one phoneme in MFA's german_mfa: palatal [ɟ, c] for
# /ɡ, k/ before front vowels, and vocalised [ɐ] vs consonantal [ʁ] for /r/
# ("Ort" alone is /ɔɐt/, inside "Standort" it is /ɔʁt/).
_ALLOPHONES = {"ɟ": "ɡ", "c": "k", "ɐ": "ʁ"}


def _norm(phones: Sequence[str]) -> List[str]:
    """Compare phonemes, not allophones: aspiration and palatalisation vary
    with position ("Agentur" alone vs. inside "Versicherungsagentur")."""
    out = []
    for p in phones:
        q = p.replace("ʰ", "")
        out.append(_ALLOPHONES.get(q, q))
    return out


class CompoundSplitter:
    def __init__(self, pron: Dict[str, List[str]], count: Callable[[str], float],
                 vowels: FrozenSet[str], min_part: int = 3):
        self.pron = pron
        self.count = count
        self.vowels = vowels
        self.min_part = min_part
        self._memo: Dict[str, List[int]] = {}

    def boundaries(self, word: str) -> List[int]:
        """Phoneme indices at which a morph boundary falls (sorted; empty if
        the word is not split)."""
        w = word.lower()
        if w in self._memo:
            return self._memo[w]
        self._memo[w] = []                      # guards against cycles
        P = self.pron.get(w)
        m = self.min_part
        if not P or len(w) < 2 * m:
            return []
        # numerals are compounds too: "sechs|und|fünfzig"
        if "und" in w:
            from .normalize_de import german_number
            a, _, b = w.partition("und")
            pa, pu, pb = self.pron.get(a), self.pron.get("und"), self.pron.get(b)
            if (german_number(w) is not None and pa and pu and pb
                    and _norm(P) == _norm(list(pa) + list(pu) + list(pb))):
                cuts = [len(pa), len(pa) + len(pu)] + [len(pa) + len(pu) + c for c in self.boundaries(b)]
                self._memo[w] = cuts
                return cuts
        best = None
        for i in range(m, len(w) - m + 1):
            head = w[i:]
            hp = self.pron.get(head)
            if not hp or len(hp) >= len(P):
                continue
            b = len(P) - len(hp)
            if _norm(P[b:]) != _norm(hp) or not any(p in self.vowels for p in P[:b]):
                continue
            mod = w[:i]
            base = mod if mod in self.pron else None
            if base is None:
                for link, sounds in LINKING_ELEMENTS.items():
                    if (mod.endswith(link) and len(mod) - len(link) >= m
                            and mod[:-len(link)] in self.pron
                            and any(tuple(_norm(P[b - len(snd):b])) == snd for snd in sounds)):
                        base = mod[:-len(link)]
                        break
            if base is None:
                continue
            score = math.sqrt((self.count(base) + 1.0) * (self.count(head) + 1.0))
            if best is None or score > best[0]:
                best = (score, b, base, head)
        if best is None or best[0] <= self.count(w) + 1.0:
            return []
        _, b, base, head = best
        cuts = {b} | {b + c for c in self.boundaries(head)}
        bp = self.pron[base]
        if _norm(P[:len(bp)]) == _norm(bp):
            cuts |= {c for c in self.boundaries(base) if 0 < c < b}
        out = sorted(cuts)
        self._memo[w] = out
        return out
