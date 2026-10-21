"""The candidate distribution: P(form | context) for every candidate word at
every position of a text. One distribution feeds the syntactic, semantic,
phonemic and syllabic levels.

Candidates are every dictionary word (both casings, and the standard "ß"
spelling where the dictionary writes "ss"), the text's own spellings, digit
spellings of numerals, and punctuation. All spellings of one lexical item
are summed into its probability; the item key is the pronunciation
dictionary's spelling. `VocabularyScorer` does this exactly and fast for a
GPT-2 model (see `vocab_scorer.py`); `GenericVocabularyScorer` is the slow
reference path for any `LanguageModel` (the toy model).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import config
from .normalize_de import german_number, spoken_variants
from .text import SENTENCE_END, is_word_unit
from .vocab_scorer import FormTrie

PUNCT_FORMS = (".", ",", "!", "?", ";", ":")


def _sentence_starts(layout, idl) -> List[int]:
    """Token index where each sentence after the first begins."""
    return [idl.last[i] + 1 for i, w in enumerate(layout.words) if w in SENTENCE_END and idl.last[i] >= 0]


class PositionResult:
    """Everything the five levels need at one text position.

    - lexical: `logprob` (natural log, end-of-word corrected if enabled),
      `logprob_raw` (plain token product, as in Heilbron et al.),
      `entropy_nucleus`, `entropy_full`
    - candidates: `forms` (surface strings, correct casing), `form_items`
      (lowercase lexical item each form belongs to; punctuation maps to
      itself) and `form_p` (probability of each form; 0 where pruned)
    """
    __slots__ = ("logprob", "logprob_raw", "logprob_spoken", "entropy_nucleus", "entropy_full",
                 "forms", "form_items", "form_p", "pruned_mass", "scored_mass", "n_expanded")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        if self.logprob_spoken is None:
            self.logprob_spoken = self.logprob

    def item_probs(self) -> Dict[str, float]:
        """P(lexical item | context): every casing of a word summed ("Baum"
        and "baum" are two spellings of one item, and so are "hundert" and
        "100"). Punctuation excluded."""
        out: Dict[str, float] = {}
        for it, p, f in zip(self.form_items, self.form_p, self.forms):
            if p > 0 and f[:1].isalnum():
                out[it] = out.get(it, 0.0) + float(p)
        return out

    def form_probs(self, include_punct: bool = True) -> Dict[str, float]:
        """P(surface form | context) -- what the POS and semantic levels
        need: "Essen" (noun) and "essen" (verb) are different candidates."""
        return {f: float(p) for f, p in zip(self.forms, self.form_p)
                if p > 0 and (include_punct or f[:1].isalnum())}


def _build_forms(items: Sequence[str], story_words: Sequence[str], case: str,
                 include_punct: bool = True, canonical=None, spellings=None
                 ) -> Tuple[List[str], List[str]]:
    """Candidate surface forms and the lexical item each belongs to.
    `canonical(word)` gives a word's item key (the pronunciation dictionary's
    spelling; default lowercase); `spellings(item)` gives further written
    forms of an item (e.g. "weiß" for the dictionary key "weiss") -- all of
    an item's forms are summed into its probability."""
    canonical = canonical or (lambda w: w.lower())
    forms: Dict[str, str] = {}
    for it in items:
        for base in [it] + list(spellings(it) if spellings else []):
            for f in ([base] if case == "lower" else [base, base[:1].upper() + base[1:]]):
                forms.setdefault(f, it)
    for w in story_words:                     # the text's own spellings are always candidates
        if w[:1].isalpha():
            forms.setdefault(w, canonical(w))
    # numerals: the digit string is another spelling of the same spoken word.
    # Several items can name one number ("hundert", "einhundert"); "100" is
    # read as the shortest of them.
    reading: Dict[str, str] = {}
    for it in dict.fromkeys(forms.values()):
        n = german_number(it)
        if n is not None:
            d = str(n)
            if d not in reading or (len(it), it) < (len(reading[d]), reading[d]):
                reading[d] = it
    for d, it in reading.items():
        forms.setdefault(d, it)
    if include_punct:
        for p in PUNCT_FORMS:
            forms.setdefault(p, p)
    return list(forms), list(forms.values())


class VocabularyScorer:
    """Fast path for a real GPT-2: at every unit of `layout`, P(form | context)
    for every candidate surface form (all dictionary items, both casings,
    the text's own spellings, and punctuation), via the shared-context
    branch-and-bound scorer in `vocab_scorer.py` -- see that module for the
    exactness guarantee. GPT-2 reads the natural text (see `lexical.py`)."""

    def __init__(self, lm, layout, items: Sequence[str], phon_units: Dict[str, Sequence[str]],
                 case: str = "max", eps: float = 0.01, top_p: float = config.TOP_P,
                 top_k: int = config.TOP_K, canonical=None, spellings=None):
        self.lm = lm
        self.layout = layout
        self.idl = lm.layout_ids(layout)
        self.eps = eps
        self.top_p, self.top_k = top_p, top_k
        self.canonical = canonical or (lambda w: w.lower())
        self.forms, self.form_items = _build_forms(items, layout.words, "lower" if case == "lower" else "both",
                                                   canonical=self.canonical, spellings=spellings)
        self.form_index = {f: k for k, f in enumerate(self.forms)}
        # words and numerals are space-led and take the end-of-word factor;
        # punctuation attaches to the word before it
        self.is_word = np.array([f[:1].isalnum() for f in self.forms])
        self.item_forms: Dict[str, List[int]] = {}
        for k, it in enumerate(self.form_items):
            self.item_forms.setdefault(it, []).append(k)
        codes: Dict[str, int] = {}
        self.first_phon = np.array([codes.setdefault(phon_units[it][0], len(codes))
                                    if phon_units.get(it) else -1 for it in self.form_items])
        self._phon_codes = codes
        self._phon_units = phon_units
        self._tries: Dict[str, object] = {}
        self.stream = lm.context_stream(self.idl, sentence_starts=_sentence_starts(layout, self.idl))

    def trie(self, word_lead: str):
        """Token trie of all forms for positions where a WORD would be
        written with `word_lead` (" " mid-sentence; "" after a line break,
        an opening quote or the text start). Punctuation is never space-led."""
        if word_lead not in self._tries:
            seqs = [self.lm.encode((word_lead if w else "") + f) for f, w in zip(self.forms, self.is_word)]
            self._tries[word_lead] = FormTrie(seqs)
        return self._tries[word_lead]

    def self_check(self, positions: Sequence[int] = (1, 2), tol: float = 0.02) -> None:
        """Cross-check the fast scorer (exact mode) against the independent
        `lm.word_logprob` on a few forms. Raises on disagreement."""
        lay = self.layout
        for i in positions:
            if i >= len(lay.words) or self.idl.first[i] < 0 or not lay.words[i][:1].isalpha():
                continue
            prefix = lay.text[:lay.spans[i][0]]
            sample = [lay.words[i], "Haus", "haus", "gestern", ".", ","]
            st = self.stream.state_at(self.idl.first[i])
            lead = self.idl.lead[i]
            dens = [self.lm.position_log_den(st.last_logprobs, lead if f[:1].isalpha() else "",
                                             self.idl.adjacent[i]) for f in sample]
            seqs = [self.lm.encode((lead if f[:1].isalpha() else "") + f) for f in sample]
            lp, _ = self.lm.scorer.score(st, FormTrie(seqs), log_den=np.array(dens), eps=0.0,
                                         eow=self.lm.eow_correction,
                                         form_eow=np.array([f[:1].isalpha() for f in sample]))
            for f, a in zip(sample, lp):
                # punctuation attaches to the preceding word: no space before it
                b = self.lm.word_logprob(prefix if f[:1].isalpha() else prefix.rstrip(" "), f)
                if abs(a - b) > tol:
                    raise RuntimeError(f"fast vocabulary scorer disagrees with word_logprob for {f!r} at "
                                       f"position {i}: {a:.4f} vs {b:.4f} nats -- do not trust this run.")

    def position(self, i: int, candidates: bool = True) -> PositionResult:
        """`candidates=False`: only the presented unit's own values (no
        vocabulary scoring) -- e.g. for punctuation rows."""
        lay, idl, lm = self.layout, self.idl, self.lm
        word = lay.words[i]
        f, l = idl.first[i], idl.last[i]
        nan = float("nan")
        if f < 0:
            return PositionResult(logprob=nan, logprob_raw=nan, entropy_nucleus=nan, entropy_full=nan,
                                  forms=[], form_items=[], form_p=np.zeros(0), pruned_mass=nan,
                                  scored_mass=nan, n_expanded=0)
        st = self.stream.state_at(f)
        is_w = is_word_unit(word)
        word_lead = idl.lead[i] if is_w else " "
        adj = idl.adjacent[i]
        den_word = lm.position_log_den(st.last_logprobs, word_lead, adj)
        den_punct = lm.position_log_den(st.last_logprobs, "", adj)

        # the presented unit, scored on its own tokens from the text itself
        raw, eow = lm.scorer.score_sequence(st, idl.ids[f:l + 1])
        logprob = raw + (eow if (is_w and lm.eow_correction) else 0.0) - (den_word if is_w else den_punct)
        _, e_nuc, e_full = lm._stats_from_logprobs(st.last_logprobs, self.top_p, self.top_k,
                                                    idl.lead[i], want_nucleus=is_w)
        if not candidates:
            return PositionResult(logprob=logprob, logprob_raw=raw, entropy_nucleus=e_nuc,
                                  entropy_full=e_full, forms=[], form_items=[], form_p=np.zeros(0),
                                  pruned_mass=nan, scored_mass=nan, n_expanded=0)

        # every candidate form
        trie = self.trie(word_lead)
        tf = self.form_index.get(word)
        strict = None
        tcode = self._phon_codes.get((self._phon_units.get(self.canonical(word)) or [None])[0], -2)
        if tcode >= 0:
            strict = self.first_phon == tcode
        form_lp, info = lm.scorer.score(
            st, trie, log_den=np.where(self.is_word, den_word, den_punct), eow=lm.eow_correction,
            target_form=tf, strict_forms=strict, eps=self.eps, form_eow=self.is_word)
        # spoken-word probability: every spelling of the same item the listener
        # cannot distinguish (casings, digits for a numeral), the written form
        # at its exact value
        logprob_spoken = logprob
        if is_w:
            alt = [k for k in self.item_forms.get(self.canonical(word), []) if self.forms[k] != word
                   and np.isfinite(form_lp[k])]
            if alt:
                logprob_spoken = float(np.logaddexp.reduce(np.concatenate([[logprob], form_lp[alt]])))
        return PositionResult(logprob=logprob, logprob_raw=raw, logprob_spoken=logprob_spoken,
                              entropy_nucleus=e_nuc, entropy_full=e_full,
                              forms=self.forms, form_items=self.form_items,
                              form_p=np.exp(np.minimum(form_lp, 0.0)),
                              pruned_mass=info.pruned_mass_loose + info.pruned_mass_strict,
                              scored_mass=info.scored_mass, n_expanded=info.n_expanded)


class GenericVocabularyScorer:
    """Slow reference path for any `LanguageModel` (e.g. the toy bigram model
    used by `--backend toy`): same interface as `VocabularyScorer`, built on
    `word_logprob` / `score_words_batch` with unit-list contexts."""

    def __init__(self, lm, layout, items: Sequence[str], phon_units=None, case: str = "max",
                 eps: float = 0.0, top_p: float = config.TOP_P, top_k: int = config.TOP_K,
                 canonical=None, spellings=None):
        self.lm = lm
        self.layout = layout
        self.top_p, self.top_k = top_p, top_k
        self.forms, self.form_items = _build_forms(items, layout.words, "lower" if case == "lower" else "both")

    def self_check(self, positions=(), tol: float = 0.05) -> None:
        return None

    def position(self, i: int, candidates: bool = True) -> PositionResult:
        words = self.layout.words
        ctx, w = words[:i], words[i]
        lp = self.lm.word_logprob(ctx, w)
        _, e_nuc, e_full = self.lm.next_word_start_distribution(ctx, top_p=self.top_p, top_k=self.top_k)
        if not candidates:
            nan = float("nan")
            return PositionResult(logprob=lp, logprob_raw=lp, entropy_nucleus=e_nuc, entropy_full=e_full,
                                  forms=[], form_items=[], form_p=np.zeros(0), pruned_mass=nan,
                                  scored_mass=nan, n_expanded=0)
        scores = self.lm.score_words_batch(ctx, self.forms)
        p = np.array([math.exp(scores[f]) if scores[f] > -700 else 0.0 for f in self.forms])
        return PositionResult(logprob=lp, logprob_raw=lp, entropy_nucleus=e_nuc, entropy_full=e_full,
                              forms=self.forms, form_items=self.form_items, form_p=p,
                              pruned_mass=0.0, scored_mass=float(p.sum()), n_expanded=0)


def spoken_lexical_sweep(lm, layout, top_p: float = config.TOP_P, top_k: int = config.TOP_K):
    """Per unit `(logprob_written, logprob_spoken, entropy_nucleus)` from one
    walk over the text -- no vocabulary scoring, so it is cheap enough to run
    for additional models. The spoken value sums the unit's spellings that a
    listener cannot tell apart (`normalize_de.spoken_variants`), each scored in
    the same context, with the same end-of-word / whitespace definition as
    the written one."""
    idl = lm.layout_ids(layout)
    stream = lm.context_stream(idl, sentence_starts=_sentence_starts(layout, idl))
    out = []
    nan = float("nan")
    for i, word in enumerate(layout.words):
        f, l = idl.first[i], idl.last[i]
        if f < 0:
            out.append((nan, nan, nan))
            continue
        st = stream.state_at(f)
        is_w = is_word_unit(word)
        lead = idl.lead[i]
        den = lm.position_log_den(st.last_logprobs, lead if is_w else "", idl.adjacent[i])
        raw, eow = lm.scorer.score_sequence(st, idl.ids[f:l + 1])
        lp = raw + (eow if (is_w and lm.eow_correction) else 0.0) - den
        spoken = lp
        if is_w:
            vals = [lp]
            for v in spoken_variants(word)[1:]:
                r2, e2 = lm.scorer.score_sequence(st, lm.encode(lead + v))
                vals.append(r2 + (e2 if lm.eow_correction else 0.0) - den)
            spoken = float(np.logaddexp.reduce(vals))
        _, e_nuc, _ = lm._stats_from_logprobs(st.last_logprobs, top_p, top_k, lead, want_nucleus=is_w)
        out.append((lp, spoken, e_nuc))
    return out
