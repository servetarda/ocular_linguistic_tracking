"""All five levels for one text: `compute_predictors(text, components, options)`.

`components` comes from `components_de.load_components` (real models) or
`toy.toy_components` (offline). Returns one row per unit (word or punctuation
mark) as a DataFrame; phoneme/syllable values are lists, one value per
phoneme/syllable. Surprisal and entropy are in bits (`config.LOG_BASE`).

Column reference
----------------
word, word_model            the unit as written / as the models read it (spelling-normalised)
lexical_surprisal           -log2 P(word | preceding text), the word as written (Heilbron et al. 2022),
                            with the end-of-word correction
lexical_surprisal_raw       the same without the end-of-word correction (plain token product)
lexical_surprisal_spoken    summed over spellings a listener cannot tell apart (casing, digits)
lexical_entropy             entropy of the nucleus of the complete-word candidate distribution
                            (the same nucleus as POS and semantics; punctuation is a candidate)
entropy_full, entropy_words entropy of the full next-token distribution / of the scored vocabulary
unigram_surprisal           SUBTLEX-DE, Laplace-smoothed (in_subtlex: seen in the corpus)
pos, pos_surprisal, pos_entropy        syntactic level
semantic_prediction_error   cosine distance predicted vs actual vector (content words)
semantic_distance           Broderick et al. (2018) control
phon_* / syll_*             per-phoneme / per-syllable surprisal and entropy, hierarchical
                            (language-model prior) and freq (frequency prior)
dict_mass, pruned_mass      dictionary probability mass scored / skipped by the error bound
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd

from . import config
from .candidates import GenericVocabularyScorer, VocabularyScorer, spoken_lexical_sweep
from .cohort import sequential_cohort_stats, single_level_stats
from .lexical import GPT2LanguageModel
from .phonemic import phoneme_units
from .semantic import semantic_distance_broderick, semantic_prediction_error
from .syllabic import syllable_strings, syllable_units
from .syntactic import pos_surprisal_entropy
from .text import TextLayout, sentence_bounds, sentence_ids

# Universal Dependencies content-word tags (spaCy `.pos_`)
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


@dataclass
class PipelineOptions:
    top_p: float = config.TOP_P                 # nucleus: cumulative probability ...
    top_k: int = config.TOP_K                   # ... and at least this many candidates
    prune_eps: float = 0.01                     # relative error bound of the candidate scorer
    eow_correction: bool = True                 # end-of-word corrected word probabilities
    vocab_size: int = 0                         # 0 = the whole pronunciation dictionary
    case_fix: bool = True                       # score candidates in both casings
    spelling_normalization: bool = True         # Swiss "ss" -> "ß" for the models
    compounds: bool = True                      # compound/prefix boundaries in syllabification
    score_punctuation: bool = False             # full candidate scoring at punctuation rows
    pos_max_candidates: int = 0                 # 0 = tag the whole nucleus
    extra_lexical_models: Sequence[str] = ("benjamin/gerpt2-large",)
    batch_size: int = 128
    progress: bool = False


def _canonical(pron_dict) -> Callable[[str], str]:
    return getattr(pron_dict, "canonical", None) or (lambda w: w.lower())


def choose_candidate_vocab(pron_dict, freq_prior, words: Sequence[str], vocab_size: int) -> List[str]:
    """The dictionary (optionally its `vocab_size` most frequent words) plus
    every word of the text, as item keys."""
    full = [w.lower() for w in pron_dict.vocabulary()]
    if vocab_size and 0 < vocab_size < len(full):
        probs = freq_prior.distribution()
        full = sorted(full, key=lambda w: probs.get(w, 0.0), reverse=True)[:vocab_size]
    canon = _canonical(pron_dict)
    return sorted(set(full) | {canon(w) for w in words if any(ch.isalpha() for ch in w)})


def _nucleus(dist: Dict[str, float], top_p: float, top_k: int, cap: int) -> Dict[str, float]:
    total = sum(dist.values())
    out, cum = {}, 0.0
    for f, p in sorted(dist.items(), key=lambda kv: kv[1], reverse=True):
        out[f] = p
        cum += p / total
        if cum >= top_p and len(out) >= top_k:
            break
    if cap and len(out) > cap:
        out = dict(list(out.items())[:cap])
    return out


def compute_predictors(text: str, components: dict, options: PipelineOptions = None,
                       log: Callable = print) -> Tuple[pd.DataFrame, dict]:
    """(one row per unit, run metadata) for `text`. See the module docstring."""
    opt = options or PipelineOptions()
    lm, pron_dict, pos_tagger = components["lm"], components["pron_dict"], components["pos_tagger"]
    embeddings, subtlex = components["embeddings"], components["subtlex"]
    freq_prior, vowel_set = components["freq_prior"], components["vowel_set"]
    is_gpt2 = isinstance(lm, GPT2LanguageModel)
    canon = _canonical(pron_dict)

    # ---- the text, and the spelling the models read ----------------------
    orig_layout = TextLayout.from_text(text)
    layout, spelling_changes = orig_layout, []
    if subtlex is not None and opt.spelling_normalization:
        from .normalize_de import standardize_swiss_spelling
        norm_text, spelling_changes = standardize_swiss_spelling(orig_layout.text, subtlex.count)
        layout = TextLayout.from_text(norm_text)
        assert len(layout.words) == len(orig_layout.words), "spelling normalisation changed the unit count"
        if spelling_changes:
            log(f"      spelling normalised for the models: {len(spelling_changes)} word(s), "
                f"e.g. {spelling_changes[:5]}")
    words, display_words = layout.words, orig_layout.words
    if is_gpt2:
        lm.eow_correction = opt.eow_correction
        lm.scorer.max_tokens_per_batch = max(16, opt.batch_size * 16)

    # ---- pronunciations, frequency prior, syllable inventory --------------
    precompute_oov = getattr(pron_dict, "precompute_oov", None)
    if precompute_oov is not None:
        n_oov = precompute_oov(words)
        if n_oov:
            log(f"      resolved {n_oov} out-of-dictionary word(s) via G2P")
    if subtlex is not None:
        freq_prior = subtlex.frequency_prior(pron_dict.vocabulary())   # G2P words join with count + 1
        if opt.compounds and hasattr(pron_dict, "enable_compound_boundaries"):
            n_split = pron_dict.enable_compound_boundaries(subtlex.count_any_spelling)
            log(f"      compound/prefix boundaries for syllabification: {n_split} dictionary words split")
    vowel_set = getattr(pron_dict, "vowel_phones", None) or vowel_set

    cand_vocab = choose_candidate_vocab(pron_dict, freq_prior, words, opt.vocab_size)
    log(f"      candidate vocabulary: {len(cand_vocab)} words")
    phon_units = {w: p for w, p in ((w, pron_dict.phonemes(w)) for w in cand_vocab) if p}
    syll_units = syllable_units(phon_units, vowel_set)
    freq_dist = freq_prior.distribution()
    freq_phon_units = phoneme_units(list(freq_dist), pron_dict)
    freq_syll_units = syllable_units(freq_phon_units, vowel_set)

    # unigram probability: SUBTLEX-DE (Laplace-smoothed), else the raw wordfreq
    # probability -- never one renormalised over the dictionary, which would
    # add a dictionary-size-dependent constant
    if subtlex is not None:
        def unigram_prob(w: str):
            return subtlex.prob(w)
    else:
        try:
            from wordfreq import word_frequency

            def unigram_prob(w: str):
                f = word_frequency(w.lower(), "de")
                return f if f > 0 else None
        except ImportError:
            def unigram_prob(w: str):
                f = freq_dist.get(w.lower(), 0.0)
                return f if f > 0 else None

    # ---- the candidate distribution at every position ----------------------
    spellings = None
    if subtlex is not None:
        from .normalize_de import standard_spellings
        spellings = lambda it: standard_spellings(it, subtlex.count)   # noqa: E731
    Scorer = VocabularyScorer if is_gpt2 else GenericVocabularyScorer
    engine = Scorer(lm, layout, cand_vocab, phon_units,
                    case=("max" if (opt.case_fix and is_gpt2) else "lower"),
                    eps=opt.prune_eps, top_p=opt.top_p, top_k=opt.top_k,
                    canonical=canon, spellings=spellings)
    engine.self_check()
    prepare = getattr(embeddings, "prepare", None)
    if prepare is not None:
        prepare(list(engine.forms) + list(words))
    if is_gpt2:
        log(f"      scorer self-check passed: {len(engine.forms)} candidate forms, eps={opt.prune_eps}, "
            f"end-of-word correction {'on' if lm.eow_correction else 'off'}")

    # ---- unit by unit ----------------------------------------------------------
    sids = sentence_ids(words)
    log(f"      scoring {len(words)} units ...")
    t0 = time.time()
    rows = []
    for i, word in enumerate(words):
        if opt.progress and i and i % 10 == 0:
            rate = i / (time.time() - t0)
            log(f"      {i}/{len(words)}  ({rate:.2f} units/s, ~{(len(words) - i) / rate:.0f}s left)")
        wl = canon(word)
        is_punct = not any(ch.isalpha() for ch in word)
        full = (not is_punct) or opt.score_punctuation
        res = engine.position(i, candidates=full)
        lo, hi = sentence_bounds(sids, i)
        row = dict(
            position=i, word=display_words[i], word_model=word, sentence=sids[i],
            is_first_word=(i == 0), is_punctuation=is_punct,
            lexical_surprisal=config.surprisal_from_logprob(res.logprob),
            lexical_surprisal_raw=config.surprisal_from_logprob(res.logprob_raw),
            lexical_surprisal_spoken=config.surprisal_from_logprob(res.logprob_spoken),
            lexical_entropy=res.entropy_nucleus, entropy_full=res.entropy_full,
        )
        if not full:
            # punctuation: its own surprisal only (no acoustic onset)
            row.update(pos=pos_tagger.tag_word_in_context(words[lo:i], word, words[i + 1:hi]),
                       is_content_word=False, entropy_words=None, unigram_surprisal=None,
                       in_subtlex=None, pos_surprisal=None, pos_entropy=None,
                       semantic_prediction_error=None, dict_mass=None, pruned_mass=None,
                       n_pos_candidates=0, **{k: [] for k in _UNIT_COLUMNS})
            rows.append(row)
            continue

        lexical_p = math.exp(res.logprob) if res.logprob > -700 else 0.0
        dist = res.form_probs(include_punct=True)
        if dist.get(word, 0.0) <= 0:
            dist[word] = lexical_p                # the presented unit is always a candidate
        merged: Dict[str, float] = {}
        for f, p in dist.items():
            merged[f.lower()] = merged.get(f.lower(), 0.0) + p
        # the predicted next-word distribution: one nucleus of complete words
        # for lexical entropy, POS and semantics. It is what the model expects
        # before the word is heard -- the presented word is in it only if it
        # earned its place.
        nucleus = _nucleus(dist, opt.top_p, opt.top_k, 0)
        lexical_entropy = config.entropy(nucleus.values())
        pos_nucleus = _nucleus(dist, opt.top_p, opt.top_k, opt.pos_max_candidates)

        pos_s, pos_h, pos_tag = pos_surprisal_entropy(words[lo:i], word, words[i + 1:hi], pos_nucleus,
                                                      pos_tagger, actual_p=dist[word])
        is_content = pos_tag in CONTENT_POS
        sem_err = None
        if is_content:
            sem_err = semantic_prediction_error({f: p for f, p in nucleus.items() if f[:1].isalpha()},
                                                word, embeddings)

        cohort = {k: [] for k in _UNIT_COLUMNS}
        target_phons = pron_dict.phonemes(wl) if not is_punct else []
        if target_phons:
            target_syls = syllable_strings(target_phons, vowel_set)
            item_p = res.item_probs()
            if item_p.get(wl, 0.0) <= 0:
                item_p[wl] = lexical_p
            phon_units.setdefault(wl, target_phons)
            syll_units.setdefault(wl, target_syls)
            cohort["phon_surprisal_hier"], cohort["phon_entropy_hier"] = \
                sequential_cohort_stats(item_p, phon_units, target_phons)
            cohort["syll_surprisal_hier"], cohort["syll_entropy_hier"] = \
                sequential_cohort_stats(item_p, syll_units, target_syls)
            cohort["phon_surprisal_freq"], cohort["phon_entropy_freq"] = \
                single_level_stats(wl, target_phons, freq_dist, freq_phon_units)
            cohort["syll_surprisal_freq"], cohort["syll_entropy_freq"] = \
                single_level_stats(wl, target_syls, freq_dist, freq_syll_units)

        up = unigram_prob(word) if not is_punct else None       # SUBTLEX spelling (ß), not the dictionary key
        row.update(pos=pos_tag, is_content_word=is_content, lexical_entropy=lexical_entropy,
                   entropy_words=config.entropy(merged.values()),
                   unigram_surprisal=(config.surprisal(up) if up else None),
                   in_subtlex=(subtlex.in_corpus(word) if (subtlex is not None and not is_punct) else None),
                   pos_surprisal=pos_s, pos_entropy=pos_h, semantic_prediction_error=sem_err,
                   dict_mass=res.scored_mass, pruned_mass=res.pruned_mass,
                   n_pos_candidates=len(pos_nucleus), **cohort)
        rows.append(row)

    df = pd.DataFrame(rows)[_COLUMN_ORDER]

    # ---- comparison models: word level only, one cheap pass each -------------
    if is_gpt2:
        for name in opt.extra_lexical_models:
            if name == getattr(lm, "model_name", None):
                continue
            tag = name.split("/")[-1]
            log(f"      comparison model: {name}")
            lm2 = GPT2LanguageModel(name, device=lm.device, eow_correction=lm.eow_correction)
            st2 = spoken_lexical_sweep(lm2, layout, top_p=opt.top_p, top_k=opt.top_k)
            df[f"lexical_surprisal__{tag}"] = [config.surprisal_from_logprob(w) for w, _, _ in st2]
            df[f"lexical_surprisal_spoken__{tag}"] = [config.surprisal_from_logprob(sp) for _, sp, _ in st2]
            del lm2

    df["semantic_distance"] = semantic_distance_broderick(words, embeddings, sids)

    # ---- cohort health: phoneme positions where competition was left ----------
    live = [v for r in rows for v in r["phon_surprisal_hier"]]
    pct_live = 100.0 * sum(1 for v in live if v > 1e-9) / len(live) if live else 0.0
    cohort_health = dict(phoneme_positions=len(live), pct_with_real_competition=round(pct_live, 1),
                         candidate_vocab_size=len(cand_vocab))
    log(f"      phoneme cohort health: {pct_live:.0f}% of phoneme positions had competition "
        f"(vocabulary {len(cand_vocab)} words)")
    if not is_gpt2 or len(cand_vocab) < 2000 or pct_live < 25:
        log("      WARNING: the candidate vocabulary is too small for the phoneme/syllable levels to\n"
            "      carry information (expected for the toy backend; not a real result).")

    meta = dict(words=words, vowel_set=vowel_set, cand_vocab_size=len(cand_vocab),
                n_forms=len(engine.forms), cohort_health=cohort_health, is_gpt2=is_gpt2,
                spelling_changes=spelling_changes, seconds=round(time.time() - t0, 1))
    return df, meta


_UNIT_COLUMNS = ("phon_surprisal_hier", "phon_entropy_hier", "phon_surprisal_freq", "phon_entropy_freq",
                 "syll_surprisal_hier", "syll_entropy_hier", "syll_surprisal_freq", "syll_entropy_freq")

_COLUMN_ORDER = [
    "position", "word", "word_model", "sentence", "pos", "is_content_word", "is_first_word",
    "is_punctuation", "lexical_surprisal", "lexical_surprisal_raw", "lexical_surprisal_spoken",
    "lexical_entropy", "entropy_full", "entropy_words", "unigram_surprisal", "in_subtlex",
    "pos_surprisal", "pos_entropy", "semantic_prediction_error", *_UNIT_COLUMNS,
    "dict_mass", "pruned_mass", "n_pos_candidates",
]
