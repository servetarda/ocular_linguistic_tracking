"""Sequential cohort model (Heilbron et al. 2022, Fig. 6 / SI Eq. 8).

Candidate words are grouped by their next unit -- phoneme, or syllable for
the syllabic extension -- and weighted by a prior: word frequency in the
single-level model, the language model's contextual probability in the
hierarchical one. `phonemic.py` and `syllabic.py` only differ in how a word
is turned into units.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

from . import config

# The outcome "the word ends here" in a cohort's next-unit distribution.
END_OF_WORD = "<eow>"


def sequential_cohort_stats(
    candidate_weights: Dict[str, float],
    candidate_units: Dict[str, Sequence[str]],
    target_units: Sequence[str],
) -> Tuple[List[float], List[float]]:
    """Position-by-position surprisal & entropy of `target_units`, given a
    weighted cohort of candidate words (each candidate must also appear in
    `candidate_units`, mapping candidate -> its own unit sequence, e.g. its
    phoneme or syllable transcription).

    The cohort at position i is every candidate consistent with the first i
    units heard so far -- INCLUDING candidates that are exactly i units long.
    Those have ended, so they contribute an explicit END_OF_WORD outcome to
    the next-unit distribution instead of being dropped. This is the
    standard cohort formula (Brodbeck et al. 2018; Gaston & Marantz 2018;
    Gwilliams et al. 2018):

        P(unit_i | units_<i) = sum_{w in cohort_{i+1}} p(w) / sum_{w in cohort_i} p(w)

    whose denominator contains the words that END after units_<i. Dropping
    them (what this function did before) renormalises over longer words
    only, i.e. it silently conditions on "the word goes on" -- which the
    listener cannot know yet -- and overstates every continuation's
    probability exactly where short words compete (e.g. "Tisch" inside
    "Tischler").

    Returns
    -------
    surprisals : list[float], length len(target_units)
        surprisal (in config.LOG_BASE units) of the actual unit at each
        position, given all preceding actual units (not the candidate's own
        units) as the "phonemes/syllables so far".
    entropies : list[float], length len(target_units)
        entropy of the predicted next-unit distribution at each position
        (END_OF_WORD included as one outcome).
    """
    # restrict to candidates we actually have a unit-transcription for
    cands = [w for w in candidate_weights if w in candidate_units and candidate_units[w]]
    if not cands:
        # nothing to compare against -> the value is undefined, not "maximal".
        # See the `total <= 0` branch below for why NaN rather than a ceiling.
        nan = float("nan")
        return ([nan] * len(target_units), [nan] * len(target_units))

    surprisals: List[float] = []
    entropies: List[float] = []

    # The cohort only ever SHRINKS as units come in: everyone matching the
    # first i+1 units also matched the first i. So narrow the survivors in
    # place instead of re-scanning the whole candidate vocabulary at every
    # position -- that rescan was O(vocab x units) per word, which dominated
    # the runtime once the vocabulary grew to dictionary size.
    cohort = cands
    for i in range(len(target_units)):
        if not cohort:
            # no candidate is consistent with the units so far -- undefined
            surprisals.append(float("nan"))
            entropies.append(float("nan"))
            continue

        total = 0.0
        dist: Dict[str, float] = {}
        for w in cohort:
            wt = candidate_weights[w]
            if wt <= 0:
                continue
            total += wt
            units = candidate_units[w]
            nxt = units[i] if len(units) > i else END_OF_WORD
            dist[nxt] = dist.get(nxt, 0.0) + wt
        if total <= 0:
            # every surviving candidate has zero probability: the model has no
            # opinion here, so the value is undefined. NaN (excluded from a
            # regression) is honest; config.surprisal(0.0) would manufacture
            # the EPSILON ceiling -- a constant masquerading as a measurement,
            # and a high-leverage outlier in any model fitted on this column.
            surprisals.append(float("nan"))
            entropies.append(float("nan"))
            continue

        actual = target_units[i]
        p_actual = dist.get(actual, 0.0) / total
        # no EPSILON clipping: a very improbable unit (e.g. inside a coined
        # compound) keeps its true surprisal instead of the 39.9-bit ceiling
        surprisals.append(config.surprisal_from_logprob(math.log(p_actual)) if p_actual > 0
                          else config.surprisal(0.0))
        entropies.append(config.entropy([v / total for v in dist.values()]))

        # survivors for the next position: those whose unit i matched
        cohort = [w for w in cohort if len(candidate_units[w]) > i and candidate_units[w][i] == actual]

    return surprisals, entropies


def single_level_stats(
    target: str,
    target_units: Sequence[str],
    frequency: Mapping[str, float],
    units: Mapping[str, Sequence[str]],
) -> Tuple[List[float], List[float]]:
    """The single-level (frequency-weighted) cohort for `target` (an item key
    of `frequency`/`units`). A target the frequency list lacks joins the
    cohort with a negligible weight, so its own continuation always exists."""
    if not target_units:
        return [], []
    if target not in frequency:
        frequency = dict(frequency)
        frequency[target] = config.EPSILON
    if target not in units:
        units = dict(units)
        units[target] = list(target_units)
    return sequential_cohort_stats(frequency, units, target_units)
