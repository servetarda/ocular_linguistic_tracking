"""Shared constants and the surprisal/entropy helpers."""

import numpy as np

# Nucleus of a next-word distribution (Heilbron et al. 2022, SI A.3): the
# most probable candidates until their cumulative probability reaches TOP_P,
# but at least TOP_K of them.
TOP_P = 0.9
TOP_K = 40

# Information-theoretic base for surprisal/entropy. 2 -> bits, e -> nats.
LOG_BASE = 2.0

# Floor for probabilities passed to `surprisal` (avoids log(0)).
EPSILON = 1e-12

# GPT-2's context window (used for the "windowed approach" the paper mentions
# for texts longer than this many tokens).
GPT2_CONTEXT_WINDOW = 1024


def log(x):
    return np.log(x) / np.log(LOG_BASE)


def surprisal(p):
    """surprisal = -log(p), in units set by LOG_BASE.

    `p` is clipped at EPSILON, so this saturates at ~39.9 bits. When you
    already hold a log-probability, prefer `surprisal_from_logprob` -- going
    log -> exp -> log loses precision for small p and silently flattens every
    genuinely improbable word onto the same ceiling value.
    """
    p = np.clip(p, EPSILON, 1.0)
    # `+ 0.0` turns the IEEE negative zero from -log(1.0) into a plain 0.0
    # (cosmetic: a perfectly-predicted unit shows "0.0", not "-0.0").
    return float(-log(p) + 0.0)


def surprisal_from_logprob(logp):
    """surprisal from a NATURAL log-probability, in units set by LOG_BASE.

    The exact route when the model hands you a log-probability: no exp() and
    no epsilon clipping, so a word GPT-2 considers vanishingly unlikely gets
    its true surprisal instead of being pinned to the EPSILON ceiling. That
    ceiling matters -- with EPSILON=1e-12 every word below p=1e-12 reads as
    exactly 39.863 bits, which would turn real variance in the tail of the
    distribution into a constant.
    """
    if logp != logp:               # NaN: undefined (e.g. a unit with no token of its own)
        return float("nan")
    if logp == float("-inf"):
        return float("inf")
    return float(-logp / np.log(LOG_BASE) + 0.0)


def entropy(probs):
    """Shannon entropy of a discrete distribution (iterable of probabilities
    that should sum to ~1; renormalised defensively)."""
    probs = np.asarray(list(probs), dtype=float)
    probs = probs[probs > 0]
    if probs.sum() <= 0:
        return 0.0
    probs = probs / probs.sum()
    # `+ 0.0` collapses IEEE negative zero (-1 * log2(1) == -0.0 for a
    # single-item distribution) to a plain 0.0, so exported CSVs read
    # "0.0" rather than "-0.0" for positions whose cohort had exactly one
    # member. Purely cosmetic; the value is already zero.
    return float(-(probs * log(probs)).sum() + 0.0)
