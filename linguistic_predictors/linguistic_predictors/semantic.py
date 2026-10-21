"""Semantic level.

Prediction error (Heilbron et al. 2022, SI A.3, Eq. 7): the predicted
meaning is the probability-weighted mean word vector of the candidates in
the nucleus; the error is the cosine distance between that vector and the
presented word's vector.

Semantic distance, the integration-difficulty control (Broderick et al.
2018): 1 - Pearson r between a word's vector and the mean vector of the
words before it in the same sentence (the previous sentence for a
sentence-initial word). Punctuation is neither scored nor averaged in.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np


class WordEmbeddings(Protocol):
    def vector(self, word: str) -> Optional[np.ndarray]:
        """The word's vector, or None if it has none."""
        ...


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def predicted_vector(candidates: Dict[str, float], embeddings: WordEmbeddings) -> Optional[np.ndarray]:
    """Probability-weighted mean vector over the candidates that have one."""
    vecs, weights = [], []
    for w, p in candidates.items():
        v = embeddings.vector(w)
        if v is not None and p > 0:
            vecs.append(v)
            weights.append(p)
    if not vecs:
        return None
    weights = np.asarray(weights, dtype=float)
    return np.tensordot(weights / weights.sum(), np.stack(vecs), axes=(0, 0))


def semantic_prediction_error(candidates: Dict[str, float], actual: str,
                              embeddings: WordEmbeddings) -> Optional[float]:
    pred = predicted_vector(candidates, embeddings)
    vec = embeddings.vector(actual)
    if pred is None or vec is None:
        return None
    return cosine_distance(pred, vec)


def semantic_distance_broderick(words: Sequence[str], embeddings: WordEmbeddings,
                                sentence_ids: Sequence[int]) -> List[Optional[float]]:
    vecs = [embeddings.vector(w) if any(ch.isalpha() for ch in w) else None for w in words]
    by_sentence: Dict[int, List[int]] = {}
    for i, s in enumerate(sentence_ids):
        by_sentence.setdefault(s, []).append(i)

    out: List[Optional[float]] = [None] * len(words)
    for s, idxs in by_sentence.items():
        prev = [vecs[j] for j in by_sentence.get(s - 1, []) if vecs[j] is not None]
        prev_mean = np.mean(np.stack(prev), axis=0) if prev else None
        for k, i in enumerate(idxs):
            if vecs[i] is None:
                continue
            if k == 0:
                ref = prev_mean
            else:
                seen = [vecs[j] for j in idxs[:k] if vecs[j] is not None]
                ref = np.mean(np.stack(seen), axis=0) if seen else None
            if ref is None:
                continue
            r = np.corrcoef(vecs[i], ref)[0, 1]
            out[i] = float(1.0 - (r if np.isfinite(r) else 0.0))
    return out
