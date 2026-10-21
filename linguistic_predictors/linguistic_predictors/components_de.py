"""The real German models and resources, and `load_components` to load them once.

- language model: a GPT-2 checkpoint (`GERMAN_GPT2_CHECKPOINTS`)
- pronunciations: MFA's `german_mfa` dictionary + G2P (`lexicon_mfa.py`)
- POS tags: spaCy `de_core_news_lg`, Universal Dependencies tags (`.pos_`)
- word vectors: fastText `cc.de.300.bin` (cased, subword-aware), else spaCy's
- frequencies: SUBTLEX-DE (`frequency_de.py`), else `wordfreq`
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .phonemic import FrequencyPrior
from .syllabic import DE_SAMPA_VOWELS

# Chosen by scripts/benchmark_models.py: perplexity on the study text and
# held-out fit to German eye-tracking reading times (PoTeC).
#                                  story bits/word   Delta-LL FPRT / TFT (milli-nats/word)
#   benjamin/gerpt2                     6.17            111.6 / 124.1   <- best human fit
#   benjamin/gerpt2-large               5.16             92.7 / 119.0
#   malteos/gpt2-xl-wechsel-german      4.99             54.1 /  87.6   (paper-sized, weakest fit)
#   dbmdz/german-gpt2                   7.57             51.6 /  79.2
GERMAN_GPT2_CHECKPOINTS = {
    "small": "benjamin/gerpt2",
    "large": "benjamin/gerpt2-large",
    "xl": "malteos/gpt2-xl-wechsel-german",
    "dbmdz": "dbmdz/german-gpt2",
}


def resolve_device(device: str = "auto") -> str:
    """'auto' -> Apple GPU ('mps') or NVIDIA GPU ('cuda') if present, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _case_variants(word: str) -> List[str]:
    """Spellings to try in a cased vector table: as written first, then the
    Capitalised / lowercase counterpart. German vectors are cased
    ("Vorruhestand" has one, "vorruhestand" does not)."""
    w = word.strip()
    out = [w]
    for v in (w[:1].upper() + w[1:], w.lower()):
        if v and v not in out:
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# spaCy                                                                       #
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4)
def _load_spacy(model_name: str):
    import spacy
    # `.pos_` comes from tagger/morphologizer/attribute_ruler; the parser, NER
    # and lemmatizer only cost time (identical UD tags without them)
    return spacy.load(model_name, disable=["parser", "ner", "lemmatizer"])


_NO_SPACE_BEFORE = set(".,!?;:")


def _doc_from_words(vocab, words: List[str]):
    """A spaCy Doc built from our own units (no re-tokenisation), so token i
    is unit i; natural spacing (no space before punctuation)."""
    from spacy.tokens import Doc
    spaces = [not (j + 1 < len(words) and words[j + 1] in _NO_SPACE_BEFORE) for j in range(len(words))]
    if spaces:
        spaces[-1] = False
    return Doc(vocab, words=list(words), spaces=spaces)


class SpacyPOSTaggerDE:
    """`syntactic.POSTagger` with spaCy's German pipeline; candidates are
    tagged in batches with `nlp.pipe`."""

    def __init__(self, model_name: str = "de_core_news_lg"):
        self.nlp = _load_spacy(model_name)

    def tag_word_in_context(self, before: List[str], word: str, after: List[str]) -> str:
        doc = self.nlp(_doc_from_words(self.nlp.vocab, list(before) + [word] + list(after)))
        return doc[len(before)].pos_

    def tag_words_batch(self, candidate_words: List[str], before: List[str], after: List[str]) -> Dict[str, str]:
        docs = [_doc_from_words(self.nlp.vocab, list(before) + [w] + list(after)) for w in candidate_words]
        return {w: doc[len(before)].pos_ for w, doc in zip(candidate_words, self.nlp.pipe(docs, batch_size=256))}


# --------------------------------------------------------------------------- #
# word vectors                                                                #
# --------------------------------------------------------------------------- #
class SpacyEmbeddingsDE:
    """spaCy's static vectors (needs a model that ships them, e.g. `_lg`)."""

    def __init__(self, model_name: str = "de_core_news_lg"):
        self.nlp = _load_spacy(model_name)
        if self.nlp.vocab.vectors.shape[0] == 0:
            raise ValueError(f"'{model_name}' has no word vectors; use de_core_news_lg.")

    def vector(self, word: str) -> Optional[np.ndarray]:
        for form in _case_variants(word):
            lex = self.nlp.vocab[form]
            if lex.has_vector and lex.vector_norm > 0:
                return lex.vector
        return None


class FastTextTableEmbeddingsDE:
    """fastText `cc.de.300.bin` vectors without keeping the 7 GB model in
    memory next to GPT-2: `prepare(words)` computes the vectors of exactly
    the words a run needs (every candidate form + the text) once and caches
    them next to the .bin (~350 MB for the full dictionary in both casings).
    The model is subword-aware, so rare compounds get real vectors."""

    def __init__(self, bin_path: str):
        self.bin_path = str(Path(bin_path).expanduser())
        stem = str(Path(self.bin_path).with_suffix("")) + ".table"
        self._words_file = Path(stem + ".words.txt")
        self._mat_file = Path(stem + ".npy")
        self._index: Dict[str, int] = {}
        self._mat = np.zeros((0, 300), dtype=np.float32)
        if self._words_file.exists() and self._mat_file.exists():
            words = self._words_file.read_text(encoding="utf-8").split("\n")
            self._mat = np.load(self._mat_file)
            self._index = {w: i for i, w in enumerate(words[:len(self._mat)])}

    def prepare(self, words) -> int:
        missing = sorted({w.strip() for w in words if w and w.strip() and w.strip() not in self._index})
        if not missing:
            return 0
        if not Path(self.bin_path).exists():
            raise FileNotFoundError(
                f"{len(missing)} word vector(s) are not in the fastText cache (e.g. {missing[:5]}) "
                f"and {self.bin_path} is not there to compute them.")
        import gc
        import fasttext
        print(f"      fastText: computing {len(missing)} word vector(s) from {self.bin_path} "
              "(loads the 7 GB model once; cached afterwards)")
        ft = fasttext.load_model(self.bin_path)
        new = np.zeros((len(missing), ft.get_dimension()), dtype=np.float32)
        for k, w in enumerate(missing):
            form = next((f for f in _case_variants(w) if ft.get_word_id(f) >= 0), w)
            new[k] = ft.get_word_vector(form)
        del ft
        gc.collect()
        start = len(self._mat)
        self._mat = np.concatenate([self._mat, new]) if start else new
        for k, w in enumerate(missing):
            self._index[w] = start + k
        np.save(self._mat_file, self._mat)
        self._words_file.write_text("\n".join(sorted(self._index, key=self._index.get)), encoding="utf-8")
        return len(missing)

    def vector(self, word: str) -> Optional[np.ndarray]:
        for form in _case_variants(word):
            i = self._index.get(form)
            if i is not None:
                v = self._mat[i]
                return v if np.linalg.norm(v) > 0 else None
        return None


def _fasttext_cache_exists(bin_path: str) -> bool:
    """The vector cache written by `FastTextTableEmbeddingsDE.prepare` is
    enough on its own when the .bin has been deleted to save disk space."""
    stem = str(Path(bin_path).expanduser().with_suffix("")) + ".table"
    return Path(stem + ".words.txt").exists() and Path(stem + ".npy").exists()


FASTTEXT_BIN_CANDIDATES = (
    "~/NLP/data/fasttext/cc.de.300.bin",
    "~/fasttext/cc.de.300.bin",
    str(Path(__file__).resolve().parent.parent / "data" / "fasttext" / "cc.de.300.bin"),
)


def build_embeddings_de(prefer: str = "auto", fasttext_path: Optional[str] = None,
                        spacy_model: str = "de_core_news_lg", log: Callable = print):
    """fastText if a `cc.de.300.bin` is found (`prefer="auto"`, the default,
    or required with `"fasttext"`), else spaCy's vectors."""
    if prefer in ("auto", "fasttext"):
        for p in ([fasttext_path] if fasttext_path else []) + list(FASTTEXT_BIN_CANDIDATES):
            if p and (Path(p).expanduser().exists() or _fasttext_cache_exists(p)):
                log(f"      embeddings: fastText {p}")
                return FastTextTableEmbeddingsDE(p)
        if prefer == "fasttext":
            raise FileNotFoundError(
                "cc.de.300.bin not found: download "
                "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz, "
                "gunzip it, and pass --fasttext-path.")
        log("      embeddings: spaCy vectors (fastText cc.de.300.bin not found)")
    return SpacyEmbeddingsDE(model_name=spacy_model)


def build_wordfreq_prior(vocabulary: Sequence[str], lang: str = "de") -> FrequencyPrior:
    """Fallback frequency prior from `wordfreq` when SUBTLEX-DE is absent."""
    from wordfreq import word_frequency
    freqs = {w: word_frequency(w, lang) for w in vocabulary}
    return FrequencyPrior({w: f for w, f in freqs.items() if f > 0})


# --------------------------------------------------------------------------- #
# everything at once                                                          #
# --------------------------------------------------------------------------- #
def load_components(mfa_dict: str, gpt2_model: str = GERMAN_GPT2_CHECKPOINTS["small"],
                    device: str = "auto", g2p_model: Optional[str] = "german_mfa",
                    spacy_model: str = "de_core_news_lg", embeddings: str = "auto",
                    fasttext_path: Optional[str] = None, frequency: str = "subtlex",
                    subtlex_path: Optional[str] = None, log: Callable = print) -> dict:
    """Load every model and resource once. Returns the component dict that
    `pipeline.compute_predictors` takes (`toy.toy_components` builds the
    same dict offline)."""
    from .lexical import GPT2LanguageModel
    from .lexicon_mfa import MFAPronDict

    dev = resolve_device(device)
    log(f"      GPT-2: {gpt2_model} on {dev}")
    lm = GPT2LanguageModel(gpt2_model, device=dev)
    pron_dict = MFAPronDict(mfa_dict, g2p_model_name=g2p_model)
    pos_tagger = SpacyPOSTaggerDE(model_name=spacy_model)
    emb = build_embeddings_de(prefer=embeddings, fasttext_path=fasttext_path,
                              spacy_model=spacy_model, log=log)
    subtlex = None
    if frequency == "subtlex":
        from .frequency_de import SubtlexDE, find_subtlex_de
        if find_subtlex_de(subtlex_path):
            subtlex = SubtlexDE(subtlex_path)
            log(f"      frequencies: SUBTLEX-DE ({subtlex.n_types} types, {subtlex.n_tokens / 1e6:.1f}M tokens)")
        else:
            log("      frequencies: SUBTLEX-DE not found (see frequency_de.py) -- using wordfreq")
    freq_prior = (subtlex.frequency_prior(pron_dict.vocabulary()) if subtlex
                  else build_wordfreq_prior(pron_dict.vocabulary()))
    return dict(lm=lm, pron_dict=pron_dict, freq_prior=freq_prior, pos_tagger=pos_tagger,
                embeddings=emb, vowel_set=DE_SAMPA_VOWELS, subtlex=subtlex)
