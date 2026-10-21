#!/usr/bin/env python3
"""Which German language model -- and which way of computing word surprisal
-- best captures human processing?

Two criteria, per model / pipeline variant:

1. Perplexity on the study text (`--story`): bits per word (lower = the
   model predicts this text better).
2. Fit to human reading times: the Potsdam Textbook Corpus (PoTeC; Jakobi et
   al. 2025, 75 native German readers x 12 texts, eye-tracking). Per word,
   first-pass reading time (FPRT) and total fixation time (TFT) are averaged
   over readers (log ms, fixated words only) and regressed on word length +
   SUBTLEX-DE frequency (current and previous word). The score is the
   held-out gain in log-likelihood per word when surprisal (current +
   previous word, for spillover) is added -- the standard Delta-LL measure
   (e.g. Wilcox et al. 2020; Oh & Schuler 2023) -- over 10 x 10-fold
   cross-validation. Higher = surprisal explains more of human reading.

    python scripts/benchmark_models.py [--device mps]

PoTeC reading measures: https://osf.io/download/3ywhz/ (unzip into
data/potec/rm_merged/). Writes docs/benchmark_results.csv.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from linguistic_predictors.frequency_de import SubtlexDE
from linguistic_predictors.lexical import GPT2LanguageModel
from linguistic_predictors.text import TextLayout, is_word_unit

MODELS = [
    "dbmdz/german-gpt2",                 # 124M, the model the pipeline originally used
    "stefan-it/german-gpt2-larger",      # 124M, more training data
    "benjamin/gerpt2",                   # 124M-class, CC-100
    "benjamin/gerpt2-large",             # 774M
    "malteos/gpt2-xl-wechsel-german",    # 1.5B -- GPT-2 XL size, as in the paper
]
NA = ['#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
      '<NA>', 'N/A', 'NA', 'NaN', 'None', 'n/a', 'nan', '']       # PoTeC text p3 contains the word "null"


# --------------------------------------------------------------------------- #
def load_potec(root: Path):
    files = sorted(glob.glob(str(root / "rm_merged" / "reading_measures_merged" / "*_merged.tsv")))
    if not files:
        sys.exit(f"no PoTeC reading measures under {root}/rm_merged/reading_measures_merged/")
    cols = ["word", "word_with_punct", "word_index_in_text", "text_id", "reader_id", "line",
            "contains_symbol", "contains_abbreviation", "FPRT", "TFT"]
    df = pd.concat([pd.read_csv(f, sep="\t", keep_default_na=False, na_values=NA, usecols=cols)
                    for f in files], ignore_index=True)
    texts = {}
    for tid, g in df.groupby("text_id"):
        w = (g.drop_duplicates("word_index_in_text").sort_values("word_index_in_text"))
        texts[tid] = w[["word_index_in_text", "word", "word_with_punct", "line",
                        "contains_symbol", "contains_abbreviation"]].reset_index(drop=True)
    rt = []
    for m in ("FPRT", "TFT"):
        x = df[df[m] > 0].assign(logrt=lambda d: np.log(d[m]))
        rt.append(x.groupby(["text_id", "word_index_in_text"])["logrt"].agg(["mean", "count"])
                  .rename(columns={"mean": f"log{m}", "count": f"n_{m}"}))
    return texts, pd.concat(rt, axis=1).reset_index()


def potec_word_surprisal(lm, words_df, variant):
    """Surprisal (bits) of each PoTeC word = sum over its letter units."""
    toks = words_df["word_with_punct"].astype(str).tolist()
    text, spans, pos = [], [], 0
    for t in toks:
        if text:
            text.append(" ")
            pos += 1
        spans.append((pos, pos + len(t)))
        text.append(t)
        pos += len(t)
    text = "".join(text)
    lay = layout_for(text, variant)
    stats = lm.sweep_layout(lay)
    starts = [a for a, _ in spans]
    out = np.zeros(len(toks))
    has = np.zeros(len(toks), dtype=bool)
    for u, (a, _), st in zip(lay.words, lay.spans, stats):
        if not is_word_unit(u):
            continue
        k = bisect.bisect_right(starts, a) - 1
        out[k] += -st["logprob"] / math.log(2)
        has[k] = True
    out[~has] = np.nan
    return out


def layout_for(text, variant):
    """'old' reproduces the pre-audit input: units re-joined with spaces
    (punctuation detached), digits/quotes dropped."""
    lay = TextLayout.from_text(text)
    if variant != "old":
        return lay
    words = lay.words
    parts, spans, pos = [], [], 0
    for w in words:
        if parts:
            parts.append(" ")
            pos += 1
        spans.append((pos, pos + len(w)))
        parts.append(w)
        pos += len(w)
    return TextLayout("".join(parts), list(words), spans)


def configure(lm, variant):
    if variant == "old":           # space seed, no end-of-word / whitespace correction
        lm.seed_ids = lm.encode(" ")[:1]
        lm.eow_correction = False
    elif variant == "no_eow":
        lm.seed_ids = lm._choose_seed()
        lm.eow_correction = False
    else:
        lm.seed_ids = lm._choose_seed()
        lm.eow_correction = True


# --------------------------------------------------------------------------- #
def cv_delta_ll(d, y, base_cols, extra_cols, reps=10, folds=10, seed=0):
    """Mean held-out Gaussian log-likelihood gain per word of OLS(base+extra)
    over OLS(base)."""
    rng = np.random.RandomState(seed)
    X0 = np.column_stack([np.ones(len(d))] + [d[c].values for c in base_cols])
    X1 = np.column_stack([X0] + [d[c].values for c in extra_cols])
    yv = d[y].values
    gains = []
    for _ in range(reps):
        idx = rng.permutation(len(d))
        for f in range(folds):
            te = idx[f::folds]
            tr = np.setdiff1d(idx, te)
            ll = []
            for X in (X0, X1):
                beta, *_ = np.linalg.lstsq(X[tr], yv[tr], rcond=None)
                res_tr = yv[tr] - X[tr] @ beta
                sigma2 = res_tr.var()
                res = yv[te] - X[te] @ beta
                ll.append(-0.5 * (np.log(2 * np.pi * sigma2) + res ** 2 / sigma2))
            gains.append((ll[1] - ll[0]).mean())
    g = np.array(gains)
    return g.mean(), g.std(ddof=1) / math.sqrt(len(g))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--potec", default=str(ROOT / "data" / "potec"))
    ap.add_argument("--story", default=str(ROOT / "stimuli" / "story_part1.txt"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out", default=str(ROOT / "docs" / "benchmark_results.csv"))
    a = ap.parse_args()

    subtlex = SubtlexDE()
    texts, rts = load_potec(Path(a.potec))
    story = Path(a.story).read_text(encoding="utf-8")
    story_lay = TextLayout.from_text(story)

    rows = []
    word_tables = {}
    for name in a.models:
        t0 = time.time()
        lm = GPT2LanguageModel(name, device=a.device)
        variants = ["new", "no_eow"] + (["old"] if name == "dbmdz/german-gpt2" else [])
        for variant in variants:
            configure(lm, variant)
            lay = layout_for(story_lay.text, variant)
            st = lm.sweep_layout(lay)
            bits = [-s["logprob"] / math.log(2) for u, s in zip(lay.words, st) if is_word_unit(u)]
            recs = []
            for tid, wdf in texts.items():
                s = potec_word_surprisal(lm, wdf, variant)
                recs.append(wdf.assign(text_id=tid, surprisal=s))
            word_tables[(name, variant)] = pd.concat(recs, ignore_index=True)
            rows.append(dict(model=name, variant=variant, story_bits_per_word=float(np.mean(bits)),
                             story_words=len(bits)))
            print(f"{name:34s} {variant:7s} story {np.mean(bits):6.2f} bits/word  ({time.time()-t0:.0f}s)",
                  flush=True)
        del lm
        try:
            import torch
            if a.device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    # one shared word sample for every model: fixated, all surprisals defined,
    # no symbols/abbreviations/digits, not first or last on a screen line
    base = next(iter(word_tables.values()))[["text_id", "word_index_in_text", "word", "line",
                                             "contains_symbol", "contains_abbreviation"]].copy()
    base["len"] = base["word"].astype(str).str.len()
    base["zipf"] = [subtlex.zipf(w) for w in base["word"].astype(str)]
    for (name, variant), tab in word_tables.items():
        base[f"s|{name}|{variant}"] = tab["surprisal"].values
    base = base.sort_values(["text_id", "word_index_in_text"]).reset_index(drop=True)
    g = base.groupby("text_id")
    for c in ["len", "zipf"] + [c for c in base.columns if c.startswith("s|")]:
        base[f"prev_{c}"] = g[c].shift(1)
    line_pos = base.groupby(["text_id", "line"])["word_index_in_text"]
    base["edge"] = (base["word_index_in_text"] == line_pos.transform("min")) | \
                   (base["word_index_in_text"] == line_pos.transform("max"))
    d = base.merge(rts, on=["text_id", "word_index_in_text"], how="inner")
    scols = [c for c in d.columns if c.startswith("s|") or c.startswith("prev_s|")]
    keep = (~d["edge"]) & (d["contains_symbol"] == 0) & (d["contains_abbreviation"] == 0) & \
           d["word"].astype(str).str.isalpha() & d[scols + ["prev_len", "prev_zipf"]].notna().all(axis=1)
    d = d[keep].copy()
    for c in ["len", "zipf", "prev_len", "prev_zipf"] + scols:
        d[c] = (d[c] - d[c].mean()) / d[c].std()
    print(f"\nPoTeC analysis sample: {len(d)} words (fixated, clean, same set for every model)")

    for r in rows:
        key = f"{r['model']}|{r['variant']}"
        for m in ("FPRT", "TFT"):
            dm = d[d[f"log{m}"].notna()]
            gain, se = cv_delta_ll(dm, f"log{m}", ["len", "zipf", "prev_len", "prev_zipf"],
                                   [f"s|{key}", f"prev_s|{key}"])
            r[f"dLL_{m}"] = gain * 1000      # milli-nats per word
            r[f"dLL_{m}_se"] = se * 1000
            r[f"r_{m}"] = float(np.corrcoef(dm[f"s|{key}"], dm[f"log{m}"])[0, 1])
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print("\nDelta-LL = held-out log-likelihood gain per word (milli-nats) from adding surprisal "
          "to length + SUBTLEX frequency; higher = better fit to human reading times.")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
