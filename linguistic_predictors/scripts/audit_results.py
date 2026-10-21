"""Scan a results/ folder for values that cannot be right.

    python scripts/audit_results.py results/

Checks every numeric column for NaN-only / constant / negative / infinite
values, for anything pinned at the EPSILON ceiling (a manufactured constant,
not a measurement), for entropies above log2(vocab), and for non-contiguous
per-phoneme indices. Run it on real output before trusting it: a clean report
is not proof of correctness, but every anomaly it prints is worth explaining.
"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

d = Path(sys.argv[1])
issues = []
def flag(msg): issues.append(msg); print("  !! " + msg)

CEIL = 39.863137138648355          # config.surprisal's EPSILON ceiling, in bits

print(f"=== {d} ===")
w = pd.read_csv(d / "words.csv")
print(f"words.csv: {len(w)} rows, {len(w.columns)} columns")

num = ["lexical_surprisal","lexical_entropy","entropy_full","unigram_surprisal",
       "pos_surprisal","pos_entropy","semantic_prediction_error","semantic_distance"]
for c in [c for c in num if c in w.columns]:
    s = pd.to_numeric(w[c], errors="coerce")
    live = s.dropna()
    if len(live) == 0:
        flag(f"{c}: every value is NaN"); continue
    if np.isinf(live).any():          flag(f"{c}: contains infinity")
    if (live < -1e-9).any():          flag(f"{c}: negative values (min {live.min():.4f})")
    if live.nunique() == 1 and len(live) > 3:
        flag(f"{c}: constant at {live.iloc[0]:.4f} across all rows")
    hits = int((np.abs(live - CEIL) < 1e-6).sum())
    if hits:                          flag(f"{c}: {hits} value(s) pinned at the EPSILON ceiling {CEIL:.3f}")
    print(f"   {c:<28} n={len(live):>4}  min={live.min():>9.4f}  "
          f"median={live.median():>9.4f}  max={live.max():>9.4f}  NaN={s.isna().sum()}")

# entropy must never exceed log2(vocab); nucleus entropy <= full entropy is NOT
# guaranteed (different supports) but a huge gap is suspicious
if {"lexical_entropy","entropy_full"} <= set(w.columns):
    a = pd.to_numeric(w.lexical_entropy, errors="coerce")
    b = pd.to_numeric(w.entropy_full, errors="coerce")
    if (a > math.log2(50265) + 1e-6).any(): flag("lexical_entropy exceeds log2(vocab)")
    if (b > math.log2(50265) + 1e-6).any(): flag("entropy_full exceeds log2(vocab)")

# per-phoneme tables
for name, idx in (("phonemes.csv","phon_index"), ("syllables.csv","syll_index")):
    f = d / name
    if not f.exists(): continue
    t = pd.read_csv(f)
    print(f"\n{name}: {len(t)} rows")
    if len(t) == 0: flag(f"{name} is empty"); continue
    # index must restart at 0 for each word and be contiguous
    for (wp, _), g in t.groupby([ "word_position", "word"]):
        v = list(g[idx])
        if v != list(range(len(v))):
            flag(f"{name}: word at position {wp} has non-contiguous {idx}: {v}"); break
    for c in [c for c in t.columns if "surprisal" in c or "entropy" in c]:
        s = pd.to_numeric(t[c], errors="coerce").dropna()
        if len(s) == 0: continue
        if (s < -1e-9).any(): flag(f"{name}:{c} negative")
        if np.isinf(s).any(): flag(f"{name}:{c} infinite")
        zero = int((s.abs() < 1e-9).sum())
        print(f"   {c:<24} min={s.min():>8.4f} median={s.median():>8.4f} "
              f"max={s.max():>9.4f}  exact-zero={zero}/{len(s)} ({100*zero/len(s):.0f}%)")

print("\n" + ("NO ANOMALIES FOUND" if not issues else f"{len(issues)} ANOMALIES"))
