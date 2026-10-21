"""Summarise meg_compare.py output: gain of each model's surprisal over the controls."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

VARIANTS = ["theirs", "dbmdz", "gerpt2", "gerpt2_raw", "gerpt2_large"]
LABEL = {"theirs": "dbmdz, their implementation", "dbmdz": "dbmdz, our pipeline",
         "gerpt2": "GerPT2, our pipeline", "gerpt2_raw": "GerPT2, no end-of-word correction",
         "gerpt2_large": "GerPT2-large, our pipeline"}

df = pd.concat([pd.read_csv(p) for p in sorted(Path(sys.argv[1]).glob("P*.csv"))])
subjects = sorted(df.subject.unique(), key=lambda s: int(s[1:]))
# region of interest: the 10% of parcels best predicted by the controls-only model
# (chosen without reference to any language model, so it favours none)
base = df.groupby("parcel")["controls"].mean()
roi = base.nlargest(36).index

rows = []
for s in subjects:
    d = df[df.subject == s].set_index("parcel")
    for v in VARIANTS:
        gain = d[v] - d["controls"]
        rows.append(dict(subject=s, model=v, gain_all=gain.mean(), gain_roi=gain.loc[roi].mean()))
g = pd.DataFrame(rows)

print(f"{len(subjects)} participants; controls-only accuracy (Fisher z): "
      f"all parcels {df.controls.mean():.4f}, ROI {df[df.parcel.isin(roi)].controls.mean():.4f}\n")
summary = g.groupby("model")[["gain_all", "gain_roi"]].agg(["mean", "sem"]).loc[VARIANTS]
summary.index = [LABEL[v] for v in VARIANTS]
print("Gain in held-out accuracy from adding surprisal (Fisher z x 1000, mean ± SEM over participants):")
for name, r in summary.iterrows():
    n_pos = int((g[(g.model == [k for k, l in LABEL.items() if l == name][0])].gain_roi > 0).sum())
    print(f"  {name:36s} all parcels {1000 * r[('gain_all', 'mean')]:6.2f} ± {1000 * r[('gain_all', 'sem')]:.2f}"
          f"   ROI {1000 * r[('gain_roi', 'mean')]:6.2f} ± {1000 * r[('gain_roi', 'sem')]:.2f}"
          f"   (ROI gain > 0 in {n_pos}/{len(subjects)})")

print("\nPairwise, ROI (Wilcoxon signed-rank over participants, two-sided; wins = participants where A > B):")
wide = g.pivot(index="subject", columns="model", values="gain_roi")
for a, b in [("gerpt2", "gerpt2_large"), ("gerpt2", "dbmdz"), ("gerpt2", "theirs"),
             ("gerpt2_large", "dbmdz"), ("gerpt2", "gerpt2_raw"), ("dbmdz", "theirs")]:
    diff = wide[a] - wide[b]
    p = wilcoxon(wide[a], wide[b]).pvalue
    print(f"  {LABEL[a]:34s} vs {LABEL[b]:34s} diff {1000 * diff.mean():+6.2f}  wins {int((diff > 0).sum())}/{len(diff)}  p = {p:.3f}")

wide_all = g.pivot(index="subject", columns="model", values="gain_all")
print("\nPairwise, all parcels:")
for a, b in [("gerpt2", "gerpt2_large"), ("gerpt2", "dbmdz"), ("gerpt2", "theirs")]:
    diff = wide_all[a] - wide_all[b]
    p = wilcoxon(wide_all[a], wide_all[b]).pvalue
    print(f"  {LABEL[a]:34s} vs {LABEL[b]:34s} diff {1000 * diff.mean():+6.2f}  wins {int((diff > 0).sum())}/{len(diff)}  p = {p:.3f}")
g.to_csv(Path(sys.argv[1]) / "summary_per_subject.csv", index=False)
