"""Word surprisal of each German GPT-2 model for the Gehmacher et al. (2026) stories.

Each segment's words (their forced-aligned transcript, no punctuation) are
read as one text. A word's surprisal is summed over the pipeline's units
inside it (e.g. both halves of a hyphenated word). Output: one CSV per
segment with a column per model/variant, in bits.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from linguistic_predictors.lexical import GPT2LanguageModel
from linguistic_predictors.text import TextLayout

DATA = REPO / "data/gehmacher2026/data/predictors/word_level_gpt/target"
OUT = Path(sys.argv[1])
MODELS = {"gerpt2": "benjamin/gerpt2", "gerpt2_large": "benjamin/gerpt2-large", "dbmdz": "dbmdz/german-gpt2"}
LN2 = np.log(2)

# every recording (story part x speaker) has its own aligned transcript
texts = {f"{p.parent.name}/{p.stem}": pd.read_csv(p) for p in sorted(DATA.glob("*/*.csv"))}

cols = {s: {} for s in texts}
for tag, name in MODELS.items():
    lm = GPT2LanguageModel(name, device="mps")
    for s, df in texts.items():
        words = [str(w) for w in df["word"]]
        text = " ".join(words)
        layout = TextLayout.from_text(text)
        sweep = lm.sweep_layout(layout)
        starts, pos = [], 0                      # character span of each word
        for w in words:
            starts.append(pos); pos += len(w) + 1
        corrected = np.zeros(len(words)); raw = np.zeros(len(words)); seen = np.zeros(len(words), bool)
        k = 0
        for (a, _), r in zip(layout.spans, sweep):
            while k + 1 < len(words) and starts[k + 1] <= a:
                k += 1
            if np.isfinite(r["logprob"]):
                corrected[k] -= r["logprob"] / LN2
                raw[k] -= r["logprob_raw"] / LN2
                seen[k] = True
        corrected[~seen] = np.nan; raw[~seen] = np.nan
        cols[s][f"{tag}"] = corrected
        if tag == "gerpt2":
            cols[s]["gerpt2_raw"] = raw
    print("scored", name, flush=True)
    del lm

OUT.mkdir(parents=True, exist_ok=True)
for s, df in texts.items():
    res = pd.DataFrame({"word": df["word"]})
    res["theirs"] = df["lexical_surprise"] / LN2     # their dbmdz values (natural log -> bits)
    for c, v in cols[s].items():
        res[c] = v
    (OUT / s).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"{s}.csv", index=False)
    print(s, len(res), res.drop(columns="word").corr(method="spearman").round(2).iloc[0].to_dict())
