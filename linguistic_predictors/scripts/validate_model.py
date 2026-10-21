#!/usr/bin/env python3
"""Behavioural validation: does the pipeline reproduce classic psycholinguistic
contrasts? Minimal sentence pairs that differ in ONE critical word, with a
known expected direction at each level (N400-style semantic congruity, word-
class violation, cloze predictability). Loads the real components once.

    python scripts/validate_model.py --mfa-dict /path/to/german_mfa.dict [--device mps]

Prints one line per check and writes docs/validation_results.csv.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from linguistic_predictors.components_de import GERMAN_GPT2_CHECKPOINTS, load_components
from linguistic_predictors.pipeline import PipelineOptions, compute_predictors

# (label, sentence, critical word)
PAIRS = [
    ("congruity", ("Sie trinkt ihren Kaffee immer mit Milch und Zucker.", "Zucker"),
                  ("Sie trinkt ihren Kaffee immer mit Milch und Socken.", "Socken")),
    ("congruity", ("Der Bäcker backt jeden Morgen frisches Brot.", "Brot"),
                  ("Der Bäcker backt jeden Morgen frisches Metall.", "Metall")),
    ("cloze",     ("Nach dem Essen putzt er sich die Zähne.", "Zähne"),
                  ("Nach dem Essen putzt er sich die Schuhe.", "Schuhe")),
    # an open-class violation: spaCy re-tags "laufen" into the slot (ADJ with the
    # full sentence, NOUN with left context only) -- a known limit of tagger-based
    # POS surprisal (the paper's method), reported as NOTE, not as a failure
    ("word class", ("Die Frau kauft einen roten Mantel.", "roten"),
                   ("Die Frau kauft einen laufen Mantel.", "laufen")),
    # a closed-class violation the tagger cannot repair
    ("word class", ("Die Frau kauft einen roten Mantel.", "roten"),
                   ("Die Frau kauft einen oder Mantel.", "oder")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfa-dict", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gpt2-size", default="small", choices=sorted(GERMAN_GPT2_CHECKPOINTS))
    ap.add_argument("--only", default=None, help="run only pairs whose critical violation word is this")
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation_results.csv"))
    a = ap.parse_args()
    comps = load_components(a.mfa_dict, gpt2_model=GERMAN_GPT2_CHECKPOINTS[a.gpt2_size], device=a.device)
    options = PipelineOptions(extra_lexical_models=())
    computed = {}

    rows = []
    for label, good, bad in [p for p in PAIRS if a.only in (None, p[2][1])]:
        for cond, (sent, crit) in (("expected", good), ("violation", bad)):
            if sent not in computed:
                computed[sent] = compute_predictors(sent, comps, options)[0]
            df = computed[sent]
            r = df[df.word == crit].iloc[0]
            ph = ast.literal_eval(r.phon_surprisal_hier) if isinstance(r.phon_surprisal_hier, str) else r.phon_surprisal_hier
            pf = ast.literal_eval(r.phon_surprisal_freq) if isinstance(r.phon_surprisal_freq, str) else r.phon_surprisal_freq
            rows.append(dict(contrast=label, condition=cond, sentence=sent, word=crit,
                             lexical_surprisal=r.lexical_surprisal, pos=r.pos,
                             pos_surprisal=r.pos_surprisal,
                             semantic_prediction_error=r.semantic_prediction_error,
                             phon1_hier=ph[0] if len(ph) else None, phon1_freq=pf[0] if len(pf) else None))
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(res.drop(columns=["sentence"]).round(2).to_string())

    ok = []
    def check(name, cond):
        ok.append(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    for k in range(0, len(res), 2):
        g, b = res.iloc[k], res.iloc[k + 1]
        tag = f"{g.word} vs {b.word}"
        check(f"{tag}: lexical surprisal higher for the {b.condition}", b.lexical_surprisal > g.lexical_surprisal + 2)
        if g.contrast in ("congruity", "cloze"):
            check(f"{tag}: semantic prediction error higher for the {b.condition}",
                  b.semantic_prediction_error > g.semantic_prediction_error)
            check(f"{tag}: first-phoneme surprisal higher with context (hierarchical) for the {b.condition}",
                  b.phon1_hier > g.phon1_hier)
        if g.contrast == "cloze":
            check(f"{tag}: same word class -> POS surprisal similar (< 1 bit apart)",
                  abs(b.pos_surprisal - g.pos_surprisal) < 1)
        if g.contrast == "word class":
            if b.pos == g.pos:
                print(f"[NOTE] {tag}: the tagger re-tagged '{b.word}' as {b.pos} to fit the slot, so "
                      "POS surprisal cannot see this violation (limit of tagger-based POS surprisal)")
            else:
                check(f"{tag}: POS surprisal much higher for the {b.condition} ({b.pos})",
                      b.pos_surprisal > g.pos_surprisal + 2)
    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
