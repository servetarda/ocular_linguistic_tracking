"""Checks against the REAL German GPT-2 the pipeline uses (benjamin/gerpt2):
every fast path is compared with a brute-force Hugging Face computation.
Exits with code 77 (reported as SKIPPED by run_all.py) when
torch/transformers or the checkpoint are unavailable. ~1-2 min on CPU.

    python tests/test_real_gpt2.py [cpu|mps|cuda]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

try:
    import torch
    from linguistic_predictors.lexical import GPT2LanguageModel
    from linguistic_predictors.text import TextLayout
    MODEL = "benjamin/gerpt2"
    lm = GPT2LanguageModel(MODEL, device=(sys.argv[1] if len(sys.argv) > 1 else "cpu"))
except Exception as e:  # noqa: BLE001
    print(f"SKIPPED: real GPT-2 not available ({type(e).__name__}: {e})")
    sys.exit(77)

import numpy as np

from linguistic_predictors.cohort import sequential_cohort_stats
from linguistic_predictors.vocab_scorer import FormTrie

failures = []


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(name)


bits = lambda lp: -lp / math.log(2)
TEXT = ("Die Nachricht kam völlig überraschend. Marlene sollte in den Vorruhestand geschickt "
        "werden, weil die Versicherungsagentur ihren Standort in Dortmund aufgeben würde.")
lay = TextLayout.from_text(TEXT)
idl = lm.layout_ids(lay)
dev = lm.device

check("seed is a real document start (the BOS token, else a newline; never a space)",
      lm.seed_ids[0] == lm.tokenizer.bos_token_id or lm.decode_token(lm.seed_ids[0]) == "\n")
check("layout tokens == whole-text tokenisation", idl.ids == lm.seed_ids + lm.tokenizer.encode(lay.text))

sweep = lm.sweep_layout(lay)
check("first word 'Die' is ordinary (< 8 bits; the old space seed gave 15)", bits(sweep[0]["logprob"]) < 8)
dot = lay.words.index(".")
check("'.' after 'überraschend' is expected (< 4 bits; spaced ' .' gave 11)", bits(sweep[dot]["logprob"]) < 4)

# independent single-position scorer agrees with the one-pass sweep
d = max(abs(lm.word_logprob(lay.words[:i], w) - sweep[i]["logprob"]) for i, w in enumerate(lay.words))
check(f"sweep == word_logprob for every unit (max diff {d:.1e} nats)", d < 1e-3)

# custom shared-context attention == Hugging Face forward
sc = lm.scorer
ctx = idl.ids[:15]
st = sc.context_state(ctx)
paths = torch.tensor([[562, 1500, 300], [927, 44, 12000]], device=dev)
with torch.no_grad():
    mine = sc.forward_paths(st, paths, all_positions=True)
    ref = torch.stack([torch.log_softmax(lm.model(torch.tensor([ctx + p.tolist()], device=dev))
                                         .logits[0, -3:].float(), -1) for p in paths])
check(f"forward_paths == HF forward (max diff {float((mine - ref).abs().max()):.1e})",
      float((mine - ref).abs().max()) < 1e-3)
st2 = sc.extend_state(sc.context_state(ctx[:8]), ctx[8:])
check("incremental KV cache == fresh context", float((st2.last_logprobs - st.last_logprobs).abs().max()) < 1e-3)

# batched trie scorer == word_logprob
cands = ["Nachricht", "nachricht", "Haus", "Hausaufgaben", "kam", "Vorruhestand", "über", "Die", "die"]
for i in (0, 3, 9):
    b = lm.score_words_batch(lay.words[:i], cands)
    diff = max(abs(b[c] - lm.word_logprob(lay.words[:i], c)) for c in cands)
    check(f"score_words_batch == word_logprob after {i} units (max diff {diff:.1e})", diff < 1e-3)

# end-of-word correction: a prefix must not inherit its compounds' mass.
# " Vor" is the first token of " Vor"+"ruhe"+"stand", so P(" Vor") counts
# "Vorruhestand" (and every other Vor- word) as if it were "Vor".
ctx = "Marlene sollte in den "
lm_raw = lm.eow_correction
lm.eow_correction = False
raw = lm.score_words_batch(ctx, ["Vor", "Vorruhestand"])
lm.eow_correction = lm_raw
cor = lm.score_words_batch(ctx, ["Vor", "Vorruhestand"])
check("uncorrected: fragment 'Vor' beats the actual word 'Vorruhestand' (the bug)",
      raw["Vor"] > raw["Vorruhestand"])
check("corrected: P(Vor) < P(Vorruhestand)", cor["Vor"] < cor["Vorruhestand"])
check("correction leaves a complete word almost unchanged (< 0.1 bits)",
      abs(bits(cor["Vorruhestand"]) - bits(raw["Vorruhestand"])) < 0.1)

# branch-and-bound pruning stays within its error bound on the cohort
rng = np.random.RandomState(0)
letters = "abcdefghiklmnoprstuvwz"
vocab = sorted({"".join(rng.choice(list(letters), rng.randint(2, 8))) for _ in range(3000)}
               | {"nachricht", "nachts", "nacht", "nase", "haus", "kam", "kamm"})
forms = [v for w in vocab for v in (w, w.capitalize())]
item_of = [f.lower() for f in forms]
units = {w: list(w) for w in vocab}          # letters stand in for phonemes
trie = FormTrie([lm.encode(" " + f) for f in forms])
i = lay.words.index("Nachricht")
st = sc.context_state(idl.ids[:idl.first[i]])
den = lm.position_log_den(st.last_logprobs, idl.lead[i], idl.adjacent[i])
tf = forms.index("Nachricht")
strict = np.array([f[0].lower() == "n" for f in forms])
out = {}
for eps in (0.0, 0.01):
    lp, info = sc.score(st, trie, log_den=den, target_form=tf, strict_forms=strict, eps=eps)
    p = {}
    for it, v in zip(item_of, np.exp(lp)):
        p[it] = p.get(it, 0.0) + v
    out[eps] = sequential_cohort_stats(p, units, list("nachricht"))
ds = max(abs(a - b) for a, b in zip(out[0.0][0], out[0.01][0]))
check(f"pruned cohort surprisal within 0.03 bits of exact (max {ds:.4f})", ds < 0.03)

# word-start nucleus: the smallest set reaching top_p, including the token that crosses it
logits = st.last_logprobs
nuc, _, _ = lm._stats_from_logprobs(logits, 0.9, 1, " ")
mask = lm._space_alpha
p_all = torch.exp(logits)[mask]
p_all = (p_all / p_all.sum()).sort(descending=True).values
cdf = torch.cumsum(p_all, 0)
need = int((cdf < 0.9).sum().item()) + 1
check(f"word-start nucleus keeps the threshold-crossing token ({need} tokens, mass "
      f"{float(cdf[need - 1]):.3f} >= 0.9)", float(cdf[need - 1]) >= 0.9 and len(nuc) <= need)

print()
print(f"Model: {MODEL}")
print("All checks passed." if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
