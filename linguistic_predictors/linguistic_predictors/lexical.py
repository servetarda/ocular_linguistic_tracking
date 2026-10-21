"""Lexical level: word probability, surprisal and entropy from a GPT-2 model.

`GPT2LanguageModel` reads the text as written and defines a unit's
probability as P(its tokens) x P(a new unit starts after it) / P(the
previous word ended) -- the end-of-word correction of Pimentel & Meister
(2024) -- with the space-vs-line-break choice conditioned away for
space-led words (see the class docstring). `LanguageModel` is the minimal
interface the toy model (`toy.py`) implements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .text import PUNCT_UNITS, TextLayout, detokenize, is_word_unit  # noqa: F401 (re-exported)


def truncate_nucleus(prob_by_item: Dict[str, float], top_p: float = config.TOP_P,
                      top_k: int = config.TOP_K) -> Dict[str, float]:
    """Nucleus (top-p) truncation with a top-k floor, renormalised
    (Heilbron et al. 2022, SI A.3)."""
    items = sorted(prob_by_item.items(), key=lambda kv: kv[1], reverse=True)
    cum = 0.0
    kept = []
    for i, (w, p) in enumerate(items):
        kept.append((w, p))
        cum += p
        if cum >= top_p and len(kept) >= top_k:
            break
    total = sum(p for _, p in kept)
    if total <= 0:
        return {}
    return {w: p / total for w, p in kept}



@dataclass
class IdLayout:
    """The text as one token sequence (seed token first) plus, per unit, the
    index of its first/last token (-1 if the unit owns no token), whether its
    first token carries the preceding space (`lead`), and whether a WORD ends
    right where its first token starts (`adjacent`, see end-of-word correction)."""
    ids: List[int]
    first: List[int]
    last: List[int]
    lead: List[str]
    adjacent: List[bool]


def _prefix_text(context, next_unit: Optional[str] = None) -> str:
    """The text preceding the next unit, as GPT-2 should read it. `context`
    is either the raw preceding text (a str, used verbatim) or a list of
    preceding units (detokenised naturally)."""
    if isinstance(context, str):
        return context
    ctx = detokenize(list(context))
    if ctx and (next_unit is None or next_unit not in PUNCT_UNITS):
        ctx += " "
    return ctx


def _split_prefix(prefix: str) -> Tuple[str, str]:
    """(context text, lead) -- a trailing space belongs to the NEXT token in
    GPT-2's byte-level BPE ("Ġkam"), so it moves from the context to the lead."""
    ctx = prefix.rstrip(" ")
    lead = " " if (len(ctx) < len(prefix) and ctx and not ctx.endswith("\n")) else ""
    return ctx, lead


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's byte <-> printable-unicode table (same as the reference BPE)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _first_char_class(body: bytes) -> str:
    if not body:
        return "empty"
    b0 = body[0]
    if b0 < 0x80:
        c = chr(b0)
    elif b0 < 0xC0:
        return "cont"                     # middle of a multi-byte character
    else:
        n = 2 if b0 < 0xE0 else 3 if b0 < 0xF0 else 4
        try:
            c = body[:n].decode("utf-8")[0]
        except UnicodeDecodeError:        # token holds only the first byte(s)
            return "alpha" if 0xC3 <= b0 <= 0xC9 else "punct"   # Latin letters vs. „ “ – …
    if c.isalpha():
        return "alpha"
    if c.isdigit():
        return "digit"
    return "space" if c.isspace() else "punct"


def token_classes(token_strings: Sequence[str]) -> Dict[str, np.ndarray]:
    """Classify every vocabulary token by what it can do at a unit boundary:

    - `space_alpha`: " " + letter -- starts a new word after a space
    - `bare_alpha` : letter, no leading whitespace -- starts a word directly
      after a newline, a quote or the text start (and otherwise continues one)
    - `boundary`   : cannot continue a word (leading space/newline,
      punctuation, special tokens) -- its mass is the end-of-word probability
    - `has_space`  : token begins with a space
    - `newline_start`: token begins with a line break

    Classification goes through the raw bytes, so umlauts / ß / „ are
    handled correctly (the byte-level string "Ã¼" is ü).
    """
    dec = {v: k for k, v in _bytes_to_unicode().items()}
    n = len(token_strings)
    out = {k: np.zeros(n, dtype=bool) for k in ("space_alpha", "bare_alpha", "boundary", "has_space",
                                                  "newline_start")}
    for i, t in enumerate(token_strings):
        if not t:
            out["boundary"][i] = True
            continue
        try:
            b = bytes(dec[c] for c in t)
        except KeyError:                  # special token such as <|endoftext|>
            out["boundary"][i] = True
            continue
        space = b[:1] == b" "
        lead_ws = b[:1] in (b" ", b"\n", b"\r", b"\t")
        cls = _first_char_class(b[1:] if space else b)
        out["has_space"][i] = space
        out["newline_start"][i] = b[:1] in (b"\n", b"\r")
        out["space_alpha"][i] = space and cls == "alpha"
        out["bare_alpha"][i] = (not lead_ws) and cls == "alpha"
        out["boundary"][i] = lead_ws or cls in ("punct", "space", "empty")
    return out


class LanguageModel:
    """Interface both the real GPT-2 model and the toy demo model implement.

    `context` arguments are either a list of preceding units (words and
    punctuation marks) or the raw preceding text as a string."""

    def next_word_distribution(self, context_words) -> Dict[str, float]:
        raise NotImplementedError

    def word_logprob(self, context_words, word: str) -> float:
        raise NotImplementedError

    def score_words_batch(
        self, context_words, words: Sequence[str], batch_size: int = 128
    ) -> Dict[str, float]:
        """Natural-log P(word | context_words) for every word in `words`.
        The default just loops `word_logprob`; `GPT2LanguageModel` overrides
        it with the fast shared-context scorer (`vocab_scorer.py`).
        `batch_size` is ignored by this fallback."""
        return {w: self.word_logprob(context_words, w) for w in words}

    def next_word_start_distribution(
        self, context_words, top_p: float = config.TOP_P, top_k: int = config.TOP_K
    ) -> Tuple[Dict[str, float], float, float]:
        """A true next-WORD distribution, its entropy, and the entropy of the
        model's complete untruncated distribution -- see
        `GPT2LanguageModel.next_word_start_distribution`. This default is for
        models whose `next_word_distribution` is ALREADY keyed by whole words
        (the toy bigram model): there is nothing to mask, so the nucleus is
        just the truncated distribution and "full" entropy is the entropy of
        the untruncated one."""
        dist = self.next_word_distribution(context_words)
        nucleus = truncate_nucleus(dist, top_p=top_p, top_k=top_k)
        return nucleus, config.entropy(nucleus.values()), config.entropy(dist.values())

class GPT2LanguageModel(LanguageModel):
    """A GPT-2 checkpoint from the Hugging Face Hub (needs torch + transformers).

    - **Input is the text as written**: punctuation attached, digits, quotes
      and line breaks kept. Tokens are mapped back to units by character
      offsets (`layout_ids`).
    - **Document-start seed**: the tokenizer's BOS token if the model can
      embed it, else a newline (a lone-space seed made "Die" cost 15 bits).
    - **End-of-word correction** (`eow_correction`): P(w) = P(tokens(w)) x
      P(next token starts a new unit) / P(the previous word ended). GPT-2's
      tokens carry the space before a word, so the plain token product is
      the probability that the next word merely STARTS with w.
    - **Line breaks are formatting**: for a space-led word the factor
      [P(space) + P(newline)] / P(space) conditions on "a word follows"
      rather than "a space follows" (identical for every candidate at a
      position, so cohorts are unaffected).
    `eow_correction=False` gives the plain token product of Heilbron et al.
    (2022); sweep results always carry it as `logprob_raw`.
    """

    def __init__(self, model_name: str = "gpt2-xl", device: str = "cpu",
                 eow_correction: bool = True, stride: int = 256):
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        except ImportError as e:
            raise ImportError(
                "GPT2LanguageModel needs `torch` and `transformers` "
                "(`pip install torch transformers`)."
            ) from e
        self._torch = torch
        self.model_name = model_name
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = device
        self.eow_correction = eow_correction
        # rows of the input embedding matrix: any id >= this crashes the
        # forward pass (dbmdz/german-gpt2's declared BOS is exactly this value)
        self._model_vocab = int(self.model.get_input_embeddings().weight.shape[0])
        self.n_positions = int(getattr(self.model.config, "n_positions", config.GPT2_CONTEXT_WINDOW))
        self.stride = max(1, min(stride, self.n_positions - 1))
        self._tok_strings = self.tokenizer.convert_ids_to_tokens(list(range(self._model_vocab)))
        cls = token_classes(self._tok_strings)
        self._space_alpha = torch.tensor(cls["space_alpha"], device=device)
        self._bare_alpha = torch.tensor(cls["bare_alpha"], device=device)
        self._boundary = torch.tensor(cls["boundary"], device=device)
        self._has_space = cls["has_space"]
        self._space_start = torch.tensor(cls["has_space"], device=device)
        self._newline_start = torch.tensor(cls["newline_start"], device=device)
        self.seed_ids = self._choose_seed()
        self._scorer = None

    # -- tokenisation --------------------------------------------------------
    def _choose_seed(self) -> List[int]:
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is not None and 0 <= bos < self._model_vocab:
            return [int(bos)]
        nl = self.encode("\n")
        if len(nl) == 1:
            return nl
        return self.encode(" ")[:1] or [0]

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return [t for t in self.tokenizer.encode(text) if 0 <= t < self._model_vocab]

    def decode_token(self, tok_id: int) -> str:
        return self.tokenizer.convert_tokens_to_string([self._tok_strings[tok_id]])

    def layout_ids(self, layout: TextLayout) -> IdLayout:
        """Tokenise the whole text once and assign each token to the unit whose
        characters it overlaps (via the fast tokenizer's offsets). Tokens that
        overlap no unit -- digits, quotes, dashes, line breaks -- stay in the
        sequence as context but belong to no unit."""
        enc = self.tokenizer(layout.text, add_special_tokens=False, return_offsets_mapping=True)
        tids, offs = list(enc["input_ids"]), list(enc["offset_mapping"])
        if any(not (0 <= t < self._model_vocab) for t in tids):
            keep = [k for k, t in enumerate(tids) if 0 <= t < self._model_vocab]
            tids, offs = [tids[k] for k in keep], [offs[k] for k in keep]
        base = len(self.seed_ids)
        ids = list(self.seed_ids) + tids
        n = len(layout.words)
        first, last = [-1] * n, [-1] * n
        u = 0
        for t, (a, b) in enumerate(offs):
            b = max(b, a + 1)
            while u < n and layout.spans[u][1] <= a:
                u += 1
            if u < n and max(a, layout.spans[u][0]) < min(b, layout.spans[u][1]):
                if first[u] < 0:
                    first[u] = base + t
                last[u] = base + t
        lead, adjacent = [], []
        for i in range(n):
            f = first[i]
            if f < 0:
                lead.append("")
                adjacent.append(False)
                continue
            lead.append(" " if self._has_space[ids[f]] else "")
            start = offs[f - base][0]
            adjacent.append(start > 0 and layout.text[start - 1].isalpha())
        return IdLayout(ids, first, last, lead, adjacent)

    def _logprob_rows(self, ids: Sequence[int]):
        """[len(ids), V] next-token log-probabilities for one forward pass
        (the caller keeps `ids` within the context window)."""
        torch = self._torch
        with torch.no_grad():
            logits = self.model(torch.tensor([list(ids)], device=self.device)).logits[0].float()
        return torch.log_softmax(logits, dim=-1)

    def _boundary_lse(self, logprobs) -> float:
        return float(self._torch.logsumexp(logprobs[self._boundary], dim=-1))

    def _ws_correction(self, logprobs) -> float:
        """log[P(space) + P(newline)] - log P(space) at one position (see the
        class docstring): converts P(" w") into P(w | a word follows)."""
        torch = self._torch
        ls = torch.logsumexp(logprobs[self._space_start], dim=-1)
        ln = torch.logsumexp(logprobs[self._newline_start], dim=-1)
        return float(torch.logaddexp(ls, ln) - ls)

    def position_log_den(self, logprobs, lead: str, adjacent: bool) -> float:
        """Everything that is divided out of a unit's token probability at
        its position: the previous word's end-of-word mass (if a word ends
        right before it) and the line-break-vs-space choice (if it is
        space-led). Zero when `eow_correction` is off."""
        if not self.eow_correction:
            return 0.0
        den = self._boundary_lse(logprobs) if adjacent else 0.0
        if lead == " ":
            den -= self._ws_correction(logprobs)
        return den

    # -- one text, one pass --------------------------------------------------
    def sweep_layout(self, layout: TextLayout, top_p: float = config.TOP_P, top_k: int = config.TOP_K,
                     idl: Optional[IdLayout] = None) -> List[dict]:
        """Score every unit of a text in one left-to-right pass.

        Per unit: `logprob` (natural log; end-of-word corrected if enabled),
        `logprob_raw` (plain token product), `log_eow`, `log_den`,
        `entropy_nucleus` (next-WORD distribution: word-initial tokens only,
        nucleus-truncated; NaN for punctuation units), `entropy_full` (whole
        next-token distribution), `nucleus` ({word: p}). Entropies in
        `config.LOG_BASE` units.

        Texts longer than the model's window are processed in overlapping
        windows: each position is scored in the first window that contains
        it, with at least `n_positions - stride` tokens of real context.
        """
        torch = self._torch
        idl = idl or self.layout_ids(layout)
        ids = idl.ids
        n_ids = len(ids)
        n = len(layout.words)
        need_tok: Dict[int, int] = {}          # position p -> token index j = p+1 to score
        need_b = set()                          # positions needing boundary mass (end of a word)
        need_ent: Dict[int, int] = {}          # position -> unit starting right after it
        for i in range(n):
            f, l = idl.first[i], idl.last[i]
            if f < 0:
                continue
            for j in range(f, l + 1):
                need_tok[j - 1] = j
            need_ent[f - 1] = i
            if is_word_unit(layout.words[i]):
                need_b.add(l)
        need = np.array(sorted(set(need_tok) | need_b | set(need_ent)), dtype=np.int64)

        tok_lp: Dict[int, float] = {}
        blse: Dict[int, float] = {}
        dens: Dict[int, float] = {}
        stats: Dict[int, tuple] = {}
        W = self.n_positions
        start, prev_stop = 0, 0
        while True:
            stop = min(start + W, n_ids)
            resp = need[(need >= prev_stop) & (need < stop)]
            if resp.size:
                lps = self._logprob_rows(ids[start:stop])
                rows = torch.as_tensor(resp - start, device=self.device)
                sub = lps.index_select(0, rows)
                tgt = [need_tok.get(int(p)) for p in resp]
                has_t = [k for k, j in enumerate(tgt) if j is not None]
                if has_t:
                    idx = torch.as_tensor([ids[tgt[k]] for k in has_t], device=self.device)
                    vals = sub[torch.as_tensor(has_t, device=self.device), idx].cpu().double().tolist()
                    for k, v in zip(has_t, vals):
                        tok_lp[tgt[k]] = v
                bl = torch.logsumexp(sub[:, self._boundary], dim=-1).cpu().double().tolist()
                for k, p in enumerate(resp.tolist()):
                    if p in need_b:
                        blse[p] = bl[k]
                    if p in need_ent:
                        i = need_ent[p]
                        stats[p] = self._stats_from_logprobs(sub[k], top_p, top_k, idl.lead[i],
                                                             want_nucleus=is_word_unit(layout.words[i]))
                        dens[p] = self.position_log_den(sub[k], idl.lead[i], idl.adjacent[i])
            if stop >= n_ids:
                break
            prev_stop = stop
            start = stop - (W - self.stride)

        out = []
        for i in range(n):
            f, l = idl.first[i], idl.last[i]
            if f < 0:
                out.append({"logprob": float("nan"), "logprob_raw": float("nan"), "log_eow": 0.0,
                            "log_den": 0.0, "entropy_nucleus": float("nan"),
                            "entropy_full": float("nan"), "nucleus": {}})
                continue
            raw = float(sum(tok_lp[j] for j in range(f, l + 1)))
            log_eow = blse[l] if (is_word_unit(layout.words[i]) and self.eow_correction) else 0.0
            log_den = dens[f - 1]
            lp = raw + log_eow - log_den
            nuc, e_nuc, e_full = stats[f - 1]
            out.append({"logprob": lp, "logprob_raw": raw, "log_eow": log_eow, "log_den": log_den,
                        "entropy_nucleus": e_nuc, "entropy_full": e_full, "nucleus": nuc})
        return out

    # -- single positions (used by pipeline.py and the self-checks) ----------
    def _context_ids(self, ctx_text: str) -> List[int]:
        ids = list(self.seed_ids) + self.encode(ctx_text)
        return ids[-(self.n_positions - 1):]

    def _stats_from_logprobs(self, logprobs, top_p: float, top_k: int, lead: str = " ",
                             want_nucleus: bool = True):
        """One position's log-probability vector -> (word-start nucleus,
        its entropy, full entropy). The word-start set depends on `lead`:
        after a space it is " "+letter tokens; directly after a newline /
        quote / text start it is bare letter tokens."""
        torch = self._torch
        full_nats = float(-(torch.exp(logprobs) * logprobs).sum().item())
        entropy_full = full_nats / math.log(config.LOG_BASE)
        if not want_nucleus:
            return {}, float("nan"), entropy_full
        mask = self._space_alpha if lead == " " else self._bare_alpha
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return {}, 0.0, entropy_full
        p = torch.exp(logprobs)[idx]
        p = p / torch.clamp(p.sum(), min=1e-12)
        s, order = torch.sort(p, descending=True)
        cdf = torch.cumsum(s, dim=0)
        keep = (cdf - s) < top_p          # up to and including the candidate that reaches top_p
        need = min(int(top_k), int((s > 0).sum().item()))
        if int(keep.sum().item()) < need:
            keep = keep.clone()
            keep[:need] = True
        kept_ids = idx[order[keep]].tolist()
        kept_p = s[keep]
        kept_p = (kept_p / torch.clamp(kept_p.sum(), min=1e-12)).tolist()
        nucleus: Dict[str, float] = {}
        for t, pr in zip(kept_ids, kept_p):
            w = self.decode_token(t).strip()
            if w:
                nucleus[w] = nucleus.get(w, 0.0) + float(pr)
        return nucleus, config.entropy(nucleus.values()), entropy_full

    def word_logprob(self, context_words, word: str) -> float:
        """Natural-log P(`word` | context), defined exactly as in
        `sweep_layout` (natural text, document-start seed, end-of-word
        correction if enabled) and computed independently of the batched
        scorer, so `ContextualWordPrior.self_check` can cross-check the two."""
        prefix = _prefix_text(context_words, word)
        layout = TextLayout(prefix + word, [word], [(len(prefix), len(prefix) + len(word))])
        idl = self.layout_ids(layout)
        f, l = idl.first[0], idl.last[0]
        if f < 0:
            return float("-inf")
        ids = idl.ids[:l + 1]
        cut = max(0, len(ids) - self.n_positions)
        ids, f, l = ids[cut:], f - cut, l - cut
        lps = self._logprob_rows(ids)
        raw = float(sum(float(lps[j - 1, ids[j]]) for j in range(f, l + 1)))
        if not self.eow_correction:
            return raw
        if is_word_unit(word):
            raw += self._boundary_lse(lps[l])
        return raw - self.position_log_den(lps[f - 1], idl.lead[0], idl.adjacent[0])

    # -- many candidate words at one position --------------------------------
    @property
    def scorer(self):
        if self._scorer is None:
            from .vocab_scorer import GPT2PrefixScorer
            self._scorer = GPT2PrefixScorer(self.model, self.device, self._boundary)
        return self._scorer

    def score_words_batch(
        self, context_words, words: Sequence[str], batch_size: int = 128
    ) -> Dict[str, float]:
        """Natural-log P(word | context) for a large `words` list, exactly
        (no pruning), with the shared-context trie scorer in `vocab_scorer.py`.
        Same definition as `word_logprob`. `batch_size` is kept for API
        compatibility; batching is by tokens inside the scorer."""
        from .vocab_scorer import FormTrie
        words = list(words)
        ctx, lead = _split_prefix(_prefix_text(context_words))
        st = self.scorer.context_state(self._context_ids(ctx))
        log_den = self.position_log_den(st.last_logprobs, lead, bool(ctx) and ctx[-1].isalpha())
        seqs = [self.encode(lead + w) for w in words]
        trie = FormTrie(seqs)
        form_lp, _ = self.scorer.score(st, trie, log_den=log_den, eow=self.eow_correction, eps=0.0)
        out = {}
        for k, w in enumerate(words):
            lp = float(form_lp[k])
            if self.eow_correction and not is_word_unit(w) and seqs[k]:
                # punctuation candidates carry no end-of-word factor
                lp -= float(self.scorer.score_sequence(st, seqs[k])[1])
            out[w] = lp
        return out

    def context_stream(self, idl: IdLayout, rebuild_margin: int = 200,
                       sentence_starts: Sequence[int] = ()) -> "ContextStream":
        return ContextStream(self, idl, rebuild_margin, sentence_starts)


class ContextStream:
    """Walks a tokenised text left to right keeping GPT-2's KV cache for the
    context before each unit, so each position costs only its new tokens.

    When the window fills up, the context is rebuilt the way Heilbron et al.
    (2022, SI A.2) placed their windows: it starts at a SENTENCE START (never
    mid-sentence), preceded by the document-start seed, and keeps as much of
    the recent text as fits while leaving `rebuild_margin` tokens to grow
    into -- about 800 tokens for GPT-2's 1024 window, i.e. above their
    700-token minimum overlap. `sentence_starts` are token indices into
    `idl.ids`; without them the rebuild falls back to a plain token cut."""

    def __init__(self, lm: GPT2LanguageModel, idl: IdLayout, rebuild_margin: int = 200,
                 sentence_starts: Sequence[int] = ()):
        self.lm = lm
        self.ids = idl.ids
        self.scorer = lm.scorer
        self.margin = rebuild_margin
        self.sentence_starts = sorted(sentence_starts)
        self.state = None
        self.pos = 0

    def state_at(self, pos: int):
        """Context state for `ids[:pos]` (pos must not decrease between calls
        for the incremental path; going back just rebuilds)."""
        if self.state is not None and pos >= self.pos:
            st = self.scorer.extend_state(self.state, self.ids[self.pos:pos])
            if st is not None:
                self.state, self.pos = st, pos
                return st
        seed = list(self.lm.seed_ids)
        keep = max(1, self.scorer.max_ctx - self.margin - len(seed))
        lo = max(0, pos - keep)
        if lo == 0:
            ctx = self.ids[:pos]                  # the whole text so far fits
        else:
            import bisect
            k = bisect.bisect_left(self.sentence_starts, lo)
            if k < len(self.sentence_starts) and pos - self.sentence_starts[k] >= keep // 2:
                lo = self.sentence_starts[k]
            ctx = seed + self.ids[lo:pos]
        self.state = self.scorer.context_state(ctx)
        self.pos = pos
        return self.state
