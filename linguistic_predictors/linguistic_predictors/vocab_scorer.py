"""Fast, exact scoring of a whole pronunciation-dictionary vocabulary under a
GPT-2 model: P(word | context) for ~100k+ candidate words at every text
position.

Why this module exists
----------------------
The hierarchical phoneme/syllable cohort needs the language model's
probability of every dictionary word at every text position -- ~288k
candidate forms per position. Scoring each candidate by re-running the
context (what a naive batched scorer does) costs context length x
vocabulary size per position.

Three things make this exact and fast:

1. **Shared context, computed once.** The context's keys/values are computed
   once per position; each candidate only runs its OWN tokens, attending to
   that shared cache. The cache is broadcast inside the attention matmul
   rather than copied per candidate (copying it was what made naive KV
   caching memory-bound: ~11 GB of traffic per batch at 600 context tokens).

2. **A token trie.** Candidates that share a token prefix ("ĠVer" + ...)
   share the forward pass for that prefix: 286k surface forms of the German
   MFA dictionary are 712k tokens but only 366k distinct prefixes.

3. **Branch-and-bound with a guaranteed error bound.** With the end-of-word
   correction (below) the words below a trie node are disjoint events, so
   their total probability is at most P(node prefix) -- known before the node
   is expanded. Subtrees are skipped only while the total skipped mass stays
   below `eps` times (a) the target word's own probability, for subtrees
   holding words that share the target's first phoneme (so could enter a
   deeper cohort, whose total is always >= P(target)), or (b) the total
   dictionary mass scored so far, for all other subtrees (they only enter
   the position-0 cohort). Every cohort probability is therefore exact to
   within a relative error of `eps` (default 1%, i.e. < 0.015 bits).

End-of-word correction (Pimentel & Meister 2024, "How to compute the
probability of a word"; Oh & Schuler 2024)
--------------------------------------------------------------------------
GPT-2's tokens carry the space BEFORE a word, so P(" Haus") is really the
probability that the next word STARTS with "Haus" -- it includes "Hausaufgaben",
"Haustür", ... Multiplying by the probability that the following token starts
a new unit (space, punctuation, newline) turns it into P(word == "Haus"):

    P(w | ctx) = P(tokens(w) | ctx) * P(boundary | ctx, w) / P(boundary | ctx)

The denominator (the previous word's own end-of-word factor) is the same for
every candidate at a position. Without the correction, a short word inherits
the probability mass of every longer word it prefixes -- in German, with its
compounds, that systematically inflates short words in the cohort and double
counts the compounds.

Only GPT-2-architecture checkpoints (`GPT2LMHeadModel`) are supported, which
covers every checkpoint in `components_de.GERMAN_GPT2_CHECKPOINTS`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# token trie over candidate surface forms                                     #
# --------------------------------------------------------------------------- #
class FormTrie:
    """Prefix tree over token-id sequences (one per candidate surface form).

    Nodes are numbered depth by depth (node 0 is the root = the context), so
    every depth is a contiguous id range and `paths[d]` holds the full token
    path of every depth-d node as one [n_d, d] array.
    """

    def __init__(self, seqs: Sequence[Sequence[int]]):
        key_to_node: Dict[tuple, int] = {}
        parent, token, depth = [-1], [-1], [0]
        form_node = np.zeros(len(seqs), dtype=np.int64)
        for f, seq in enumerate(seqs):
            node = 0
            for t in seq:
                k = (node, int(t))
                nid = key_to_node.get(k)
                if nid is None:
                    nid = len(parent)
                    key_to_node[k] = nid
                    parent.append(node)
                    token.append(int(t))
                    depth.append(depth[node] + 1)
                node = nid
            form_node[f] = node          # 0 == empty sequence (unscorable)
        del key_to_node

        depth_a = np.asarray(depth, dtype=np.int64)
        order = np.argsort(depth_a, kind="stable")
        new_id = np.empty_like(order)
        new_id[order] = np.arange(len(order))
        par = np.asarray(parent, dtype=np.int64)[order]
        par[1:] = new_id[par[1:]]
        self.parent = par
        self.token = np.asarray(token, dtype=np.int64)[order]
        self.depth = depth_a[order]
        self.form_node = new_id[form_node]
        self.n_nodes = len(self.parent)
        self.n_forms = len(seqs)
        self.max_depth = int(self.depth.max()) if self.n_nodes > 1 else 0
        self.depth_start = np.searchsorted(self.depth, np.arange(self.max_depth + 2))

        # children CSR
        kids = np.argsort(self.parent[1:], kind="stable") + 1
        counts = np.bincount(self.parent[1:], minlength=self.n_nodes)
        self.child_ptr = np.concatenate([[0], np.cumsum(counts)])
        self.child_idx = kids
        self.has_children = counts > 0

        # forms ending at each node (CSR); the root never ends a form
        valid = np.nonzero(self.form_node > 0)[0]
        fo = valid[np.argsort(self.form_node[valid], kind="stable")]
        ecount = np.bincount(self.form_node[valid], minlength=self.n_nodes)
        self.end_ptr = np.concatenate([[0], np.cumsum(ecount)])
        self.end_forms = fo
        self.is_end = ecount > 0

        # subtree height (levels below and including the node), for the
        # no-end-of-word-correction bound (see `score`)
        height = np.ones(self.n_nodes, dtype=np.int64)
        for d in range(self.max_depth, 0, -1):
            lo, hi = self.depth_start[d], self.depth_start[d + 1]
            np.maximum.at(height, self.parent[lo:hi], height[lo:hi] + 1)
        self.height = height

        # full token path of every node, depth by depth
        self.paths: List[np.ndarray] = [np.zeros((1, 0), dtype=np.int64)]
        for d in range(1, self.max_depth + 1):
            lo, hi = self.depth_start[d], self.depth_start[d + 1]
            plo = self.depth_start[d - 1]
            prev = self.paths[d - 1][self.parent[lo:hi] - plo]
            self.paths.append(np.concatenate([prev, self.token[lo:hi, None]], axis=1))

    def level(self, d: int) -> np.ndarray:
        return np.arange(self.depth_start[d], self.depth_start[d + 1])

    def ancestors(self, node: int) -> List[int]:
        out = []
        while node > 0:
            out.append(node)
            node = int(self.parent[node])
        return out[::-1]

    def subtree_any(self, form_flags: np.ndarray) -> np.ndarray:
        """Per node: does any form in its subtree have the flag set?"""
        flag = np.zeros(self.n_nodes, dtype=bool)
        f = np.nonzero(form_flags)[0]
        flag[self.form_node[f]] = True
        for d in range(self.max_depth, 0, -1):
            lo, hi = self.depth_start[d], self.depth_start[d + 1]
            sub = flag[lo:hi]
            if sub.any():
                flag[self.parent[lo:hi][sub]] = True
        return flag


# --------------------------------------------------------------------------- #
# context state + GPT-2 forward over candidate paths                           #
# --------------------------------------------------------------------------- #
@dataclass
class ContextState:
    """Keys/values of the context (per layer, [heads, L, head_dim]) plus the
    next-token log-probabilities after it."""
    kv: list
    length: int
    last_logprobs: "object"                 # torch tensor [V], float32, on device
    _np: Optional[np.ndarray] = field(default=None, repr=False)

    def logprobs_np(self) -> np.ndarray:
        if self._np is None:
            self._np = self.last_logprobs.cpu().double().numpy()
        return self._np


@dataclass
class ScoreInfo:
    n_expanded: int = 0
    pruned_mass_loose: float = 0.0
    pruned_mass_strict: float = 0.0
    scored_mass: float = 0.0
    target_prob: float = float("nan")


def _legacy_past(pkv):
    if hasattr(pkv, "to_legacy_cache"):
        return pkv.to_legacy_cache()
    return pkv


class GPT2PrefixScorer:
    """Scores candidate token sequences against a shared context with a
    GPT-2 `GPT2LMHeadModel` (see module docstring)."""

    def __init__(self, model, device, boundary_mask, max_tokens_per_batch: int = 2048,
                 max_word_tokens: int = 16):
        import torch
        self.torch = torch
        self.model = model
        self.device = device
        cfg = model.config
        if getattr(cfg, "scale_attn_by_inverse_layer_idx", False) or getattr(cfg, "reorder_and_upcast_attn", False):
            raise NotImplementedError("GPT2PrefixScorer: this checkpoint uses a GPT-2 attention "
                                      "variant (inverse-layer scaling / upcast reordering) it does not implement.")
        self.scale = bool(getattr(cfg, "scale_attn_weights", True))
        self.n_head = cfg.n_head
        self.hidden = cfg.n_embd
        self.head_dim = self.hidden // self.n_head
        self.n_positions = cfg.n_positions
        self.max_word_tokens = max_word_tokens
        self.max_ctx = self.n_positions - max_word_tokens - 1
        self.boundary_mask = boundary_mask.to(device)
        self.boundary_idx = torch.nonzero(self.boundary_mask, as_tuple=False).squeeze(1)
        self.max_tokens_per_batch = max_tokens_per_batch

    # -- context -------------------------------------------------------------
    def _state_from_output(self, out) -> ContextState:
        torch = self.torch
        past = _legacy_past(out.past_key_values)
        kv = [(k[0], v[0]) for k, v in past]
        lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return ContextState(kv=kv, length=int(kv[0][0].shape[1]), last_logprobs=lp)

    def context_state(self, ctx_ids: Sequence[int]) -> ContextState:
        torch = self.torch
        ids = list(ctx_ids)[-self.max_ctx:]
        with torch.no_grad():
            out = self.model(torch.tensor([ids], device=self.device), use_cache=True)
        return self._state_from_output(out)

    def extend_state(self, state: ContextState, new_ids: Sequence[int]) -> Optional[ContextState]:
        """Append tokens to a context (incremental KV cache). Returns None if
        the result would not fit the model's window -- rebuild instead."""
        torch = self.torch
        if not new_ids:
            return state
        if state.length + len(new_ids) > self.max_ctx:
            return None
        past = tuple((k[None], v[None]) for k, v in state.kv)
        try:
            from transformers.cache_utils import DynamicCache  # transformers >= 4.36
            past = DynamicCache.from_legacy_cache(past)
        except Exception:
            pass
        with torch.no_grad():
            out = self.model(torch.tensor([list(new_ids)], device=self.device),
                             past_key_values=past, use_cache=True)
        return self._state_from_output(out)

    # -- candidate paths -----------------------------------------------------
    def forward_paths(self, state: ContextState, paths, all_positions: bool = False):
        """Log-probabilities of the next token after each candidate path.

        `paths`: LongTensor [N, d] (all rows the same length). Every row
        attends to the shared context cache (broadcast, never copied) and to
        its own earlier tokens. Returns [N, V] (after the last token) or
        [N, d, V] with `all_positions=True`.
        """
        torch = self.torch
        tr = self.model.transformer
        N, d = paths.shape
        L = state.length
        nh, dh, H = self.n_head, self.head_dim, self.hidden
        pos = torch.arange(L, L + d, device=self.device)
        x = tr.wte(paths) + tr.wpe(pos)[None]
        causal = torch.tril(torch.ones(d, d, dtype=torch.bool, device=self.device))
        scale = 1.0 / math.sqrt(dh) if self.scale else 1.0
        for li, block in enumerate(tr.h):
            a = block.ln_1(x)
            q, k, v = block.attn.c_attn(a).split(H, dim=2)
            q = q.view(N, d, nh, dh).permute(2, 0, 1, 3)            # [nh, N, d, dh]
            k = k.view(N, d, nh, dh).permute(2, 0, 1, 3)
            v = v.view(N, d, nh, dh).permute(2, 0, 1, 3)
            Kc, Vc = state.kv[li]                                   # [nh, L, dh]
            s_ctx = torch.bmm(q.reshape(nh, N * d, dh), Kc.transpose(1, 2)).view(nh, N, d, L)
            s_self = torch.matmul(q, k.transpose(-1, -2))           # [nh, N, d, d]
            s_self = s_self.masked_fill(~causal, torch.finfo(s_self.dtype).min)
            w = torch.softmax(torch.cat([s_ctx, s_self], dim=-1) * scale, dim=-1)
            o = torch.bmm(w[..., :L].reshape(nh, N * d, L), Vc).view(nh, N, d, dh)
            o = o + torch.matmul(w[..., L:], v)
            o = o.permute(1, 2, 0, 3).reshape(N, d, H)
            x = x + block.attn.c_proj(o)
            x = x + block.mlp(block.ln_2(x))
        h = tr.ln_f(x if all_positions else x[:, -1])
        return torch.log_softmax(self.model.lm_head(h).float(), dim=-1)

    def boundary_logmass(self, logprobs):
        """log P(next token starts a new unit), per row."""
        return self.torch.logsumexp(logprobs.index_select(-1, self.boundary_idx), dim=-1)

    def score_sequence(self, state: ContextState, seq: Sequence[int]):
        """(sum of token logprobs, log end-of-word mass after the last token)
        for ONE token sequence -- the target word, scored exactly."""
        torch = self.torch
        seq = list(seq)
        if not seq:
            return float("-inf"), 0.0
        lp0 = state.logprobs_np()
        with torch.no_grad():
            lps = self.forward_paths(state, torch.tensor([seq], device=self.device),
                                     all_positions=True)[0]          # [d, V]
            tail = lps[:-1].gather(1, torch.tensor(seq[1:], device=self.device)[:, None]).squeeze(1)
            eow = float(self.boundary_logmass(lps[-1]))
        return float(lp0[seq[0]]) + float(tail.sum()), eow

    # -- the vocabulary ------------------------------------------------------
    def score(self, state: ContextState, trie: FormTrie, log_den=0.0, eow: bool = True,
              target_form: Optional[int] = None, strict_forms: Optional[np.ndarray] = None,
              eps: float = 0.0, level_frac: float = 0.5, chunk_nodes: int = 4096,
              form_eow: Optional[np.ndarray] = None):
        """Natural-log P(form | context) for every form in `trie`
        (end-of-word corrected if `eow`, minus `log_den`). Forms inside
        pruned subtrees get -inf. With `eps=0` nothing is pruned (exact).

        `log_den`: a scalar, or one value per form (forms at the same
        position can need different denominators, e.g. space-led words vs.
        attached punctuation). `form_eow`: bool per form, False for forms
        that take no end-of-word factor (punctuation); default all True.

        `target_form`: the actually presented word; always scored exactly and
        first, since its probability sets the strict pruning budget.
        `strict_forms`: bool per form -- True for forms that could enter a
        cohort beyond position 0 (share the target's first phoneme).
        """
        torch = self.torch
        info = ScoreInfo()
        n = trie.n_nodes
        lp0 = state.logprobs_np()
        logP = np.full(n, -np.inf)
        if trie.max_depth >= 1:
            d1 = trie.level(1)
            logP[d1] = lp0[trie.token[d1]]
        form_lp = np.full(trie.n_forms, -np.inf)
        den = np.broadcast_to(np.asarray(log_den, dtype=float), (trie.n_forms,))
        feow = np.ones(trie.n_forms, dtype=bool) if form_eow is None else np.asarray(form_eow, dtype=bool)
        expanded = np.zeros(n, dtype=bool)
        node_strict = trie.subtree_any(strict_forms) if strict_forms is not None else np.zeros(n, bool)
        lb_mass = 0.0            # exact dictionary mass scored so far (lower bound on the cohort-0 total)
        used_loose = used_strict = 0.0
        # without the end-of-word correction the forms below a node are nested,
        # not disjoint: their summed probability is bounded by P(node) x height.
        # Form scores are divided by exp(den) -- the bound must be on the SAME
        # scale as the budgets (which are sums of form scores), so scale it by
        # the largest 1/exp(den) any form can get.
        bound_mult = (np.ones(n) if eow else trie.height.astype(float)) * math.exp(-float(den.min()))

        def record_forms(nodes, eow_lse):
            nonlocal lb_mass
            for node, e in zip(nodes, eow_lse):
                lo, hi = trie.end_ptr[node], trie.end_ptr[node + 1]
                if lo == hi:
                    continue
                fs = trie.end_forms[lo:hi]
                vals = logP[node] + np.where(feow[fs], e if eow else 0.0, 0.0) - den[fs]
                form_lp[fs] = vals
                lb_mass += float(np.exp(np.minimum(vals, 700.0)).sum())

        def apply_expansion(nodes, lps):
            """lps: torch [len(nodes), V] next-token logprobs after each node."""
            e_lse = self.boundary_logmass(lps).cpu().double().numpy() if eow else np.zeros(len(nodes))
            record_forms(nodes, e_lse)
            # children
            starts = trie.child_ptr[nodes]
            ends = trie.child_ptr[nodes + 1]
            cnt = ends - starts
            if cnt.sum():
                rows = np.repeat(np.arange(len(nodes)), cnt)
                kids = np.concatenate([trie.child_idx[s:e] for s, e in zip(starts, ends) if e > s])
                vals = lps[torch.as_tensor(rows, device=self.device),
                           torch.as_tensor(trie.token[kids], device=self.device)].cpu().double().numpy()
                logP[kids] = logP[nodes][rows] + vals
            expanded[nodes] = True
            info.n_expanded += len(nodes)

        def expand(nodes, d):
            if len(nodes) == 0:
                return
            per = max(1, self.max_tokens_per_batch // d)
            lo_d = trie.depth_start[d]
            for s in range(0, len(nodes), per):
                chunk = nodes[s:s + per]
                paths = torch.as_tensor(trie.paths[d][chunk - lo_d], device=self.device)
                with torch.no_grad():
                    lps = self.forward_paths(state, paths)
                apply_expansion(chunk, lps)

        # 1) the presented word first: exact, and its path comes for free
        p_target = 0.0
        if target_form is not None and trie.form_node[target_form] > 0:
            anc = trie.ancestors(int(trie.form_node[target_form]))
            seq = trie.token[anc]
            with torch.no_grad():
                lps = self.forward_paths(state, torch.as_tensor(seq[None], device=self.device),
                                         all_positions=True)[0]
            for k, node in enumerate(anc):     # in order: each node's logP comes from its parent
                apply_expansion(np.asarray([node]), lps[k:k + 1])
            p_target = math.exp(form_lp[target_form]) if form_lp[target_form] > -700 else 0.0
            info.target_prob = p_target

        # 2) level by level, branch and bound
        for d in range(1, trie.max_depth + 1):
            lvl = trie.level(d)
            F = lvl[np.isfinite(logP[lvl]) & ~expanded[lvl]]
            if not eow:
                # a leaf needs no forward pass when there is no end-of-word factor
                leaf = ~trie.has_children[F]
                if leaf.any():
                    record_forms(F[leaf], np.zeros(int(leaf.sum())))
                    expanded[F[leaf]] = True
                    F = F[~leaf]
            if F.size == 0:
                continue
            F = F[np.argsort(-logP[F], kind="stable")]           # descending P
            frac = 1.0 if d == trie.max_depth else level_frac
            while True:
                if eps <= 0:
                    expand(F, d)
                    break
                P = np.exp(logP[F]) * bound_mult[F]
                strict = node_strict[F]
                avail_l = frac * max(0.0, eps * max(lb_mass, p_target) - used_loose)
                avail_s = frac * max(0.0, eps * p_target - used_strict)
                prunable = np.zeros(F.size, dtype=bool)
                s_idx = np.nonzero(strict)[0][::-1]                 # ascending P
                cs = np.cumsum(P[s_idx])
                k_s = int(np.count_nonzero((cs <= avail_s) & (cs <= avail_l)))
                spent_s = float(cs[k_s - 1]) if k_s else 0.0
                prunable[s_idx[:k_s]] = True
                l_idx = np.nonzero(~strict)[0][::-1]
                cl = np.cumsum(P[l_idx])
                k_l = int(np.count_nonzero(cl <= avail_l - spent_s))
                spent_l = float(cl[k_l - 1]) if k_l else 0.0
                prunable[l_idx[:k_l]] = True
                keep = F[~prunable]
                if keep.size == 0:
                    used_strict += spent_s
                    used_loose += spent_s + spent_l
                    info.pruned_mass_strict += spent_s
                    info.pruned_mass_loose += spent_l
                    break
                chunk = keep[:chunk_nodes]
                expand(chunk, d)
                F = F[~np.isin(F, chunk)]
                if F.size == 0:
                    break

        info.scored_mass = lb_mass
        # On Apple GPUs, PyTorch keeps freed blocks for reuse. The batch shapes
        # change with the context length, so this store grows with every
        # position (10 GB after 100 words) until a 16 GB machine swaps and the
        # run is killed. Release it once per position.
        if str(self.device).startswith("mps"):
            self.torch.mps.empty_cache()
        return form_lp, info
