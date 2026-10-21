#!/usr/bin/env python3
"""Build docs/methods_report_heilbron_german.pdf from the saved results.

    python docs/build_methods_report.py [--validation docs/validation_results.csv]

Every number in the tables and in the figure is read from the files at build
time: results/words.csv, phonemes.csv, syllables.csv, statistics.csv,
run_config.json, stimuli/violations.csv, docs/benchmark_results.csv,
docs/listening_benchmark_results.csv and the minimal-pair CSV written by
scripts/validate_model.py. The prose around them is written here.

The page is written as HTML (docs/methods_report_heilbron_german.html is
kept next to the PDF only with --keep-html) and printed to PDF with headless
Google Chrome. Needs numpy, pandas and scipy (nlp_env).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


# ---------------------------------------------------------------- helpers

def esc(s) -> str:
    return html.escape(str(s))


def code(s) -> str:
    return f"<code>{esc(s)}</code>"


def num(x, d=2) -> str:
    return f"{x:,.{d}f}".replace("-", "−")


def pfmt(p: float) -> str:
    """p-value as text: 0.004, 1.8 × 10<sup>−4</sup>, < 10<sup>−15</sup>."""
    if p < 1e-15:
        return "&lt; 10<sup>−15</sup>"
    if p >= 0.001:
        return f"{p:.3f}" if p < 0.01 else f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10<sup>−{int(e[1:])}</sup>"


def table(head, rows, cls="", caption="", widths=None) -> str:
    cols = "".join(f'<col style="width:{w}">' for w in widths) if widths else ""
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f'<p class="caption">{caption}</p>' if caption else ""
    return f'<div class="tablewrap"><table class="{cls}"><colgroup>{cols}</colgroup><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>{cap}</div>'


def ul(items) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def tag(kind) -> str:
    return f'<span class="tag {kind}">{kind}</span>'


def git_version() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                                "linguistic_predictors", "scripts", "run_story.py", "results"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"git commit {code(sha)}" + (" with uncommitted changes" if dirty else "")
    except Exception:
        return "git version unknown"


# ---------------------------------------------------------------- data

def load(validation_csv: Path):
    d = {}
    units = pd.read_csv(RESULTS / "words.csv")
    d["units"] = units
    d["words"] = words = units[~units["is_punctuation"]].copy()
    d["phon"] = phon = pd.read_csv(RESULTS / "phonemes.csv")
    d["syll"] = pd.read_csv(RESULTS / "syllables.csv")
    d["stats"] = pd.read_csv(RESULTS / "statistics.csv")
    d["cfg"] = json.loads((RESULTS / "run_config.json").read_text())
    d["bench"] = pd.read_csv(DOCS / "benchmark_results.csv")
    d["listen"] = pd.read_csv(DOCS / "listening_benchmark_results.csv")
    d["pairs"] = pd.read_csv(validation_csv)
    viol = pd.read_csv(ROOT / "stimuli" / "violations.csv")

    first = phon[phon["phon_index"] == 0].set_index("word_position")["phon_surprisal_hier"]
    words["phon1"] = words["position"].map(first)

    # the anomalous words: the occurrence of `word` preceded by `preceded_by`
    text = list(units["word"])
    rows = []
    for w, before in zip(viol["word"], viol["preceded_by"]):
        b = before.split()
        hits = [i for i, x in enumerate(text) if x == w and text[i - len(b):i] == b]
        assert len(hits) == 1, (w, before)
        rows.append(units.index[hits[0]])
    d["anom"] = pd.Index(rows)
    return d


def sentence_end(units, idx, n=8):
    """The last n units of the sentence up to and including row idx."""
    s = units.loc[idx, "sentence"]
    sub = units[(units["sentence"] == s) & (units.index <= idx)]
    toks = list(sub["word"])
    out = ""
    tail = toks[-n:]
    while len(tail) > 1 and tail[0] in ".,;:!?":
        tail = tail[1:]
    for t in tail:
        out += t if (t in ".,;:!?" or not out) else " " + t
    return ("… " if len(toks) > n else "") + out


# ---------------------------------------------------------------- figure

def figure_svg(d) -> str:
    words, anom = d["words"], d["anom"]
    normal = words.drop(index=anom)
    stats = d["stats"].set_index("test")
    panels = [("lexical_surprisal", "Lexical (bits)", "lexical_surprisal"),
              ("semantic_prediction_error", "Semantic (cosine distance)", "semantic_prediction_error"),
              ("phon1", "First phoneme (bits)", "first_phoneme_surprisal"),
              ("pos_surprisal", "Part of speech (bits)", "pos_surprisal")]
    W, H, pw = 680, 230, 150
    top, bot = 40, 200
    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica Neue, Arial" font-size="10">']
    rng = np.random.default_rng(1)
    for k, (col, title, test) in enumerate(panels):
        x0 = 40 + k * (pw + 20)
        means = normal.groupby("word")[col].mean()
        a = words.loc[anom, col].to_numpy(float)
        c = words.loc[anom, "word"].map(means).to_numpy(float)
        ymax = np.nanmax(np.r_[a, c]) * 1.08
        step = next(s for s in (0.2, 2, 5, 10, 20) if ymax / s <= 7)
        y = lambda v: bot - (v / ymax) * (bot - top)
        p = stats.loc[f"1a. anomalous > same word in normal sentence: {test}", "p_holm"]
        if p < 0.001:
            m, e = f"{p:.1e}".split("e")
            ptxt = f"{m} \u00d7 10" + "\u207b" + str(int(e[1:])).translate(str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"))
        else:
            ptxt = f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
        out.append(f'<text x="{x0}" y="14" fill="#222">{esc(title)}</text>'
                   f'<text x="{x0}" y="27" fill="#222">p = {ptxt}</text>')
        t = 0.0
        while t <= ymax + 1e-9:
            out.append(f'<line x1="{x0}" x2="{x0 + pw - 10}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="#e6e6e6"/>'
                       f'<text x="{x0 - 4}" y="{y(t) + 3:.1f}" text-anchor="end" fill="#555">{t:g}</text>')
            t = round(t + step, 6)
        xn, xa = x0 + 35, x0 + pw - 45
        jn = rng.uniform(-6, 6, len(a))
        ja = rng.uniform(-6, 6, len(a))
        for i in range(len(a)):
            if np.isnan(a[i]) or np.isnan(c[i]):
                continue
            out.append(f'<line x1="{xn + jn[i]:.1f}" y1="{y(c[i]):.1f}" x2="{xa + ja[i]:.1f}" y2="{y(a[i]):.1f}" stroke="#bbb" stroke-width="0.8"/>')
        for i in range(len(a)):
            if not np.isnan(c[i]):
                out.append(f'<circle cx="{xn + jn[i]:.1f}" cy="{y(c[i]):.1f}" r="2.6" fill="#2a7ad5"/>')
            if not np.isnan(a[i]):
                out.append(f'<circle cx="{xa + ja[i]:.1f}" cy="{y(a[i]):.1f}" r="2.6" fill="#ee6a2b"/>')
        for xm, v in ((xn, np.nanmedian(c)), (xa, np.nanmedian(a))):
            out.append(f'<line x1="{xm - 11}" x2="{xm + 11}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#000" stroke-width="2.2"/>')
        out.append(f'<line x1="{x0}" x2="{x0 + pw - 10}" y1="{bot}" y2="{bot}" stroke="#888"/>'
                   f'<text x="{xn}" y="{bot + 14}" text-anchor="middle" fill="#333">normal</text>'
                   f'<text x="{xa}" y="{bot + 14}" text-anchor="middle" fill="#333">anomalous</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- sections

def build_html(d) -> str:
    units, words, phon, syll, cfg = d["units"], d["words"], d["phon"], d["syll"], d["cfg"]
    st = d["stats"]
    S = lambda prefix: st[st["test"].str.startswith(prefix)].iloc[0]
    n_sent = units["sentence"].nunique()
    n_units, n_words = len(units), len(words)
    n_punct = n_units - n_words
    n_phon, n_syll = len(phon), len(syll)
    vocab, forms = cfg["candidate_vocab_size"], cfg["candidate_forms"]
    cov = cfg["pronunciation_coverage"]
    zero_pct = 100 * (phon["phon_surprisal_hier"] == 0).mean()
    hours = cfg["seconds"] / 3600
    lex1a = S("1a. anomalous > same word in normal sentence: lexical")
    sem1a = S("1a. anomalous > same word in normal sentence: semantic")
    ph1a = S("1a. anomalous > same word in normal sentence: first")
    pos1a = S("1a. anomalous > same word in normal sentence: pos")
    t4, t5 = S("4."), S("5.")
    med = lambda r: r["detail"].replace("median ", "").replace(" (cosine distance)", "")

    # correlations and VIF (same predictor set as scripts/story_statistics.py)
    corr_cols = [("lexical_surprisal", "Lex. surpr."), ("lexical_entropy", "Lex. entr."),
                 ("pos_surprisal", "POS surpr."), ("semantic_prediction_error", "Sem. error"),
                 ("semantic_distance", "Sem. dist."), ("unigram_surprisal", "Unigram")]
    cc = [c for c, _ in corr_cols]
    long_names = {"lexical_surprisal": "lexical surprisal", "lexical_entropy": "lexical entropy", "pos_surprisal": "POS surprisal",
                  "semantic_prediction_error": "semantic prediction error", "semantic_distance": "semantic distance",
                  "unigram_surprisal": "unigram surprisal"}
    corr = words[cc].corr()
    complete = words[cc].dropna()
    vif = pd.Series(np.diag(np.linalg.inv(complete.corr().to_numpy())), index=cc)
    off = corr.where(~np.eye(len(cc), dtype=bool)).abs()
    max_r = off.max().max()
    a_, b_ = off.stack().idxmax()
    r_phon1 = words[["lexical_surprisal", "phon1"]].corr().iloc[0, 1]
    r_lex_uni = corr.loc["lexical_surprisal", "unigram_surprisal"]
    pos_entropy_r = words[["pos_entropy", "lexical_entropy"]].corr().iloc[0, 1]

    # listening benchmark
    L = d["listen"]
    piv = {c: L.pivot(index="subject", columns="model", values=c) * 1000 for c in ("gain_all", "gain_roi")}
    n_part = len(piv["gain_all"])

    def lrow(model, label):
        cells = [label]
        for c in ("gain_all", "gain_roi"):
            v = piv[c][model]
            cells.append(f"{num(v.mean())} ± {num(v.std(ddof=1) / np.sqrt(len(v)))}")
        cells += [f"{(piv['gain_all'][model] > 0).sum()}/{n_part}", f"{(piv['gain_roi'][model] > 0).sum()}/{n_part}"]
        return cells

    def cmp(a, b, c):
        v = piv[c]
        return (v[a] > v[b]).sum(), wilcoxon(v[a], v[b]).pvalue

    w_lg_roi, p_lg_roi = cmp("gerpt2", "gerpt2_large", "gain_roi")
    w_lg_all, p_lg_all = cmp("gerpt2", "gerpt2_large", "gain_all")
    w_db_roi, p_db_roi = cmp("gerpt2", "dbmdz", "gain_roi")
    w_db_all, p_db_all = cmp("gerpt2", "dbmdz", "gain_all")
    w_th_roi, p_th_roi = cmp("gerpt2", "theirs", "gain_roi")
    w_lgdb_roi, _ = cmp("gerpt2_large", "dbmdz", "gain_roi")
    rho_repro = piv["gain_all"][["dbmdz", "theirs"]].corr(method="spearman").iloc[0, 1]

    # reading benchmark
    B = d["bench"]
    sizes = {"benjamin/gerpt2": "124 M", "benjamin/gerpt2-large": "774 M", "stefan-it/german-gpt2-larger": "124 M",
             "malteos/gpt2-xl-wechsel-german": "1.5 B", "dbmdz/german-gpt2": "124 M"}
    brows = []
    new = B[B["variant"] == "new"].sort_values("dLL_FPRT", ascending=False)
    for _, r in new.iterrows():
        brows.append([code(r["model"]), sizes.get(r["model"], ""), num(r["story_bits_per_word"]),
                      f"{num(r['dLL_FPRT'], 1)} ± {num(r['dLL_FPRT_se'], 1)}",
                      f"{num(r['dLL_TFT'], 1)} ± {num(r['dLL_TFT_se'], 1)}"])
    old = B[B["variant"] == "old"].iloc[0]
    brows.append([code(old["model"]) + ", original implementation", "124 M", num(old["story_bits_per_word"]),
                  f"{num(old['dLL_FPRT'], 1)} ± {num(old['dLL_FPRT_se'], 1)}",
                  f"{num(old['dLL_TFT'], 1)} ± {num(old['dLL_TFT_se'], 1)}"])
    g_new = B[(B["model"] == "benjamin/gerpt2") & (B["variant"] == "new")].iloc[0]
    g_raw = B[(B["model"] == "benjamin/gerpt2") & (B["variant"] == "no_eow")].iloc[0]
    db_new = B[(B["model"] == "dbmdz/german-gpt2") & (B["variant"] == "new")].iloc[0]

    # minimal pairs
    P = d["pairs"]
    prow = []
    for k in range(0, len(P), 2):
        g, b = P.iloc[k], P.iloc[k + 1]
        sem = "—" if pd.isna(b["semantic_prediction_error"]) else \
            f"{num(g['semantic_prediction_error'])} → {num(b['semantic_prediction_error'])}"
        prow.append([esc(g["contrast"]), f"{esc(g['word'])} → <b>{esc(b['word'])}</b>",
                     f"{num(g['lexical_surprisal'], 1)} → {num(b['lexical_surprisal'], 1)}",
                     f"{esc(g['pos'])}/{esc(b['pos'])}: {num(g['pos_surprisal'], 1)} → {num(b['pos_surprisal'], 1)}",
                     sem, f"{num(g['phon1_hier'], 1)} → {num(b['phon1_hier'], 1)}"])
    laufen = P[P["word"] == "laufen"].iloc[0]

    # anomalous words table
    normal = words.drop(index=d["anom"])
    counts = normal.groupby("word").size()
    arows = []
    for i, idx in enumerate(d["anom"], 1):
        w = units.loc[idx, "word"]
        ctx = sentence_end(units, idx)
        ctx = ctx[: ctx.rfind(w)] + f"<b>{esc(w)}</b>"
        arows.append([str(i), ctx, str(counts.get(w, 0)), num(units.loc[idx, "lexical_surprisal"], 1)])

    # descriptive statistics
    def drow(label, per, s):
        s = s.dropna()
        return [label, per, f"{len(s):,}", num(s.mean()), num(s.std()), num(s.median()), num(s.max())]
    drows = [drow("Lexical surprisal", "word", words["lexical_surprisal"]),
             drow("Lexical entropy", "word", words["lexical_entropy"]),
             drow("POS surprisal", "word", words["pos_surprisal"]),
             drow("POS entropy", "word", words["pos_entropy"]),
             drow("Semantic prediction error (distance)", "content word", words["semantic_prediction_error"]),
             drow("Semantic distance (Broderick; distance)", "word", words["semantic_distance"]),
             drow("Unigram surprisal", "word", words["unigram_surprisal"]),
             drow("Phoneme surprisal, hierarchical", "phoneme", phon["phon_surprisal_hier"]),
             drow("Phoneme entropy, hierarchical", "phoneme", phon["phon_entropy_hier"]),
             drow("Phoneme surprisal, single-level", "phoneme", phon["phon_surprisal_freq"]),
             drow("Phoneme entropy, single-level", "phoneme", phon["phon_entropy_freq"]),
             drow("Syllable surprisal, hierarchical", "syllable", syll["syll_surprisal_hier"]),
             drow("Syllable entropy, hierarchical", "syllable", syll["syll_entropy_hier"]),
             drow("Syllable surprisal, single-level", "syllable", syll["syll_surprisal_freq"]),
             drow("Syllable entropy, single-level", "syllable", syll["syll_entropy_freq"])]
    n_content = int(words["semantic_prediction_error"].notna().sum())

    # statistics table
    labels = {
        "1a": "Anomalous &gt; same word in normal sentence (Wilcoxon, paired)",
        "1b": "Anomalous &gt; other nouns (Mann–Whitney U)",
    }
    measure = {"lexical_surprisal": "lexical surprisal", "pos_surprisal": "POS surprisal",
               "semantic_prediction_error": "semantic prediction error",
               "first_phoneme_surprisal": "first-phoneme surprisal (hierarchical)"}
    other = {
        "2.": "Lexical surprisal rises with unigram surprisal (Spearman)",
        "3.": "Content &gt; function words: lexical surprisal (Mann–Whitney U)",
        "4.": "Hierarchical &lt; single-level phoneme surprisal, one mean per word (Wilcoxon)",
        "5.": "First phoneme &gt; later phonemes of the same word, one pair per word (Wilcoxon)",
    }
    srows = []
    for _, r in st.iterrows():
        t = r["test"]
        if t[:2] in labels:
            lab = f"{labels[t[:2]]}: {measure[t.split(': ')[1]]}"
        else:
            lab = other[t.split(" ")[0]]
        detail = esc(r["detail"]).replace(" (cosine distance)", "").replace("rho", "ρ").replace(" vs ", " vs. ")
        n_txt = " vs. ".join(f"{int(x):,}" for x in str(r["n"]).split(" vs "))
        srows.append([lab, n_txt, detail, pfmt(r["p_holm"]),
                      "yes" if r["significant"] else "no"])

    # correlation table
    crows = []
    for i, (c, lab) in enumerate(corr_cols):
        cells = [lab] + [num(corr.loc[c, c2]) if j < i else ("—" if j == i else "")
                         for j, (c2, _) in enumerate(corr_cols)]
        crows.append(cells + [num(vif[c])])

    figure = figure_svg(d)
    version = git_version()
    today = dt.date.today().strftime("%-d %B %Y")

    # ------------------------------------------------------------ text
    H = []
    H.append(f"""
<header>
<p class="kicker">METHODS REPORT</p>
<h1>Word-by-word linguistic predictability for a German spoken story</h1>
<p class="subtitle">How the predictors are computed, how they were checked, and how the method relates to Heilbron et al. (2022)</p>
<p class="meta">Software: {code('linguistic_predictors')}, {version} &nbsp;·&nbsp; Stimulus: {code('stimuli/story_part1.txt')}
({n_sent} sentences, {n_words:,} words) &nbsp;·&nbsp; Results: {code('results/')} &nbsp;·&nbsp; {today}
(updated after an outside review: lexical entropy, part of speech, alignment checks and statistics)</p>
</header>

<div class="summary">
<p><b>Summary.</b> For every word of the story, the pipeline estimates how predictable it was at five levels:</p>
{ul(["the word itself (lexical);", "its part of speech (syntactic);", "its meaning (semantic);",
     "each of its phonemes;", "each of its syllables."])}
<p>The lexical, part-of-speech, phoneme and syllable levels each give a <b>surprisal</b> and an <b>entropy</b>, in bits.
The semantic level gives a <b>prediction error</b>: a distance between the expected and the actual meaning, not a value in bits,
and it has no entropy (as in Heilbron et al.).</p>
<p>The method is that of Heilbron et al. (2022), carried over to German:</p>
{ul(["A German GPT-2 model (GerPT2) provides the predictions. It was chosen because it best predicted German reading times. "
     f"A small check on independent German listening data (MEG, {n_part} participants) is consistent with keeping it, but is only preliminary.",
     "German resources replace the English ones.",
     "Three technical refinements compute the paper's own quantities more exactly (§4). The paper-style lexical surprisal is reported alongside.",
     "The syllable level is an extension."])}
<p>For the full story ({n_words:,} words, {n_phon:,} phonemes, {n_syll:,} syllables), the predictors behave as expected.
The 15 anomalous words are much less predictable than the same words in their normal sentences:</p>
{ul([f"lexically: median {med(lex1a)};", f"semantically: median {med(sem1a)} (distance);",
     f"at the first phoneme: median {med(ph1a)}."])}
<p>They are not less predictable in part of speech, because they are nouns in noun positions.
These are descriptive checks on one story, not evidence about listeners.</p>
</div>

<p>This report has three uses:</p>
{ul(["<b>Explanation:</b> §1–2 explain the model in plain terms.",
     "<b>For the paper:</b> §3 is a draft Methods text that can be adapted for a paper. §4 lists every difference from Heilbron et al. (2022), with reasons.",
     "<b>Checks and use:</b> §5 reports the checks, §6 how to align the audio and which control regressors to include, and §7 the limitations."])}
""")

    H.append(f"""
<h2 class="newpage">1&nbsp; What the model does</h2>
<p>Listeners continuously anticipate what comes next, at several levels at once. The model estimates these predictions for every word of the story.
From them it derives, per level:</p>
{ul(["<b>Surprisal:</b> −log<sub>2</sub> of the probability of what actually came, in bits. High means unexpected.",
     "<b>Entropy:</b> the uncertainty of the prediction before the input arrived, in bits. High means many options were open.",
     "<b>Semantic prediction error</b> (meaning level only): how far the actual meaning was from the expected meaning, "
     "as a cosine distance (0 = the same meaning; about 1 = unrelated). It is not in bits."])}
{table(["Level", "What is predicted", "Example question", "Measures", "Resource"], [
    ["Lexical", "the next word, given all preceding text", "How expected is <i>Vorruhestand</i> after “Marlene sollte in den …”?", "surprisal, entropy", "GerPT2"],
    ["Syntactic", "the part of speech of the next word", "Was a noun expected here?", "surprisal, entropy", "GerPT2 + spaCy tagger"],
    ["Semantic", "the meaning of the next word", "How far is the word's meaning from what was expected?", "prediction error (distance)", "GerPT2 + fastText vectors"],
    ["Phonemic", "the next phoneme within the word", "After /f oː ʁ/, how expected is /uː/?", "surprisal, entropy", "GerPT2 + MFA pronunciations"],
    ["Syllabic " + tag("extension"), "the next syllable within the word", "After <i>Vor-</i>, how expected is <i>-ru-</i>?", "surprisal, entropy", "same as phonemic"],
], caption="<b>Table 1.</b> The five levels.", widths=["14%", "22%", "30%", "15%", "19%"])}
<p>All levels come from one source: at each word position, the language model gives a probability to every complete word of a
{vocab:,}-word German dictionary. The levels differ only in what is read off that distribution: the probability of the word itself and
the uncertainty over the most likely words (lexical); the parts of speech of the most likely words (syntactic); their meaning vectors (semantic);
and the pronunciations of all dictionary words (phonemic and syllabic).</p>
<p>“The most likely words” is always the same set, called the <b>nucleus</b>: the top words that together reach 90% of the probability,
and at least 40 words. It is the model's prediction <i>before</i> the word is heard. The word actually heard is part of it only if the model
ranked it among those top words; a very unexpected word, such as an anomalous noun, is not added to it.</p>
<p>The phonemic and syllabic levels come in two versions:</p>
{ul(["<b>Hierarchical:</b> candidates weighted by their probability in context (the model of interest);",
     "<b>Single-level:</b> candidates weighted only by their frequency in German (the context-free baseline of Heilbron et al.)."])}
<div class="box"><p><b>Worked example: why many phonemes have zero surprisal</b></p>
<p>While <i>Vorruhestand</i> unfolds, the set of words still compatible with the phonemes heard so far shrinks. After /f o\u02d0 \u0281 u\u02d0/
only one dictionary word remains, so every later phoneme is certain: its surprisal and entropy are 0. This is the word's uniqueness point
(Marslen-Wilson, 1987); it appears in the frequency-only version too.</p>
<p>Context can make a word certain even earlier. In “… geschickt werden”, GerPT2 gives <i>werden</i> a probability of about 99%,
so all phonemes after the first carry almost no surprisal in the hierarchical model. Across the story, {zero_pct:.0f}% of phonemes have exactly
zero hierarchical surprisal. These zeros are real values and belong in the regression.</p></div>
""")

    H.append(f"""
<h2 class="newpage">2&nbsp; Stimulus</h2>
<p>The stimulus is the full story ({n_sent} sentences). It has {n_units:,} units: {n_words:,} words and {n_punct} punctuation marks.
The text was taken as provided, with these corrections of transcription artefacts:</p>
{ul(["four line breaks inside sentences (from the PDF layout) were joined;",
     "quotation marks were added to two pieces of direct speech that lacked them, and a missing space after a closing quote was added;",
     "<i>Bio gemüse</i> → <i>Biogemüse</i>, <i>quadratmeter</i> → <i>Quadratmeter</i>, <i>Brennesseln</i> → <i>Brennnesseln</i>."])}
<p>The Swiss spelling (<i>ss</i> for <i>ß</i>) was kept in the text. The models read the standard spelling where it is clearly the norm (§3.1).</p>
<p>The story contains 15 deliberately anomalous words, listed in Table 2 and in {code('stimuli/violations.csv')}. They were not altered.</p>
{ul(["Each is a noun that does not fit its sentence in meaning, often also with the wrong gender or case.",
     "Each also occurs elsewhere in the story in a normal sentence.",
     "This allows a matched comparison: the same word, differing only in context."])}
{table(["", "Anomalous sentence (end)", "Normal occurrences", "Lexical surprisal (bits)"], arows,
       caption="<b>Table 2.</b> The anomalous words, with the preceding words of their sentence, the number of normal occurrences of the same word elsewhere in the story, and their lexical surprisal.",
       widths=["4%", "60%", "18%", "18%"], cls="compact")}
""")

    H.append(f"""
<h2 class="newpage">3&nbsp; Draft Methods text</h2>
<p class="note">The following is written in the register of a Methods section. Equation numbers, section references and resource names can be adapted to the journal.
Surprisal and entropy values are in bits; the semantic prediction error is a cosine distance.</p>
<h3>3.1 Stimulus preprocessing</h3>
<p>The story text was segmented into units: maximal runs of letters, and single punctuation marks. Digits, quotation marks and dashes remained in the text
read by the language model but received no row of their own.</p>
<p>The text uses Swiss orthography, which writes <i>ss</i> for <i>ß</i>. For every word containing <i>ss</i>, the <i>ß</i> variant was presented to the models if two
conditions held in SUBTLEX-DE (Brysbaert et al., 2011):</p>
{ul(["it was at least five times as frequent as the <i>ss</i> spelling;", "it was attested at least three times."])}
<p>This affected {len(cfg['spelling_normalised'])} word forms (e.g. <i>regelmässig</i> → <i>regelmäßig</i>, <i>Klugscheisser</i> → <i>Klugscheißer</i>),
while <i>dass</i> and <i>Fitnessstudio</i> were kept. The number of units was unchanged. The original spelling was retained for alignment with the audio.</p>

<h3>3.2 Language model and lexical surprisal</h3>
<p><b>Model.</b> Next-word probabilities were obtained from GerPT2 ({code('benjamin/gerpt2')}), a German GPT-2 model (Radford et al., 2019; Minixhofer, 2020).
It was selected among five German GPT-2 models (124 M to 1.5 B parameters) as the one whose surprisal best predicted German eye-tracking reading times (§5.1).
A preliminary check on independent German listening data (source-localised MEG of {n_part} listeners; Gehmacher et al., 2026) was consistent with this choice (§5.2).</p>
<p><b>Input.</b> The model read the text exactly as written, including punctuation. The text was preceded by the model's beginning-of-sequence token.
Model tokens were assigned to units by character offsets.</p>
<p><b>Context window.</b> When the text exceeded the 1,024-token context window, the context was rebuilt so that it started at a sentence boundary and kept
about 800 tokens of preceding text. This follows Heilbron et al.'s procedure of at least 700 tokens of overlap and no windows starting mid-sentence.</p>
<p><b>Word probability.</b> Because GPT-2 tokens carry the whitespace that precedes a word, the product of a word's token probabilities is the probability that
the next word <i>begins with</i> that string. We therefore computed the probability of the word itself (Pimentel &amp; Meister, 2024). Here <i>B</i> is the set of
tokens that begin a new unit (whitespace- or punctuation-initial) and <i>c</i> is the preceding context:</p>
<p class="eq"><i>P</i>(<i>w</i> | <i>c</i>) = <i>P</i>(<i>t</i><sub>1</sub>…<i>t</i><sub>n</sub> | <i>c</i>) · <i>P</i>(<i>B</i> | <i>c</i>, <i>t</i><sub>1</sub>…<i>t</i><sub>n</sub>) / <i>P</i>(<i>B</i> | <i>c</i>)<span class="eqn">(1)</span></p>
<p><b>Layout.</b> Whether a word is preceded by a space or by a line break is a property of the text layout that a listener does not hear.
This choice was therefore conditioned away, by multiplying by [<i>P</i>(space) + <i>P</i>(newline)] / <i>P</i>(space).</p>
<p><b>Lexical surprisal</b> is −log<sub>2</sub> <i>P</i>(<i>w</i> | <i>c</i>). Lexical entropy is defined on the candidate distribution (§3.3).</p>

<h3>3.3 Candidate distribution, nucleus and lexical entropy</h3>
<p>The lexical entropy and the syntactic, semantic, phonemic and syllabic predictors were derived from one distribution over complete words per word position. The candidates were:</p>
{ul([f"all words of the German pronunciation dictionary of the Montreal Forced Aligner (MFA {code('german_mfa')}; McAuliffe et al., 2017), in lowercase and capitalised form;",
     "the standard (<i>ß</i>) spelling where applicable;", "every word of the story;", "punctuation marks."])}
<p>In total there were {vocab:,} lexical items and {forms:,} written forms. Each candidate was scored with Eq. 1 against a shared, cached context.
Candidates were arranged in a token prefix tree, and subtrees were skipped only when their total probability was provably negligible, bounded to 1% relative
error on every cohort probability. The median scored probability mass per position was {words['dict_mass'].median():.2f}, and the largest skipped mass was
{'below 0.01' if words['pruned_mass'].max() < 0.01 else num(words['pruned_mass'].max(), 3)}.</p>
<p><b>Nucleus.</b> Following Heilbron et al. (2022, SI; Holtzman et al., 2020), the predicted next word was represented by the nucleus <i>K</i> of this distribution:
the candidates ranked by probability, kept up to and including the one at which the cumulative probability reaches 0.9, and at least 40 candidates.
The same nucleus was used for lexical entropy, part of speech and meaning. It is the model's prediction before the word is heard: the presented word was
not added to it, and is a member only if it is among the top candidates.</p>
<p><b>Lexical entropy</b> is the Shannon entropy of the nucleus, renormalised to sum to 1:</p>
<p class="eq"><i>H</i> = −Σ<sub><i>k</i> ∈ <i>K</i></sub> <i>p̂</i>(<i>k</i>) log<sub>2</sub> <i>p̂</i>(<i>k</i>), &nbsp; <i>p̂</i>(<i>k</i>) = <i>P</i>(<i>k</i> | <i>c</i>) / Σ<sub><i>j</i> ∈ <i>K</i></sub> <i>P</i>(<i>j</i> | <i>c</i>)<span class="eqn">(2)</span></p>

<h3>3.4 Syntactic predictors</h3>
<p>Following Heilbron et al. (2022, SI), every candidate in the nucleus was placed after the preceding words of its sentence and tagged with spaCy
({code('de_core_news_lg')}; Honnibal et al., 2020; Universal Dependencies tags), i.e. tagging “every potential sentence”. The presented word was tagged in exactly
the same way, with the preceding words only. No word after the current one entered the computation, because a prediction cannot depend on words not yet heard.
An ambiguous word is therefore resolved by the preceding words alone: in “ein Bad zu …”, <i>zu</i> is tagged as a preposition, because the verb that makes it
infinitival (“zu nehmen”) has not been heard yet. The tag of the word within its full sentence was used only to label it as a content or function word (§3.5).</p>
<p>With <i>Q</i> the nucleus members that share the presented word's tag:</p>
<p class="eq">POS surprisal = −log<sub>2</sub> [ Σ<sub><i>q</i> ∈ <i>Q</i></sub> <i>P</i>(<i>q</i> | <i>c</i>) / Σ<sub><i>k</i> ∈ <i>K</i></sub> <i>P</i>(<i>k</i> | <i>c</i>) ]<span class="eqn">(3)</span></p>
<p>If the presented word was not in the nucleus, its own probability was added to its tag for the surprisal, so that a part of speech is never less probable than a word carrying it.
POS entropy is the entropy of the distribution over tags in the nucleus.</p>

<h3>3.5 Semantic predictors</h3>
<p>The predicted meaning was the probability-weighted mean of the fastText vectors ({code('cc.de.300')}; Grave et al., 2018) of the word candidates in the nucleus.
The semantic prediction error was the cosine distance between this vector and the presented word's vector:</p>
<p class="eq">error = 1 − cos( Σ<sub><i>k</i> ∈ <i>K</i></sub> <i>P</i>(<i>k</i> | <i>c</i>) · <b>v</b><sub><i>k</i></sub>, &nbsp;<b>v</b><sub><i>w</i></sub> )<span class="eqn">(4)</span></p>
<p>It is a distance, not a surprisal in bits, and no semantic entropy was computed (as in Heilbron et al.). It was computed for content words (nouns, proper nouns, verbs,
adjectives, adverbs, by the word's tag in its full sentence). Vectors were looked up in the written casing.</p>
<p>As a control for integration difficulty, we also computed semantic distance (Broderick et al., 2018): one minus the Pearson correlation between the word's vector
and the mean vector of the preceding words of the same sentence (of the previous sentence for sentence-initial words), excluding punctuation.</p>

<h3>3.6 Phonemic predictors</h3>
<p><b>Pronunciations.</b> Pronunciations were taken from the MFA dictionary. The {cov['via_g2p_fallback']} story word types absent from it (of {cov['total_words']})
were transcribed with the MFA German grapheme-to-phoneme model.</p>
<p><b>Cohort.</b> At phoneme position <i>i</i>, the cohort <i>C<sub>i</sub></i> is the set of words consistent with the first <i>i</i> − 1 phonemes heard.
Words that end at position <i>i</i> contribute an explicit end-of-word outcome (Brodbeck et al., 2018; Heilbron et al., 2022). The probability of the next phoneme <i>a</i> is:</p>
<p class="eq"><i>P</i>(φ<sub><i>i</i></sub> = <i>a</i> | φ<sub>&lt;<i>i</i></sub>) = Σ<sub><i>w</i> ∈ <i>C<sub>i</sub></i>, <i>w<sub>i</sub></i> = <i>a</i></sub> <i>p</i>(<i>w</i>) / Σ<sub><i>w</i> ∈ <i>C<sub>i</sub></i></sub> <i>p</i>(<i>w</i>)<span class="eqn">(5)</span></p>
<p>Two versions:</p>
{ul(["In the <b>hierarchical</b> model, <i>p</i>(<i>w</i>) is the contextual probability of word <i>w</i> from the candidate distribution (§3.3), over the whole dictionary.",
     "In the <b>single-level</b> model, <i>p</i>(<i>w</i>) is the SUBTLEX-DE frequency count of <i>w</i> plus one."])}
<p>Phoneme surprisal and entropy were computed at every phoneme.</p>

<h3>3.7 Syllabic predictors (extension)</h3>
<p>Syllable-level predictors were computed with the same cohort model (Eq. 5), with syllables as units. Pronunciations were syllabified by legal-onset maximisation:</p>
{ul(["<b>Legal onsets:</b> the consonant clusters attested word-initially in at least ten dictionary words, with loanword-only onsets excluded word-internally.",
     "<b>Hard syllable boundaries:</b> compound and prefix boundaries, which German does not resyllabify across. They were detected by a frequency criterion "
     "(Koehn &amp; Knight, 2003) combined with a pronunciation check."])}

<h3>3.8 Frequency</h3>
<p>Unigram surprisal was −log<sub>2</sub> of the Laplace-smoothed SUBTLEX-DE probability, (count + 1) / (<i>N</i> + <i>V</i>), with <i>N</i> = 25.0 million tokens and <i>V</i> = 190,500 types.</p>

<h3>3.9 Implementation</h3>
<p>The pipeline was implemented in Python 3.11 with PyTorch 2.13 (Apple M4 GPU), Hugging Face Transformers 4.28, spaCy 3.8 ({code('de_core_news_lg')} 3.8.0) and MFA 3.4.
Every fast computational path was verified against a direct computation with the Hugging Face model in an automated test suite. Punctuation marks received
lexical values only; they have no acoustic onset.</p>

<h3>3.10 Statistical checks</h3>
<p>These checks describe one story; they are sanity checks of the predictors, not confirmatory tests about listeners. We tested whether the anomalous words were less
predictable than the same words in their normal sentences:</p>
{ul(["<b>Paired comparison:</b> a Wilcoxon signed-rank test, with the control value being the mean over the word's normal occurrences.",
     "<b>Unpaired comparison:</b> a Mann–Whitney U test against all other nouns."])}
<p>Both were run at the lexical, POS, semantic and first-phoneme levels. General properties of the predictors were tested with Spearman correlations,
a Mann–Whitney U test and Wilcoxon signed-rank tests (Table 7). Phonemes of one word are not independent, so the phoneme-level tests used one value per word
(its mean phoneme surprisal, or its first versus the mean of its later phonemes). All tests were one-sided in the predicted direction, and p-values were
Holm-corrected across all twelve tests (Holm, 1979). Overlap between the word-level predictors was described with Pearson correlations and variance inflation factors (VIF).</p>
""")

    H.append(f"""
<h2>4&nbsp; Differences from Heilbron et al. (2022)</h2>
<p>The method is that of Heilbron et al. (2022): the same five-part structure, the same definitions of surprisal and entropy, the same nucleus, the same POS and
semantic procedures, and the same cohort model with an end-of-word outcome. The differences fall into three kinds, marked in Table 3:</p>
{ul([tag("language") + " resources replaced because the study is in German;",
     tag("refinement") + " computations that estimate the paper's own quantities more exactly;",
     tag("extension") + " an addition that does not alter the other levels."])}
<p>For the end-of-word correction, which is the refinement that changes word probabilities, the paper-style value is reported alongside ({code('lexical_surprisal_raw')}).</p>
{table(["Aspect", "Heilbron et al. (2022)", "This study", "Reason / how to recover the paper-style value"], [
    ["<b>Language model</b><br>" + tag("language"), "GPT-2 XL (English, 1.5 B parameters)", "GerPT2 (German, 124 M)",
     "Of five German GPT-2 models, GerPT2 best predicted German reading times (§5.1). The German 1.5 B model fitted about half as well. "
     "A preliminary listening check (§5.2) is consistent with this choice. GerPT2-large can be added as comparison columns."],
    ["<b>Other resources</b><br>" + tag("language"), "English tagger, word embeddings, pronunciation lexicon and frequencies",
     f"spaCy {code('de_core_news_lg')}; fastText {code('cc.de.300')}; MFA {code('german_mfa')} (+ G2P); SUBTLEX-DE", "Standard German counterparts."],
    ["<b>Spelling</b> " + tag("language"), "—", f"Swiss <i>ss</i> read as <i>ß</i> where standard ({len(cfg['spelling_normalised'])} forms)",
     f"The models were trained mainly on standard spelling. Switch: {code('--no-spelling-normalization')}."],
    ["<b>Word probability</b><br>" + tag("refinement"), "product of the word's token probabilities", "end-of-word corrected (Eq. 1)",
     "The token product is the probability that the next word <i>starts with</i> the string, so a fragment such as <i>Vor</i> would inherit the probability of "
     "<i>Vorruhestand</i>. Complete words are almost unaffected, and so is the fit to reading times (§5.1). "
     f"Paper-style value: column {code('lexical_surprisal_raw')}; switch {code('--no-eow-correction')}."],
    ["<b>Text layout</b><br>" + tag("refinement"), "—", "space-vs-line-break choice conditioned away",
     "Layout is inaudible. Without this, a word after a line break was charged for the model's expectation of a paragraph break (e.g. 26 bits for <i>Marlene</i>)."],
    ["<b>Phoneme prior</b><br>" + tag("refinement"), "GPT-2 probabilities for the nucleus; a flat pseudo-count for all other words",
     "GPT-2 probability of every dictionary word",
     "This is the exact value that the nucleus-plus-pseudo-count approximates. It is made tractable by the prefix-tree scorer with a 1% error bound."],
    ["<b>Syllable level</b><br>" + tag("extension"), "—", "syllabic surprisal and entropy (§3.7)", "An additional level. It does not change the other predictors."],
], caption="<b>Table 3.</b> Differences from Heilbron et al. (2022).", widths=["16%", "22%", "24%", "38%"], cls="small")}
<p><b>Unchanged from the paper:</b></p>
{ul(["surprisal and entropy definitions, in bits;",
     "the nucleus: complete words up to and including the one that crosses p = 0.9, at least 40 words; one nucleus for lexical entropy, part of speech and meaning, without adding the presented word;",
     "context windows starting at sentence boundaries, with at least 700 tokens of overlap;",
     "part of speech: candidates and the presented word tagged with the preceding words only (“every potential sentence”), and POS surprisal as the nucleus probability of the presented word's tag;",
     "semantic prediction error as the cosine distance to the probability-weighted mean vector (no semantic entropy);",
     "hierarchical versus single-level cohort models, each with an explicit end-of-word outcome."])}
<p>One implementation detail: when the presented word is not in the nucleus, its probability is added to its tag for the POS surprisal (§3.4).
This keeps P(tag) ≥ P(word). It does not change the POS entropy, which describes the prediction before the word.</p>
""")

    H.append(f"""
<h2>5&nbsp; Checks</h2>
<h3>5.1 Choice of language model</h3>
<p>Each candidate model was evaluated on its fit to German reading times, using PoTeC (Jakobi et al., 2025): eye movements of native readers on German expository texts.</p>
{ul(["<b>Measures:</b> first-pass reading time and total fixation time, averaged over readers.",
     "<b>Baseline regression:</b> word length and frequency of the current and previous word.",
     "<b>Score:</b> the held-out gain in log-likelihood (ΔLL) when surprisal of the current and previous word is added, over 10 × 10-fold cross-validation (e.g. Wilcox et al., 2020)."])}
<p>GerPT2 fitted best on both measures (Table 4).</p>
{ul(["The larger German models predicted the story text better (lower bits per word) but predicted human reading worse. This is a known dissociation between model quality and fit to humans.",
     f"For the dbmdz model, the corrected pipeline raised ΔLL from {num(old['dLL_FPRT'], 1)} to {num(db_new['dLL_FPRT'], 1)} (first-pass reading time; "
     f"{db_new['dLL_FPRT'] / old['dLL_FPRT']:.1f}×) and from {num(old['dLL_TFT'], 1)} to {num(db_new['dLL_TFT'], 1)} (total fixation time; {db_new['dLL_TFT'] / old['dLL_TFT']:.1f}×).",
     f"The end-of-word correction itself left the fit about unchanged: GerPT2 with versus without it, {num(g_new['dLL_FPRT'], 1)} vs. {num(g_raw['dLL_FPRT'], 1)} "
     f"and {num(g_new['dLL_TFT'], 1)} vs. {num(g_raw['dLL_TFT'], 1)} (SE ≈ {num(g_new['dLL_FPRT_se'], 1)})."])}
<p>Two limits: the cross-validation split words at random rather than holding out whole texts, and the standard errors treat the repeated cross-validation runs as independent,
so they are too small. The ranking is a useful guide, not a precise comparison.</p>
{table(["Model", "Parameters", "Bits per word (story, part 1)", "ΔLL first-pass RT", "ΔLL total fixation"], brows,
       caption="<b>Table 4.</b> Language-model benchmark. ΔLL: held-out log-likelihood gain on PoTeC reading times (mean ± SE over cross-validation repeats); higher is better. "
               "Bits per word: perplexity on the first part of the story (481 words); lower means the text is better predicted. The bottom row is the pipeline before the corrections.",
       widths=["38%", "12%", "18%", "16%", "16%"], cls="small")}

<h3>5.2 A preliminary check on listening data (MEG)</h3>
<p>The reading benchmark concerns silent reading of expository text, whereas this study concerns listening. The choice was therefore checked on independent German listening data:
the publicly available source-localised MEG of Gehmacher et al. (2026). Participants listened to German audiobook stories; the data cover 360 cortical parcels at 50 Hz,
with forced-aligned word onsets.</p>
{ul([f"<b>Data used:</b> {n_part} of the 24 shared participants and all 16 story recordings (462–554 words each).",
     "<b>Surprisal:</b> computed with the present pipeline for each language model, on the authors' transcripts.",
     "<b>Encoding model:</b> as described by the authors. Controls were acoustic envelope, acoustic onsets, pitch, word onset, unigram surprisal and semantic distance. "
     "It used a temporal response function with lags −0.2 to 1.2 s, ridge regression with the penalty chosen per parcel, leave-one-block-out cross-validation, "
     "and Spearman correlation between predicted and observed responses.",
     "<b>Score:</b> the gain in held-out accuracy (Fisher z) when a model's surprisal is added to the controls. It was averaged over all cortical parcels, and over "
     "a selected region: the 36 parcels best predicted by the controls alone. That region was chosen without reference to any language model."])}
{table(["Language model", "Gain, all parcels", "Gain, selected region", "Gain &gt; 0, all parcels", "Gain &gt; 0, region"], [
    lrow("gerpt2", "GerPT2 (this pipeline)"),
    lrow("gerpt2_large", "GerPT2-large (this pipeline)"),
    lrow("gerpt2_raw", "GerPT2, no end-of-word correction"),
    lrow("dbmdz", "dbmdz/german-gpt2 (this pipeline)"),
    lrow("theirs", "dbmdz/german-gpt2 (Gehmacher et al.'s values)"),
], caption=f"<b>Table 5.</b> Listening benchmark: gain in held-out MEG prediction accuracy from adding each model's surprisal to the control regressors "
           f"(Fisher z × 1000; mean ± SEM over {n_part} participants). Selected region: the 36 cortical parcels best predicted by the controls alone.",
   widths=["34%", "16%", "18%", "16%", "16%"], cls="small")}
<p>What the comparison shows, with {n_part} participants:</p>
{ul([f"<b>GerPT2 vs GerPT2-large:</b> no significant difference (GerPT2 higher in {w_lg_all} of {n_part} participants over all parcels, Wilcoxon p = {p_lg_all:.2f}; "
     f"in {w_lg_roi} of {n_part} in the selected region, p = {p_lg_roi:.2f}). With so few participants, this does <i>not</i> show that the two models are equally good.",
     f"<b>GerPT2 vs dbmdz:</b> in the selected region, GerPT2 was higher than the dbmdz model (the one Gehmacher et al. used) in {w_db_roi} of {n_part} participants "
     f"(p = {p_db_roi:.2f}; against the authors' own values {w_th_roi} of {n_part}, p = {p_th_roi:.2f}); GerPT2-large was also higher in {w_lgdb_roi} of {n_part}. "
     f"Over all parcels, GerPT2 was higher in {w_db_all} of {n_part} (p = {p_db_all:.2f}). None of these differences is significant.",
     "<b>End-of-word correction:</b> changed almost nothing, as in reading.",
     f"<b>Reproduction:</b> the present pipeline's dbmdz values closely matched the authors' own values (rank correlation of the per-participant gains {rho_repro:.2f})."])}
<p>In short, this is preliminary listening evidence consistent with keeping GerPT2. It is not a confirmation, and it is not evidence that GerPT2 is as good as GerPT2-large.
The transcripts lacked punctuation (identically for every model), and only lexical surprisal was compared.</p>

<h3>5.3 Implementation tests</h3>
<p>An automated test suite ({code('tests/run_all.py')}) checks every fast computational path against a direct computation. It covers:</p>
{ul(["the context cache, the shared-context attention and the batched vocabulary scorer;",
     "the pruning error bound, observed at most 0.01 bits in phoneme surprisal;",
     "the end-of-word correction and the cohort model;", "syllabification;",
     "the nucleus (including the word that crosses 90%), the POS level (no word after the current one may change its value) and the frequency level;",
     "the alignment merge (phone identity, syllable boundaries);",
     "the real GerPT2 model."])}
<p>The suite reports any suite it had to skip. An automatic audit of the story results found no degenerate values (constants, infinities or values pinned at a numerical floor).</p>

<h3>5.4 Minimal sentence pairs</h3>
<p>Sentence pairs that differ in one critical word, with a known expected direction, were run through the full pipeline (Table 6).
All 13 directional checks came out as expected: lexical surprisal, semantic prediction error and first-phoneme surprisal were higher for the violation
in every pair where they apply, POS surprisal stayed similar when the word class was kept, and it rose sharply for a closed-class violation
(<i>einen oder Mantel</i>).</p>
<p>The open-class word-class violation <i>einen laufen Mantel</i> is a known limit, not a check. The tagger forces <i>laufen</i> into a word class that fits
the slot (adjective in the full sentence); with only the preceding words, its tag is one the context already expects, so its POS surprisal is low
({num(laufen['pos_surprisal'], 1)} bits). POS surprisal therefore cannot register this violation (§7).</p>
{table(["Contrast", "Critical word", "Lexical", "POS", "Semantic", "Phoneme 1 (hier.)"], prow,
       caption="<b>Table 6.</b> Minimal pairs: values for the expected word → the violation (bits; semantic: cosine distance). POS column: full-sentence labels of both words, then POS surprisal.",
       widths=["13%", "20%", "14%", "25%", "14%", "14%"], cls="small")}

<h3>5.5 The story: anomalous words and general properties</h3>
<p>Compared with the same words in their normal sentences, the anomalous words had higher surprisal at the lexical, semantic and first-phoneme levels, but not at the POS level (Figure 1, Table 7):</p>
{ul([f"lexical: median {med(lex1a)}, p<sub>Holm</sub> = {pfmt(lex1a['p_holm'])};",
     f"semantic: median {med(sem1a)} (distance), p<sub>Holm</sub> = {pfmt(sem1a['p_holm'])};",
     f"first phoneme: median {med(ph1a)}, p<sub>Holm</sub> = {pfmt(ph1a['p_holm'])};",
     f"POS: median {med(pos1a)}, p<sub>Holm</sub> = {pfmt(pos1a['p_holm'])}."])}
<p>The same held against all other nouns. The absent POS effect is expected, because every anomalous word is a noun in a position where a noun is expected.</p>
<p>The general properties held as predicted:</p>
{ul(["less frequent words were more surprising;", "content words were more surprising than function words;",
     f"context-based (hierarchical) phoneme surprisal was about half the frequency-based value (per word, median {t4['detail'].replace('median ', '').replace(' vs ', ' vs. ')});",
     f"the first phoneme of a word was much more surprising than its later phonemes (per word, median {t5['detail'].replace('median ', '').replace(' vs ', ' vs. ')})."])}
<figure>{figure}<figcaption><b>Figure 1.</b> Each of the 15 anomalous words (orange) against the mean of the same word in its normal sentences (blue); lines connect the pairs,
black bars are medians. p: one-sided Wilcoxon signed-rank test, Holm-corrected across all twelve tests.</figcaption></figure>
{table(["Test (one-sided)", "n", "Result", "p (Holm)", "Sig."], srows,
       caption="<b>Table 7.</b> Statistical tests on the full story. All tests are one-sided in the predicted direction; p-values are Holm-corrected across the twelve tests. "
               "Tests 4 and 5 use one value per word, because the phonemes of a word are not independent.",
       widths=["48%", "11%", "21%", "12%", "8%"], cls="small")}

<h3>5.6 Descriptive statistics and overlap between predictors</h3>
{table(["Predictor", "Per", "n", "Mean", "SD", "Median", "Max"], drows,
       caption="<b>Table 8.</b> Predictor distributions for the full story (bits; semantic measures: distances). Semantic prediction error is defined for content words only. "
               "The single-level entropy of the first syllable is the same for every word (the prior has no context).",
       widths=["38%", "14%", "10%", "9%", "9%", "10%", "10%"], cls="small")}
{table([""] + [lab for _, lab in corr_cols] + ["VIF"], crows,
       caption=f"<b>Table 9.</b> Pearson correlations between word-level predictors, and variance inflation factors (VIF; computed on the {len(complete)} words that have every predictor, i.e. content words).",
       widths=["16%"] + ["12%"] * 6 + ["12%"], cls="small")}
<p>The highest correlation between two word-level predictors is r = {max_r:.2f} ({long_names[a_]} with {long_names[b_]}), and all VIF values are below
{np.ceil(vif.max() * 10) / 10:.1f}. So there is no gross overlap. First-phoneme surprisal, a phoneme-level predictor, correlates r = {r_phon1:.2f} with lexical surprisal;
POS entropy correlates r = {pos_entropy_r:.2f} with lexical entropy. Correlations and VIFs do not show that each predictor explains something of its own:
that has to be tested in the TRF models themselves (§6.2).</p>
""")

    H.append(f"""
<h2>6&nbsp; Using the predictors: alignment and control regressors</h2>
<h3>6.1 Forced alignment</h3>
<p>Onset times come from forced alignment of the story recording with the Montreal Forced Aligner, using the same resources as the predictors:</p>
{ul([f"the {code('german_mfa')} pronunciation dictionary;", f"the {code('german_mfa')} grapheme-to-phoneme model for words the dictionary lacks;",
     f"the {code('german_mfa')} acoustic model."])}
<p>Every aligned phoneme then corresponds to a phoneme with predictor values. Two requirements follow:</p>
{ul(["The transcript must match the recording word for word. Any change to the text requires recomputing the predictors.",
     "The alignment should be inspected (e.g. in Praat) before use."])}
<p>The resulting TextGrid is merged with {code('--textgrid')}, which adds onset, offset and duration to the phoneme and syllable tables. The merge checks that each word's
aligned phones are the <i>same</i> phones, in the same order, as the pronunciation the predictors were computed on; equal numbers are not enough, because a different
pronunciation of the same length would otherwise pass. Syllables are timed by cutting the aligned phones exactly where the pipeline's own syllables end, so each timed
syllable is the one its values belong to. A mismatch stops the merge, or the word is dropped with a warning under {code('--textgrid-lenient')}.</p>
<p>Commands:</p>
<pre>mfa model download acoustic german_mfa
mfa align &lt;folder with story.wav + story.txt&gt; german_mfa.dict german_mfa &lt;output&gt; \\
    --g2p_model_path german_mfa.zip --beam 100 --retry_beam 400</pre>
<h3>6.2 Control regressors</h3>
<p>Contextual predictors should be tested against non-contextual regressors that are present in every model. A “surprisal effect” could otherwise reflect acoustics,
the mere occurrence of a word, or word frequency (lexical surprisal correlates r = {r_lex_uni:.2f} with unigram surprisal in this story). Following the logic of
Heilbron et al. (2022) and the regressor set of Gehmacher et al. (2026), every model should contain:</p>
{ul(["<b>Acoustics:</b> the speech envelope, acoustic onsets and pitch.",
     f"<b>Word level:</b> word-onset impulses, unigram surprisal ({code('unigram_surprisal')}) and semantic distance ({code('semantic_distance')}).",
     f"<b>Phoneme level:</b> phoneme-onset impulses, and the single-level (frequency-based) phoneme surprisal and entropy ({code('phon_*_freq')}). "
     "These are the non-contextual counterparts of the hierarchical predictors."])}
<p>The contribution of each contextual predictor is its unique explained variance: the drop in cross-validated prediction accuracy when it is removed from the full model.
The moderate correlations (§5.6) make this possible but do not guarantee it; whether each predictor has a unique contribution is a result of the TRF analysis, not something
the correlations can show.</p>

<h2>7&nbsp; Limitations</h2>
{ul([
    "<b>Tagger-based POS surprisal.</b> Every word is tagged with the preceding words only, i.e. inside an unfinished sentence. The tagger can force an odd word into its slot, "
    "so violations that keep the word class, such as the story's nouns in noun positions, are invisible to this measure. It can also misread an unfinished sentence: for example, "
    "“ein Bad” is read as a name (9.4 bits) and “Wie krieg” as a noun (13.2 bits). Both are properties of the method of Heilbron et al. (2022), not new errors.",
    f"<b>Model choice.</b> GerPT2 was selected on reading times. The listening check (§5.2) used another study's data and only {n_part} participants: it is consistent with "
    "keeping GerPT2, but it cannot show that GerPT2 is as good as GerPT2-large, and the advantage over dbmdz was not significant. Only lexical surprisal was compared, and "
    "not on the study's own recordings. If the study's data are used to compare models, this should be done on data held out from the main analysis.",
    "<b>Statistics.</b> All checks come from one story. They describe the predictors; they are not evidence about listeners. The reading benchmark split words, not texts, "
    "and its standard errors are too small (§5.1).",
    f"<b>Heuristic components.</b> Several components rely on heuristics: compound splitting, the Swiss-spelling rule, and grapheme-to-phoneme transcriptions for "
    f"{cov['via_g2p_fallback']} word types. A spot-check by a native speaker ({code('results/qa_sample.csv')}) is recommended.",
    "<b>Onset times.</b> The predictors have no times until the recording is aligned (§6.1).",
])}

<h2>8&nbsp; Output and reproduction</h2>
{table(["File", "Contents"], [
    [code("results/words.csv"), f"one row per unit ({n_units:,}); all word-level predictors, plus phoneme and syllable values as lists"],
    [code("results/phonemes.csv"), f"one row per phoneme ({n_phon:,}); hierarchical and single-level surprisal and entropy"],
    [code("results/syllables.csv"), f"one row per syllable ({n_syll:,})"],
    [code("results/statistics.csv"), "the tests of Table 7"],
    [code("results/run_config.json"), "all settings, model names and pronunciation coverage"],
    [code("docs/listening_benchmark_results.csv"), f"the listening benchmark of Table 5, per participant (scripts in {code('scripts/listening_benchmark/')})"],
    [code("docs/build_methods_report.py"), "builds this report from the files above"],
], widths=["38%", "62%"], cls="small")}
<p>Main columns of {code('words.csv')}:</p>
{ul([f"{code('lexical_surprisal')}: Eq. 1;", f"{code('lexical_surprisal_raw')}: the token product, i.e. the paper's definition;",
     f"{code('lexical_surprisal_spoken')}: spellings a listener cannot distinguish summed;",
     f"{code('lexical_entropy')}: Eq. 2, the entropy of the complete-word nucleus;",
     f"{code('pos_surprisal')}, {code('pos_entropy')}, {code('semantic_prediction_error')} (distance), {code('semantic_distance')} (distance), {code('unigram_surprisal')};",
     f"{code('…__gerpt2-large')}: comparison model, present only when the run includes it ({code('--extra-lexical-models')})."])}
<p>Reproduce with:</p>
<pre>python run_story.py --text-file stimuli/story_part1.txt --mfa-dict &lt;german_mfa.dict&gt; \\
    --device mps --extra-lexical-models none --out results/
python scripts/story_statistics.py results/
python docs/build_methods_report.py</pre>
<p>The run takes about {hours:.0f} hours on an Apple M4 (longer with the GerPT2-large comparison). The last command rebuilds this report.</p>
""")

    H.append("""
<h2>References</h2>
<div class="refs">
<p>Brodbeck, C., Hong, L. E., &amp; Simon, J. Z. (2018). Rapid transformation from auditory to linguistic representations of continuous speech. <i>Current Biology, 28</i>(24), 3976–3983.</p>
<p>Broderick, M. P., Anderson, A. J., Di Liberto, G. M., Crosse, M. J., &amp; Lalor, E. C. (2018). Electrophysiological correlates of semantic dissimilarity reflect the comprehension of natural, narrative speech. <i>Current Biology, 28</i>(5), 803–809.</p>
<p>Brysbaert, M., Buchmeier, M., Conrad, M., Jacobs, A. M., Bölte, J., &amp; Böhl, A. (2011). The word frequency effect: A review of recent developments and implications for the choice of frequency estimates in German. <i>Experimental Psychology, 58</i>(5), 412–424.</p>
<p>Gehmacher, Q., Kaltenmaier, A., Schubert, J., Weisz, N., &amp; Press, C. (2026). The dynamic relationship between pupil dilation and neural surprise in natural language comprehension. <i>bioRxiv</i>. https://doi.org/10.64898/2026.06.12.731866</p>
<p>Grave, E., Bojanowski, P., Gupta, P., Joulin, A., &amp; Mikolov, T. (2018). Learning word vectors for 157 languages. <i>Proceedings of LREC 2018</i>.</p>
<p>Heilbron, M., Armeni, K., Schoffelen, J.-M., Hagoort, P., &amp; de Lange, F. P. (2022). A hierarchy of linguistic predictions during natural language comprehension. <i>Proceedings of the National Academy of Sciences, 119</i>(32), e2201968119.</p>
<p>Holm, S. (1979). A simple sequentially rejective multiple test procedure. <i>Scandinavian Journal of Statistics, 6</i>(2), 65–70.</p>
<p>Holtzman, A., Buys, J., Du, L., Forbes, M., &amp; Choi, Y. (2020). The curious case of neural text degeneration. <i>Proceedings of ICLR 2020</i>.</p>
<p>Honnibal, M., Montani, I., Van Landeghem, S., &amp; Boyd, A. (2020). spaCy: Industrial-strength natural language processing in Python. Zenodo.</p>
<p>Jakobi, D. N., Kern, T., Reich, D. R., Haller, P., &amp; Jäger, L. A. (2025). PoTeC: A German naturalistic eye-tracking-while-reading corpus.</p>
<p>Koehn, P., &amp; Knight, K. (2003). Empirical methods for compound splitting. <i>Proceedings of EACL 2003</i>, 187–193.</p>
<p>Marslen-Wilson, W. D. (1987). Functional parallelism in spoken word-recognition. <i>Cognition, 25</i>(1–2), 71–102.</p>
<p>McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., &amp; Sonderegger, M. (2017). Montreal Forced Aligner: Trainable text-speech alignment using Kaldi. <i>Proceedings of Interspeech 2017</i>, 498–502.</p>
<p>Minixhofer, B. (2020). GerPT2: German large and small versions of GPT2 [Language model]. Hugging Face: <code>benjamin/gerpt2</code>.</p>
<p>Pimentel, T., &amp; Meister, C. (2024). How to compute the probability of a word. <i>Proceedings of EMNLP 2024</i>.</p>
<p>Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., &amp; Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI.</p>
<p>Wilcox, E. G., Gauthier, J., Hu, J., Qian, P., &amp; Levy, R. (2020). On the predictive power of neural language models for human real-time comprehension behavior. <i>Proceedings of CogSci 2020</i>.</p>
<p class="note">Reference details (pages, venues, the PoTeC publication) should be checked against the originals before submission.</p>
</div>
""")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Linguistic Predictability Report</title><style>{CSS}</style></head><body>{''.join(H)}</body></html>"""


CSS = """
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: Charter, "Bitstream Charter", Georgia, serif; font-size: 10.2pt; line-height: 1.42; color: #1a1a1a; margin: 0;
       text-align: justify; hyphens: auto; }
h1, h2, h3, .kicker, .meta, table, .caption, figcaption, .tag { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.kicker { color: #2b6cb0; font-size: 8.5pt; letter-spacing: .04em; margin: 0 0 4pt; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 6pt; text-align: left; }
.subtitle { font-size: 11.5pt; color: #444; margin: 0 0 10pt; }
.meta { font-size: 7.6pt; color: #555; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; padding: 6pt 0; margin: 0 0 14pt; text-align: left; }
h2 { font-size: 13.5pt; border-top: 1px solid #999; padding-top: 8pt; margin: 18pt 0 6pt; break-after: avoid; text-align: left; }
h2.newpage { break-before: page; margin-top: 0; }
h3 { font-size: 10.8pt; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 0 0 6pt; }
ul { margin: 0 0 6pt; padding-left: 16pt; }
li { margin: 0 0 2pt; }
code, pre { font-family: Menlo, "SF Mono", monospace; font-size: 8pt; }
code { hyphens: none; -webkit-hyphens: none; background: #eef0f3; padding: 0 2pt; border-radius: 2pt; }
pre { background: #eef0f3; padding: 6pt 8pt; white-space: pre-wrap; text-align: left; }
.summary { background: #f3f2ee; border-left: 3px solid #2b6cb0; padding: 9pt 12pt 4pt; margin: 0 0 12pt; }
.box { background: #f6f6f4; border: 1px solid #e0e0dc; padding: 8pt 11pt 3pt; margin: 8pt 0; break-inside: avoid; }
.note { font-size: 8.8pt; color: #555; font-style: italic; }
.eq { text-align: center; margin: 8pt 0; position: relative; }
.eqn { position: absolute; right: 0; color: #444; }
.tablewrap { margin: 8pt 0 10pt; break-inside: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 8.3pt; line-height: 1.32; text-align: left; table-layout: fixed; }
table.small { font-size: 7.7pt; }
table.compact td { padding-top: 1.2pt; padding-bottom: 1.2pt; }
th { text-align: left; font-weight: 600; border-bottom: 1.2px solid #333; padding: 3pt 5pt 3pt 0; vertical-align: bottom; }
td { border-bottom: 1px solid #e2e2e2; padding: 3pt 5pt 3pt 0; vertical-align: top; }
td code { font-size: 7.2pt; overflow-wrap: anywhere; }
.caption, figcaption { font-size: 7.6pt; color: #444; margin-top: 4pt; text-align: left; }
.tag { display: inline-block; font-size: 6.6pt; padding: 0 4pt; border-radius: 3pt; margin-top: 2pt; }
.tag.language { background: #e3edf8; color: #24548a; }
.tag.refinement { background: #fbe9e2; color: #9a3b1e; }
.tag.extension { background: #e4f3e6; color: #2e6b37; }
figure { margin: 10pt 0; break-inside: avoid; }
figure svg { width: 100%; height: auto; }
.refs p { font-size: 8.4pt; padding-left: 14pt; text-indent: -14pt; text-align: left; margin-bottom: 3pt; }
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validation", type=Path, default=DOCS / "validation_results.csv",
                    help="minimal-pair CSV from scripts/validate_model.py")
    ap.add_argument("--out", type=Path, default=DOCS / "methods_report_heilbron_german.pdf")
    ap.add_argument("--keep-html", action="store_true", help="also write the HTML next to the PDF")
    a = ap.parse_args()
    page = build_html(load(a.validation))
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "report.html"
        src.write_text(page, encoding="utf-8")
        if a.keep_html:
            a.out.with_suffix(".html").write_text(page, encoding="utf-8")
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={a.out}", src.as_uri()], check=True, capture_output=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
