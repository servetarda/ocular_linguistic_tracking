"""Offline tests: every component that needs no model download, including a
regression check for each error found in the 2026-09 audit, and an
end-to-end run of the pipeline on the toy resources.

    python tests/test_offline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import numpy as np

from linguistic_predictors import config
from linguistic_predictors.cohort import sequential_cohort_stats, single_level_stats
from linguistic_predictors.components_de import _case_variants
from linguistic_predictors.lexical import _bytes_to_unicode, _prefix_text, _split_prefix, token_classes, truncate_nucleus
from linguistic_predictors.lexicon_mfa import _is_lexical, _run_mfa_g2p
from linguistic_predictors.text import TextLayout, detokenize, sentence_bounds, sentence_ids
from linguistic_predictors.semantic import semantic_distance_broderick
from linguistic_predictors.syllabic import PhoneInventory, attested_onsets, syllabify, syllable_strings
from linguistic_predictors.vocab_scorer import FormTrie

failures = []


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(name)


# --- basics ------------------------------------------------------------------
check("entropy of a point mass is 0", math.isclose(config.entropy([1.0, 0.0]), 0.0, abs_tol=1e-12))
check("entropy of uniform-4 is 2 bits", math.isclose(config.entropy([0.25] * 4), 2.0))
check("surprisal of p=0.5 is 1 bit", math.isclose(config.surprisal(0.5), 1.0))
check("surprisal_from_logprob is exact below the EPSILON floor",
      math.isclose(config.surprisal_from_logprob(-100.0), 100 / math.log(2)))
nuc = truncate_nucleus({"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}, top_p=0.7, top_k=1)
check("nucleus: smallest set reaching top_p, renormalised", set(nuc) == {"a", "b"} and math.isclose(sum(nuc.values()), 1.0))
check("nucleus: top_k is a floor", len(truncate_nucleus({"a": 0.9, "b": 0.05, "c": 0.05}, top_p=0.5, top_k=2)) == 2)
sid = sentence_ids(["Er", "kam", ".", "Sie", "ging", "."])
check("sentence ids: the full stop belongs to its sentence", sid == [0, 0, 0, 1, 1, 1])
check("sentence bounds", sentence_bounds(sid, 4) == (3, 6))

# --- text handling: GPT-2 must see natural text ---------------------------
check("detokenize attaches punctuation",
      detokenize(["Die", "Nachricht", "kam", ".", "Marlene", ",", "ja"]) == "Die Nachricht kam. Marlene, ja")
lay = TextLayout.from_text("Die Nachricht kam völlig überraschend.  Über 100 Stellen\r\n\r\n\r\n„Nein“, sagte sie.")
check("normalize_text collapses whitespace, keeps one blank line",
      lay.text == "Die Nachricht kam völlig überraschend. Über 100 Stellen\n\n„Nein“, sagte sie.")
check("units skip digits/quotes but keep umlauts",
      lay.words == ["Die", "Nachricht", "kam", "völlig", "überraschend", ".", "Über", "Stellen",
                    "Nein", ",", "sagte", "sie", "."])
check("spans point at the units", all(lay.text[a:b] == w for w, (a, b) in zip(lay.words, lay.spans)))
lw = TextLayout.from_words(["Er", "kam", ".", "Sie", "ging"])
check("from_words builds exact spans", lw.text == "Er kam. Sie ging" and
      all(lw.text[a:b] == w for w, (a, b) in zip(lw.words, lw.spans)))
check("_split_prefix: trailing space moves to the lead", _split_prefix("Die Nachricht ") == ("Die Nachricht", " "))
check("_split_prefix: newline context has no lead", _split_prefix("Satz.\n") == ("Satz.\n", ""))
check("_split_prefix: empty context", _split_prefix("") == ("", ""))
check("_prefix_text: no space before punctuation", _prefix_text(["Er", "kam"], ".") == "Er kam")
check("_prefix_text: space before a word", _prefix_text(["Er", "kam"], "und") == "Er kam ")

# --- token classes: byte-level, umlaut- and quote-aware --------------------
b2u = _bytes_to_unicode()
enc = lambda s: "".join(b2u[b] for b in s.encode("utf-8"))
toks = [enc(" über"), enc("über"), enc("ber"), enc(" „"), enc("."), enc("\n"), enc(" 100"), enc(" Haus"),
        "<|endoftext|>", enc("ä")[:1]]
tc = token_classes(toks)
check("' über' is a space-led word start", tc["space_alpha"][0] and tc["has_space"][0])
check("'über' (bare) is a bare word start, not a boundary", tc["bare_alpha"][1] and not tc["boundary"][1])
check("'ber' continues a word", not tc["boundary"][2])
check("' „' is a boundary, not a word start", tc["boundary"][3] and not tc["space_alpha"][3])
check("'.' is a boundary", tc["boundary"][4])
check("newline is a boundary and newline-start", tc["boundary"][5] and tc["newline_start"][5])
check("' 100' is a boundary but no word start", tc["boundary"][6] and not tc["space_alpha"][6])
check("special tokens are boundaries", tc["boundary"][8])
check("lone umlaut lead byte counts as a letter", not tc["boundary"][9])

# --- syllabification -------------------------------------------------------
V = frozenset({"a", "aː", "eː", "iː", "ə", "ɐ", "ɔʏ", "aw", "ʊ", "ɪ", "n̩", "ɛ", "ɔ"})
lex = [["ɡ", "a", "ʁ", "tʰ", "n̩"], ["tʰ", "a", "k"], ["d", "a"], ["ʃ", "t", "a", "t"], ["ʃ", "t", "ʁ", "a"],
       ["m", "a"], ["b", "a"], ["n", "a"], ["v", "a"], ["f", "a"]] * 10
inv = PhoneInventory(V, legal_onsets=attested_onsets(lex, V, min_count=5))
syl = lambda p, v=inv: [" ".join(s) for s in syllabify(p, v)]
check("werden: off-glide stays with its vowel", syl(["v", "eː", "ɐ", "d", "ə", "n"]) == ["v eː ɐ", "d ə n"])
check("werden also without legal onsets", syllabify(["v", "eː", "ɐ", "d", "ə", "n"], V) ==
      [["v", "eː", "ɐ"], ["d", "ə", "n"]])
check("vier: one syllable", len(syllabify(["f", "iː", "ɐ"], inv)) == 1)
check("Feuer: ɐ after a diphthong is its own syllable", syl(["f", "ɔʏ", "ɐ"]) == ["f ɔʏ", "ɐ"])
check("Bauer: two syllables", len(syllabify(["b", "aw", "ɐ"], inv)) == 2)
check("Garten: illegal onset ʁtʰ is split", syl(["ɡ", "a", "ʁ", "tʰ", "n̩"]) == ["ɡ a ʁ", "tʰ n̩"])
check("hundert: nd is split", syl(["h", "ʊ", "n", "d", "ɐ", "t"]) == ["h ʊ n", "d ɐ t"])
check("legal cluster ʃtʁ stays together", syl(["a", "ʃ", "t", "ʁ", "a"]) == ["a", "ʃ t ʁ a"])
check("Dortmund: glide + cluster", syl(["d", "ɔ", "ɐ", "t", "m", "ʊ", "n", "t"]) == ["d ɔ ɐ t", "m ʊ n t"])
check("PhoneInventory is still a vowel frozenset", "a" in inv and isinstance(inv, frozenset))
check("PhoneInventory survives | with its onsets", (inv | {"y"}).legal_onsets == inv.legal_onsets)
check("syllable_strings uses the inventory", syllable_strings(["ɡ", "a", "ʁ", "tʰ", "n̩"], inv) == ["ɡ-a-ʁ", "tʰ-n̩"])

# --- cohort: words that END are an outcome, not dropped ---------------------
units = {"tisch": ["t", "ɪ", "ʃ"], "tischler": ["t", "ɪ", "ʃ", "l", "ɐ"], "tier": ["t", "iː", "ɐ"]}
w = {"tisch": 0.6, "tischler": 0.2, "tier": 0.2}
s, h = sequential_cohort_stats(w, units, units["tischler"])
# after t-ɪ-ʃ the cohort is {tisch (ends), tischler}: P(l) = 0.2 / 0.8
check("cohort: P(next | prefix) keeps the ended word in the denominator",
      math.isclose(s[3], -math.log2(0.2 / 0.8), rel_tol=1e-9))
check("cohort: entropy includes the end-of-word outcome",
      math.isclose(h[3], -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)), rel_tol=1e-9))
s2, _ = sequential_cohort_stats(w, units, units["tisch"])
check("cohort: first phoneme unchanged", math.isclose(s2[0], 0.0, abs_tol=1e-12))
sl, _ = single_level_stats("tischler", units["tischler"], {"tisch": 6.0, "tier": 2.0}, units)
check("single-level cohort: an absent target still has its own continuation", all(math.isfinite(v) for v in sl))

# --- MFA dictionary helpers --------------------------------------------------
check("_is_lexical rejects meta entries and punctuation",
      not _is_lexical("[bracketed]") and not _is_lexical(",") and _is_lexical("'s") and _is_lexical("über"))
check("G2P never receives punctuation", _run_mfa_g2p("german_mfa", [",", ".", "  "]) == {})

# --- embeddings keep German casing ------------------------------------------
check("_case_variants tries the written form first",
      _case_variants("Vorruhestand") == ["Vorruhestand", "vorruhestand"] and
      _case_variants("kam") == ["kam", "Kam"])


class _Emb:
    def vector(self, w):
        return {"a": np.array([1.0, 0.0, 0.2]), "b": np.array([0.9, 0.1, 0.0]),
                ".": np.array([-5.0, 3.0, 9.0])}.get(w)


d = semantic_distance_broderick(["a", ".", "b"], _Emb(), [0, 0, 0])
d_nop = semantic_distance_broderick(["a", "b"], _Emb(), [0, 0])
check("Broderick distance ignores punctuation (no value, not in the context mean)",
      d[1] is None and math.isclose(d[2], d_nop[1]))

# --- token trie -------------------------------------------------------------
trie = FormTrie([[5, 7], [5], [5, 7, 9], [8], [5, 6], []])
check("trie shares prefixes", trie.n_nodes == 1 + 5)
check("trie depth blocks are contiguous", list(trie.depth) == sorted(trie.depth))
check("trie paths reproduce every form",
      all(list(trie.paths[len(s)][trie.form_node[f] - trie.depth_start[len(s)]]) == s
          for f, s in enumerate([[5, 7], [5], [5, 7, 9], [8], [5, 6]])))
check("empty form maps to the root", trie.form_node[5] == 0)
flags = trie.subtree_any(np.array([False, False, True, False, False, False]))
check("subtree_any propagates to ancestors",
      flags[trie.form_node[2]] and flags[trie.form_node[0]] and flags[trie.form_node[1]]
      and not flags[trie.form_node[3]])

# --- 2026-09-24: spelling, compounds, onsets, POS context, SUBTLEX -----------
import tempfile
from linguistic_predictors.normalize_de import standardize_swiss_spelling
from linguistic_predictors.compounds_de import CompoundSplitter
from linguistic_predictors.syntactic import pos_surprisal_entropy
from linguistic_predictors.frequency_de import SubtlexDE

freq = {"äußerst": 489, "saß": 712, "dass": 141710, "masse": 237, "maße": 0, "weiß": 57465}.get
txt, ch = standardize_swiss_spelling("Äusserst selten sass sie, dass die Masse weiss war.",
                                     frequency=lambda w: freq(w.lower(), 0))
check("Swiss ss -> ß only where the ß form is the standard spelling",
      txt == "Äußerst selten saß sie, dass die Masse weiß war." and len(ch) == 3)

VV = frozenset({"a", "oː", "ʊ", "ɛ", "eː", "ɐ", "ə", "n̩"})
pron = {"stromanschluss": ["ʃ", "t", "ʁ", "oː", "m", "a", "n", "ʃ", "l", "ʊ", "s"],
        "strom": ["ʃ", "t", "ʁ", "oː", "m"], "anschluss": ["a", "n", "ʃ", "l", "ʊ", "s"],
        "fernseher": ["f", "ɛ", "ɐ", "n", "z", "eː", "ɐ"], "fern": ["f", "ɛ", "ɐ", "n"],
        "seher": ["z", "eː", "ɐ"], "eher": ["eː", "ɐ"],
        "garten": ["ɡ", "a", "ʁ", "tʰ", "n̩"], "gar": ["ɡ", "a", "ʁ"], "ten": ["tʰ", "ɛ", "n"]}
cnt = {"stromanschluss": 5, "strom": 1000, "anschluss": 800, "fernseher": 800, "fern": 900,
       "seher": 18, "eher": 2500, "garten": 900, "gar": 5000, "ten": 50}
sp = CompoundSplitter(pron, lambda w: cnt.get(w, 0), VV)
check("Strom|anschluss is split before the vowel-initial head", sp.boundaries("stromanschluss") == [5])
check("never Fern-s|eher: a linking 's' must be pronounced [s], not [z]", 5 not in sp.boundaries("fernseher"))
cnt["seher"] = 5000   # make Fern|seher frequent enough to split
sp = CompoundSplitter(pron, lambda w: cnt.get(w, 0), VV)
check("Fern|seher splits before the [z] when the parts are frequent", sp.boundaries("fernseher") == [4])
check("Gar|ten is rejected: 'ten' is not how Garten ends", sp.boundaries("garten") == [])
inv2 = PhoneInventory(VV | {"iː"}, legal_onsets={(), ("ʃ", "t", "ʁ"), ("m",), ("n",), ("ʃ", "l")},
                      morph_boundaries={tuple(pron["stromanschluss"]): (5,)})
check("morph boundary is a hard syllable edge (ʃtʁoːm.an.ʃlʊs, not ʃtʁoː.man...)",
      syllable_strings(pron["stromanschluss"], inv2) == ["ʃ-t-ʁ-oː-m", "a-n", "ʃ-l-ʊ-s"])
check("PhoneInventory keeps morph boundaries through |", (inv2 | {"y"}).morph_boundaries == inv2.morph_boundaries)
ons = attested_onsets([["s", "t", "a"]] * 20 + [["tʰ", "a"]] * 20 + [["ʃ", "t", "a"]] * 20, frozenset({"a"}))
check("loan onset s+C is not legal word-internally", ("s", "t") not in ons and ("ʃ", "t") in ons)
check("onset legality ignores aspiration (tʰ attested -> t legal)", ("t",) in ons)
invo = PhoneInventory(frozenset({"ɛ", "ɐ"}), legal_onsets=ons)
check("Fenster = fɛns.tɐ", syllable_strings(["f", "ɛ", "n", "s", "t", "ɐ"], invo) == ["f-ɛ-n-s", "t-ɐ"])


class _SlotTagger:
    """Tags a word ADJ when a noun follows it, else by a lexicon (like a
    contextual tagger that repairs slots)."""
    lex = {"roten": "ADJ", "Mantel": "NOUN", "laufen": "VERB", "Hut": "NOUN"}

    def tag_word_in_context(self, before, word, after):
        return "ADJ" if after[:1] == ["Mantel"] else self.lex.get(word, "X")


ps, ph, tag = pos_surprisal_entropy(["einen"], "roten", ["Mantel"],
                                    {"roten": 0.5, "laufen": 0.3, "Hut": 0.2}, _SlotTagger())
check("POS: candidates and the presented word are tagged without the words that follow; "
      "the full-sentence tag is only the label",
      tag == "ADJ" and math.isclose(ps, -math.log2(0.5)))
ps2, ph2, _ = pos_surprisal_entropy(["einen"], "roten", ["Hut"],
                                    {"roten": 0.5, "laufen": 0.3, "Hut": 0.2}, _SlotTagger())
check("POS: the same left context gives the same surprisal and entropy whatever follows",
      math.isclose(ps, ps2) and math.isclose(ph, ph2))


class _ZuTagger:
    """Infinitival "zu" (before a verb) is PART; with nothing after it, ADP."""
    def tag_word_in_context(self, before, word, after):
        if word == "zu":
            return "PART" if after[:1] == ["nehmen"] else "ADP"
        return "NOUN"


ps, ph, tag = pos_surprisal_entropy(["ein", "Bad"], "zu", ["nehmen"],
                                    {"zu": 0.9, "Haus": 0.1}, _ZuTagger())
check("POS: ambiguous 'zu' is scored on its left-context tag (no look-ahead), so P(tag) >= P(word)",
      tag == "PART" and math.isclose(ps, -math.log2(0.9)))
ps, ph, _ = pos_surprisal_entropy(["ein", "Bad"], "Hund", [], {"zu": 0.9, "Haus": 0.1}, _ZuTagger(),
                                  actual_p=0.01)
check("POS: a presented word outside the nucleus adds its own probability to its tag; "
      "entropy is the nucleus's alone",
      math.isclose(ps, -math.log2((0.1 + 0.01) / 1.01)) and math.isclose(ph, config.entropy([0.9, 0.1])))

from linguistic_predictors.pipeline import _nucleus
nuc = _nucleus({"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}, 0.9, 1, 0)
check("nucleus: smallest set reaching top_p, including the word that crosses it", list(nuc) == ["a", "b", "c"])
check("nucleus: at least top_k words", len(_nucleus({"a": 0.95, "b": 0.03, "c": 0.02}, 0.9, 2, 0)) == 2)

with tempfile.TemporaryDirectory() as tmp:
    f = Path(tmp) / "subtlex_de.tsv"
    f.write_text("Word\tWFfreqcount\tCUMfreqcount\tSUBTLEX\ndie\t90\t100\t100.0\nHaus\t9\t10\t10.0\n", encoding="utf-8")
    sx = SubtlexDE(str(f))
    check("SUBTLEX: corpus size from the per-million column", math.isclose(sx.n_tokens, 1e6))
    check("SUBTLEX: case-insensitive, Laplace-smoothed",
          math.isclose(sx.prob("Die"), 101 / (1e6 + 2)) and math.isclose(sx.prob("xyz"), 1 / (1e6 + 2)))
    check("SUBTLEX frequency prior: unseen dictionary words compete with count 1",
          math.isclose(sx.frequency_prior(["haus", "xyz"]).distribution()["xyz"], 1 / 12))

from linguistic_predictors.normalize_de import german_number, spoken_variants
from linguistic_predictors.candidates import _build_forms
check("German numerals parse (hundert, zweihundert, sechsundfünfzig, tausend)",
      [german_number(w) for w in ["hundert", "Zweihundert", "sechsundfünfzig", "tausend", "dreizehn"]]
      == [100, 200, 56, 1000, 13])
check("articles are never numerals", german_number("ein") is None and german_number("eine") is None)
check("spoken variants of a numeral include its digits", spoken_variants("hundert") == ["hundert", "Hundert", "100"])
fz, iz = _build_forms(["einhundert", "hundert", "haus"], [], "both")
check("'100' is read as the shortest numeral ('hundert', not 'einhundert')", dict(zip(fz, iz))["100"] == "hundert")
sp2 = CompoundSplitter({"sechsundfünfzig": ["z", "ɛ", "k", "s", "ʊ", "n", "t", "f", "ʏ", "n", "f", "ts", "ɪ", "ç"],
                        "sechs": ["z", "ɛ", "k", "s"], "und": ["ʊ", "n", "t"],
                        "fünfzig": ["f", "ʏ", "n", "f", "ts", "ɪ", "ç"]},
                       lambda w: 1, frozenset({"ɛ", "ʊ", "ʏ", "ɪ"}))
check("numeral compounds split around 'und' (sechs|und|fünfzig)", sp2.boundaries("sechsundfünfzig") == [4, 7])

from linguistic_predictors.lexicon_mfa import MFAPronDict
from linguistic_predictors.normalize_de import standard_spellings
check("dictionary key: lowercase + 'ss' for 'ß' (german_mfa has no ß)",
      MFAPronDict.canonical("Straße") == "strasse" and MFAPronDict.canonical("dass") == "dass")
sfreq = {"weiß": 57465, "weiss": 0, "dass": 141710, "daß": 0}.get
check("standard spelling of a dictionary key: weiss -> weiß, dass stays",
      standard_spellings("weiss", lambda w: sfreq(w, 0)) == ["weiß"]
      and standard_spellings("dass", lambda w: sfreq(w, 0)) == [])
fs, its = _build_forms(["weiss"], ["weiß"], "both", canonical=MFAPronDict.canonical,
                       spellings=lambda it: standard_spellings(it, lambda w: sfreq(w, 0)))
check("both spellings are forms of ONE item (probabilities summed)",
      set(fs) >= {"weiss", "Weiss", "weiß", "Weiß"}
      and {i for f, i in zip(fs, its) if f[:1].isalpha()} == {"weiss"})

# --- end to end on the toy resources -----------------------------------------
import subprocess
from linguistic_predictors.pipeline import PipelineOptions, compute_predictors
from linguistic_predictors.toy import toy_components

df, meta = compute_predictors("Die Katze saß auf der Matte. Der Hund rannte schnell.", toy_components(),
                              PipelineOptions(), log=lambda *a: None)
check("toy pipeline: one row per unit", len(df) == 12 and list(df.word[:3]) == ["Die", "Katze", "saß"])
check("toy pipeline: finite, non-negative lexical surprisal",
      df.lexical_surprisal.notna().all() and (df.lexical_surprisal >= 0).all())
check("toy pipeline: content words get a semantic value, function words none",
      df.loc[df.word == "Katze", "semantic_prediction_error"].notna().all()
      and df.loc[df.word == "Die", "semantic_prediction_error"].isna().all())
check("toy pipeline: one phoneme value per phoneme", len(df.loc[df.word == "Katze", "phon_surprisal_hier"].iloc[0]) == 4)
check("toy pipeline: punctuation rows have no cohort values", df.loc[df.word == ".", "phon_surprisal_hier"].map(len).eq(0).all())
with tempfile.TemporaryDirectory() as tmp:
    r = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "run_story.py"), "--text",
                        "Die Katze saß auf der Matte.", "--backend", "toy", "--out", tmp],
                       capture_output=True, text=True)
    check("run_story.py --backend toy exits 0", r.returncode == 0)
    check("run_story.py writes every output file",
          all((Path(tmp) / f).exists() for f in ("words.csv", "phonemes.csv", "syllables.csv",
                                                  "qa_sample.csv", "run_config.json")))

print()
print("All checks passed." if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
