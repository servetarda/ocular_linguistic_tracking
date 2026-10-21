"""Statistical tests of the predictors computed for the story.

    python scripts/story_statistics.py results/

Reads words.csv and phonemes.csv from a results folder and tests whether the
predictors behave the way the theory behind Heilbron et al. (2022) says they
should. Every test is one-sided, in the predicted direction, and the p-values
are Holm-corrected across all tests.

  1. The story's anomalous words are less predictable than the same words
     in their normal sentences elsewhere in the story (paired Wilcoxon
     signed-rank; the control value is the mean over the word's normal
     occurrences), and less predictable than the other nouns (Mann-Whitney
     U), at the lexical, POS, semantic and first-phoneme levels. The
     anomalous words are listed in stimuli/violations.csv, each with the
     words that precede it, so that the right occurrence is used.
  2. Frequent words are less surprising: lexical surprisal rises with
     unigram (frequency) surprisal (Spearman).
  3. Content words are less predictable than function words (Mann-Whitney U).
  4. Context helps phoneme prediction: per word, the mean context-based
     (hierarchical) phoneme surprisal is lower than the mean frequency-based
     one (Wilcoxon signed-rank, paired, one value per word).
  5. Phonemes become more predictable further into a word: per word (two or
     more phonemes), the first phoneme is more surprising than the mean of
     the later ones (Wilcoxon signed-rank, one value per word).

The unit of every test is the word (the sounds of one word are not
independent). All tests use one story, so tests 2-5 are descriptive sanity
checks of the predictors, not confirmatory inference about listeners.

It also prints the correlations and variance inflation factors (VIF) of the
word-level predictors that enter the regression. These only flag gross
overlap; whether each predictor explains unique variance must be tested in
the time-lagged (TRF) model itself, by held-out model comparison.

Results are written to <results>/statistics.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

VIOLATIONS = Path(__file__).resolve().parent.parent / "stimuli" / "violations.csv"

LEVELS = ["lexical_surprisal", "pos_surprisal", "semantic_prediction_error", "first_phoneme_surprisal"]

UNITS = {"semantic_prediction_error": "(cosine distance)"}

# the word-level regressors of the analysis (Heilbron et al. 2022): predictors and controls
WORD_PREDICTORS = ["lexical_surprisal", "lexical_entropy", "pos_surprisal", "semantic_prediction_error",
                   "semantic_distance", "unigram_surprisal"]


def first_phoneme_surprisal(phonemes: pd.DataFrame) -> pd.Series:
    """Hierarchical surprisal of each word's first phoneme, by word position."""
    first = phonemes[phonemes["phon_index"] == 0]
    return first.set_index("word_position")["phon_surprisal_hier"]


def find_violations(units: pd.DataFrame, violations: pd.DataFrame) -> pd.Index:
    """Row index of each anomalous word: the occurrence of `word` whose
    preceding units are `preceded_by`."""
    text = list(units["word"])
    rows = []
    for word, preceded_by in zip(violations["word"], violations["preceded_by"]):
        before = preceded_by.split()
        hits = [i for i, w in enumerate(text) if w == word and text[i - len(before):i] == before]
        if len(hits) != 1:
            raise ValueError(f"{preceded_by} {word}: found {len(hits)} times in the text, expected once")
        rows.append(units.index[hits[0]])
    return pd.Index(rows)


def control_means(words: pd.DataFrame, anomalous: pd.Index, col: str) -> pd.Series:
    """For each anomalous row, the mean of `col` over the same word's
    occurrences that are not anomalous."""
    normal = words.drop(index=anomalous)
    means = normal.groupby("word")[col].mean()
    return words.loc[anomalous, "word"].map(means).set_axis(anomalous)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the original order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--violations", type=Path, default=VIOLATIONS,
                    help="CSV of the anomalous words (columns word, preceded_by)")
    args = ap.parse_args()

    units = pd.read_csv(args.results / "words.csv")
    phonemes = pd.read_csv(args.results / "phonemes.csv")
    anomalous = find_violations(units, pd.read_csv(args.violations))
    words = units[~units["is_punctuation"]].copy()
    words["first_phoneme_surprisal"] = words["position"].map(first_phoneme_surprisal(phonemes))

    tests = []   # (test, n, statistic, p, detail)

    # 1a. anomalous words vs the same words in normal sentences
    print(f"Anomalous words: {len(anomalous)} ({', '.join(words.loc[anomalous, 'word'])})")
    for col in LEVELS:
        pairs = pd.DataFrame({"anomalous": words.loc[anomalous, col],
                              "normal": control_means(words, anomalous, col)}).dropna()
        if len(pairs) < 2:
            continue
        w = stats.wilcoxon(pairs["anomalous"], pairs["normal"], alternative="greater")
        tests.append((f"1a. anomalous > same word in normal sentence: {col}", len(pairs), w.statistic, w.pvalue,
                      f"median {pairs['anomalous'].median():.2f} vs {pairs['normal'].median():.2f} "
                      f"{UNITS.get(col, 'bits')}"))

    # 1b. anomalous words vs the other nouns
    nouns = words.index[(words["pos"] == "NOUN") & ~words.index.isin(anomalous)]
    for col in LEVELS:
        a = words.loc[anomalous, col].dropna()
        b = words.loc[nouns, col].dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        u = stats.mannwhitneyu(a, b, alternative="greater")
        tests.append((f"1b. anomalous > other nouns: {col}", f"{len(a)} vs {len(b)}", u.statistic, u.pvalue,
                      f"median {a.median():.2f} vs {b.median():.2f} {UNITS.get(col, 'bits')}"))

    # 2. frequency
    both = words[["lexical_surprisal", "unigram_surprisal"]].dropna()
    r = stats.spearmanr(both["lexical_surprisal"], both["unigram_surprisal"], alternative="greater")
    tests.append(("2. lexical surprisal rises with frequency surprisal", len(both), r.statistic, r.pvalue,
                  f"Spearman rho = {r.statistic:.2f}"))

    # 3. content vs function words
    content = words.loc[words["is_content_word"], "lexical_surprisal"].dropna()
    function = words.loc[~words["is_content_word"], "lexical_surprisal"].dropna()
    u = stats.mannwhitneyu(content, function, alternative="greater")
    tests.append(("3. content words > function words: lexical_surprisal", f"{len(content)} vs {len(function)}",
                  u.statistic, u.pvalue, f"median {content.median():.2f} vs {function.median():.2f} bits"))

    # 4. context vs frequency: one value per word (its mean phoneme surprisal)
    ph = phonemes.groupby("word_position")[["phon_surprisal_hier", "phon_surprisal_freq"]].mean().dropna()
    w = stats.wilcoxon(ph["phon_surprisal_hier"], ph["phon_surprisal_freq"], alternative="less")
    tests.append(("4. context-based < frequency-based phoneme surprisal (per word)", len(ph), w.statistic,
                  w.pvalue, f"median {ph['phon_surprisal_hier'].median():.2f} vs "
                            f"{ph['phon_surprisal_freq'].median():.2f} bits per phoneme"))

    # 5. position within the word: first phoneme vs the later ones, one value per word
    multi = phonemes[phonemes["n_phonemes"] >= 2]
    first = multi[multi["phon_index"] == 0].set_index("word_position")["phon_surprisal_hier"]
    later = multi[multi["phon_index"] > 0].groupby("word_position")["phon_surprisal_hier"].mean()
    fl = pd.DataFrame({"first": first, "later": later}).dropna()
    w = stats.wilcoxon(fl["first"], fl["later"], alternative="greater")
    tests.append(("5. first phoneme > later phonemes of the same word", len(fl), w.statistic, w.pvalue,
                  f"median {fl['first'].median():.2f} vs {fl['later'].median():.2f} bits"))

    table = pd.DataFrame(tests, columns=["test", "n", "statistic", "p", "detail"])
    table["p_holm"] = holm(table["p"])
    table["significant"] = table["p_holm"] < 0.05
    table.to_csv(args.results / "statistics.csv", index=False)

    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print("\nTests (one-sided, Holm-corrected):")
    print(table[["test", "n", "detail", "p", "p_holm", "significant"]].to_string(
        index=False, formatters={"p": "{:.2g}".format, "p_holm": "{:.2g}".format}))

    # collinearity of the word-level predictors
    cols = [c for c in WORD_PREDICTORS if c in words.columns]
    corr = words[cols].corr(method="pearson")
    print("\nCorrelations between word-level predictors (Pearson r):")
    print(corr.round(2).to_string())
    high = [(a, b, corr.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if abs(corr.loc[a, b]) > 0.7]
    print("\nPairs with |r| > 0.7:", ", ".join(f"{a} ~ {b} ({r:.2f})" for a, b, r in high) or "none")
    complete = words[cols].dropna()
    vif = pd.Series(np.diag(np.linalg.inv(complete.corr().to_numpy())), index=cols)
    print(f"\nVariance inflation factors (words with every predictor, n = {len(complete)}):")
    print(vif.round(2).to_string())


if __name__ == "__main__":
    main()
