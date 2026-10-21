"""Tests for mtrf_export.py and textgrid_align.py: a toy pipeline run
flattened to one row per phoneme/syllable, then merged with a hand-written
TextGrid (long text format) standing in for MFA forced alignment.

    python tests/test_mtrf_export.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from linguistic_predictors.pipeline import compute_predictors
from linguistic_predictors.syllabic import DE_SAMPA_VOWELS
from linguistic_predictors.toy import toy_components
from linguistic_predictors.mtrf_export import flatten_phoneme_level, flatten_syllable_level
from linguistic_predictors.textgrid_align import (
    parse_textgrid, group_phones_by_word,
    merge_onsets_into_phoneme_table, merge_onsets_into_syllable_table,
)

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# --- build a real (toy-component) pipeline run to flatten ---
comps = toy_components()
pron_dict = comps["pron_dict"]
text = "Die Katze rannte."
df, _ = compute_predictors(text, comps, log=lambda *a: None)
check("pipeline produced 4 rows (die / katze / rannte / .)", len(df) == 4)

# --- flatten_phoneme_level: every phoneme value preserved, right symbols attached ---
flat_phon = flatten_phoneme_level(df, pron_dict)
check("flatten_phoneme_level: total phoneme rows == sum of per-word phoneme counts",
      len(flat_phon) == sum(len(pron_dict.phonemes(w)) for w in df.word))
die_rows = flat_phon[flat_phon["word"] == "Die"].sort_values("phon_index")
check("flatten_phoneme_level: 'Die' has 2 phoneme rows", len(die_rows) == 2)
check("flatten_phoneme_level: 'Die' phoneme symbols are d, i: in order",
      list(die_rows["phoneme"]) == ["d", "i:"])
check("flatten_phoneme_level: no NaN in phon_surprisal_hier", flat_phon["phon_surprisal_hier"].notna().all())

# --- flatten_syllable_level ---
flat_syll = flatten_syllable_level(df, pron_dict, DE_SAMPA_VOWELS)
katze_rows = flat_syll[flat_syll["word"] == "Katze"].sort_values("syll_index")
check("flatten_syllable_level: 'Katze' has 2 syllables", len(katze_rows) == 2)
check("flatten_syllable_level: 'Katze' syllable strings are k-a, ts-@",
      list(katze_rows["syllable"]) == ["k-a", "ts-@"])

# --- synthetic TextGrid parsing ---
synthetic_textgrid = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 1.6
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0.0
        xmax = 1.6
        intervals: size = 4
        intervals [1]:
            xmin = 0.0
            xmax = 0.0
            text = ""
        intervals [2]:
            xmin = 0.0
            xmax = 0.3
            text = "die"
        intervals [3]:
            xmin = 0.3
            xmax = 0.9
            text = "katze"
        intervals [4]:
            xmin = 0.9
            xmax = 1.6
            text = "rannte"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0.0
        xmax = 1.6
        intervals: size = 11
        intervals [1]:
            xmin = 0.0
            xmax = 0.15
            text = "d"
        intervals [2]:
            xmin = 0.15
            xmax = 0.3
            text = "i:"
        intervals [3]:
            xmin = 0.3
            xmax = 0.4
            text = "k"
        intervals [4]:
            xmin = 0.4
            xmax = 0.55
            text = "a"
        intervals [5]:
            xmin = 0.55
            xmax = 0.7
            text = "ts"
        intervals [6]:
            xmin = 0.7
            xmax = 0.9
            text = "@"
        intervals [7]:
            xmin = 0.9
            xmax = 1.0
            text = "R"
        intervals [8]:
            xmin = 1.0
            xmax = 1.15
            text = "a"
        intervals [9]:
            xmin = 1.15
            xmax = 1.3
            text = "n"
        intervals [10]:
            xmin = 1.3
            xmax = 1.45
            text = "t"
        intervals [11]:
            xmin = 1.45
            xmax = 1.6
            text = "@"
'''
with tempfile.NamedTemporaryFile(mode="w", suffix=".TextGrid", delete=False, encoding="utf-8") as f:
    f.write(synthetic_textgrid)
    tg_path = f.name

tiers = parse_textgrid(tg_path)
check("parse_textgrid: found both tiers", set(tiers.keys()) == {"words", "phones"})
check("parse_textgrid: 'words' tier has 4 intervals", len(tiers["words"]) == 4)
check("parse_textgrid: 'phones' tier has 11 intervals", len(tiers["phones"]) == 11)

groups = group_phones_by_word(tiers)
check("group_phones_by_word: 3 spoken words (silence dropped)", len(groups) == 3)
check("group_phones_by_word: 'katze' got 4 phones", len(groups[1]["phones"]) == 4)

# --- merge onsets into the phoneme table ---
merged_phon = merge_onsets_into_phoneme_table(flat_phon, groups, strict=True)
check("merge_onsets_into_phoneme_table: 11 rows (punctuation '.' correctly excluded)",
      len(merged_phon) == 11)
check("merge_onsets_into_phoneme_table: sorted by onset, starts at 0.0",
      merged_phon.iloc[0]["onset"] == 0.0)
katze_k = merged_phon[(merged_phon["word"] == "Katze") & (merged_phon["phon_index"] == 0)]
check("merge_onsets_into_phoneme_table: 'Katze' phon 0 ('k') onset/offset correct",
      not katze_k.empty and katze_k.iloc[0]["onset"] == 0.3 and katze_k.iloc[0]["offset"] == 0.4)
check("merge_onsets_into_phoneme_table: aligned_phone matches TextGrid text",
      katze_k.iloc[0]["aligned_phone"] == "k")

# --- merge onsets into the syllable table ---
merged_syll = merge_onsets_into_syllable_table(flat_syll, groups, DE_SAMPA_VOWELS, strict=True)
check("merge_onsets_into_syllable_table: 5 syllables total (die:1 + katze:2 + rannte:2)",
      len(merged_syll) == 5)
katze_syl0 = merged_syll[(merged_syll["word"] == "Katze") & (merged_syll["syll_index"] == 0)]
check("merge_onsets_into_syllable_table: 'Katze' syllable 0 ('k-a') onset/offset spans k+a",
      not katze_syl0.empty and katze_syl0.iloc[0]["onset"] == 0.3 and katze_syl0.iloc[0]["offset"] == 0.55)

# --- strict mismatch detection: deliberately break word-count alignment ---
try:
    merge_onsets_into_phoneme_table(flat_phon, groups[:-1], strict=True)
    check("merge_onsets_into_phoneme_table: raises on word-count mismatch (strict=True)", False)
except ValueError:
    check("merge_onsets_into_phoneme_table: raises on word-count mismatch (strict=True)", True)

# non-strict mode should warn and drop rather than raise
result = merge_onsets_into_phoneme_table(flat_phon, groups[:-1], strict=False)
check("merge_onsets_into_phoneme_table: strict=False proceeds without raising", result is not None)

# --- equal-count pronunciation mismatch: same number of phones, different phones ---
import copy
swapped = copy.deepcopy(groups)
k0 = swapped[1]["phones"][0]
swapped[1]["phones"][0] = (k0[0], k0[1], "g")           # "gatze": 4 phones, like Katze
try:
    merge_onsets_into_phoneme_table(flat_phon, swapped, strict=True)
    check("phoneme merge: raises on an equal-count pronunciation mismatch", False)
except ValueError:
    check("phoneme merge: raises on an equal-count pronunciation mismatch", True)
check("phoneme merge (lenient): drops only the mismatched word",
      set(merge_onsets_into_phoneme_table(flat_phon, swapped, strict=False)["word"]) == {"Die", "rannte"})
check("phoneme merge: a declared equivalence is accepted",
      len(merge_onsets_into_phoneme_table(flat_phon, swapped, strict=True, equivalences={"g": "k"})) == 11)
try:
    merge_onsets_into_syllable_table(flat_syll, swapped, DE_SAMPA_VOWELS, strict=True)
    check("syllable merge: raises on an equal-count pronunciation mismatch", False)
except ValueError:
    check("syllable merge: raises on an equal-count pronunciation mismatch", True)
# syllables are cut where the pipeline's syllables end, not re-derived from the aligned phones
odd = flat_syll.copy()
kat = odd.index[odd["word"] == "Katze"]
odd.loc[kat, "syllable"] = ["k-a-ts", "@"]
ms = merge_onsets_into_syllable_table(odd, groups, DE_SAMPA_VOWELS, strict=True)
k1 = ms[(ms["word"] == "Katze") & (ms["syll_index"] == 1)]
check("syllable merge: follows the pipeline's own syllable boundaries ('k-a-ts' | '@')",
      not k1.empty and k1.iloc[0]["onset"] == 0.7 and k1.iloc[0]["offset"] == 0.9)

Path(tg_path).unlink()

print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed.")
sys.exit(1 if failures else 0)
