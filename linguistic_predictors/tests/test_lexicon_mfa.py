"""Tests for lexicon_mfa.py on a synthetic dictionary file (both MFA formats:
plain `word<TAB>phones` and with probability columns) -- no MFA install needed.

    python tests/test_lexicon_mfa.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from linguistic_predictors.lexicon_mfa import MFAPronDict, infer_vowel_phones

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# --- vowel inference on real IPA symbols ---
phones = ["m", "a", "t", "ə", "ʃ", "ɪ", "ç", "ŋ", "aɪ", "ʊ", "n", "d", "eː", "ɐ"]
vowels = infer_vowel_phones(phones)
check("infer_vowel_phones: 'a' is a vowel", "a" in vowels)
check("infer_vowel_phones: 'ə' (schwa) is a vowel", "ə" in vowels)
check("infer_vowel_phones: 'aɪ' (diphthong) is a vowel", "aɪ" in vowels)
check("infer_vowel_phones: 'eː' (long vowel with length mark) is a vowel", "eː" in vowels)
check("infer_vowel_phones: 'ɐ' is a vowel", "ɐ" in vowels)
check("infer_vowel_phones: 'm' is NOT a vowel", "m" not in vowels)
check("infer_vowel_phones: 'ʃ' is NOT a vowel", "ʃ" not in vowels)
check("infer_vowel_phones: 'ŋ' is NOT a vowel", "ŋ" not in vowels)

# --- synthetic MFA dictionary file: mixed plain + probabilistic formats ---
synthetic_dict = "\n".join([
    "mädchen\tm ɛː t ç ə n",                      # plain format
    "hund\t1.0\t0.1\t0.05\th ʊ n t",               # probabilistic format
    "hund\t0.3\t0.1\t0.05\th u n t",               # lower-prob alt pronunciation -- should be ignored
    "der\td eː ɐ",
    "katze\tk a t s ə",
])
with tempfile.NamedTemporaryFile(mode="w", suffix=".dict", delete=False, encoding="utf-8") as f:
    f.write(synthetic_dict)
    dict_path = f.name

pron_dict = MFAPronDict(dict_path, g2p_model_name=None)  # disable G2P fallback for this test
check("MFAPronDict: plain-format entry parsed", pron_dict.phonemes("mädchen") == ["m", "ɛː", "t", "ç", "ə", "n"])
check("MFAPronDict: probabilistic-format entry parsed",
      pron_dict.phonemes("hund") == ["h", "ʊ", "n", "t"])
check("MFAPronDict: keeps highest-probability pronunciation when duplicated",
      pron_dict.phonemes("hund") != ["h", "u", "n", "t"])
check("MFAPronDict: case-insensitive lookup", pron_dict.phonemes("Katze") == ["k", "a", "t", "s", "ə"])
check("MFAPronDict: unknown word with no G2P fallback returns []", pron_dict.phonemes("unbekannteswort") == [])

report = pron_dict.coverage_report(["mädchen", "hund", "der", "katze", "unbekannteswort"])
check("coverage_report: counts total_words correctly", report["total_words"] == 5)
check("coverage_report: counts in_static_dictionary correctly", report["in_static_dictionary"] == 4)
check("coverage_report: counts unresolved correctly", report["unresolved"] == 1)

check("MFAPronDict: automatic vowel inference found vowels from loaded dict",
      len(pron_dict.vowel_phones) > 0 and "a" in pron_dict.vowel_phones)

Path(dict_path).unlink()

print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed.")
sys.exit(1 if failures else 0)
