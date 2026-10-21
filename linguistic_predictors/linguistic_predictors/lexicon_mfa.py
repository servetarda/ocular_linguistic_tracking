"""Pronunciations from the Montreal Forced Aligner's German dictionary
(`german_mfa`, IPA, ~143k words), with MFA's German G2P model for words the
dictionary lacks -- in the same phone inventory, and the same resource the
audio is force-aligned with.

- Entries: the most probable pronunciation per word; meta-entries such as
  "[bracketed]" are skipped.
- Spelling: the dictionary writes every "ß" as "ss" ("strasse", "weiss");
  `canonical()` maps any spelling to that key.
- Vowel inventory, legal onsets and (optionally) compound boundaries are
  inferred from the dictionary itself and exposed as `vowel_phones`, a
  `syllabic.PhoneInventory`.
- G2P runs the `mfa` CLI, which usually lives in its own conda environment;
  `find_mfa_executable` locates it.

Setup: `conda install -c conda-forge montreal-forced-aligner`, then
`mfa model download dictionary german_mfa` and `mfa model download g2p
german_mfa`; `mfa model inspect dictionary german_mfa` prints the path.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence

from .syllabic import PhoneInventory, attested_onsets


def _is_lexical(word: str) -> bool:
    """A real word (letters, optionally with internal ' or -), not a
    dictionary meta-entry like "[bracketed]" / "<unk>" or a punctuation token."""
    core = word.replace("'", "").replace("-", "")
    return bool(core) and core.isalpha()


def find_mfa_executable() -> Optional[str]:
    """Locate the `mfa` CLI. MFA normally lives in its own conda env (it pins
    kaldi/openfst versions that clash with torch/spaCy), so it is usually not
    on this environment's PATH. Search order: $MFA_BIN, PATH, then
    `envs/*/bin/mfa` under the usual conda roots."""
    env = os.environ.get("MFA_BIN")
    if env and Path(env).exists():
        return env
    on_path = shutil.which("mfa")
    if on_path:
        return on_path
    roots = ["/opt/anaconda3", "/opt/miniconda3", "/opt/homebrew/Caskroom/miniforge/base",
             "~/anaconda3", "~/miniconda3", "~/miniforge3", "~/mambaforge", "/opt/conda"]
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        roots.insert(0, str(Path(conda_exe).parent.parent))
    for root in roots:
        for cand in sorted(glob.glob(str(Path(root).expanduser() / "envs" / "*" / "bin" / "mfa"))):
            return cand
    return None


def _mfa_subprocess_env(mfa_path: str) -> Dict[str, str]:
    """MFA shells out to openfst/kaldi binaries that sit next to `mfa` in its
    env's bin/ -- they must be on PATH or `mfa g2p` dies with "Could not find
    'fstcompile'". KMP_DUPLICATE_LIB_OK avoids the macOS double-libomp abort."""
    env = dict(os.environ)
    env["PATH"] = str(Path(mfa_path).parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env


def _run_mfa_g2p(g2p_model_name: str, words: Sequence[str]) -> Dict[str, List[str]]:
    """Run `mfa g2p` on a list of words and return {word: [phones]}.

    MFA's `mfa g2p` reads an input (a text file with one word per line is
    accepted) and WRITES a dictionary-format file -- it does NOT do
    stdin/stdout, and the positional argument order changed between MFA
    releases (2.0: `<g2p_model> <input> <output>`; 2.2+/3.x:
    `<input> <g2p_model> <output>`). This tries both orders and uses
    whichever produces an output file, so it works regardless of the
    installed MFA version. Returns {} (rather than raising) if MFA is not
    installed or every invocation fails -- the caller treats that as "no
    pronunciation", same as any other OOV.
    """
    words = [w for w in dict.fromkeys(w.lower() for w in words) if w and _is_lexical(w)]
    if not words:
        return {}
    mfa = find_mfa_executable()
    if mfa is None:
        print("[MFAPronDict] `mfa` executable not found (not on PATH, no $MFA_BIN, no "
              "conda env with it) -- G2P fallback disabled.")
        return {}
    env = _mfa_subprocess_env(mfa)
    with tempfile.TemporaryDirectory(prefix="mfa_g2p_") as d:
        wl = Path(d) / "oov_words.txt"
        wl.write_text("\n".join(words) + "\n", encoding="utf-8")
        out = Path(d) / "oov_dict.txt"
        arg_orders = (   # current MFA (2.2+/3.x) order first -- each failed try costs a slow MFA start-up
            [str(wl), g2p_model_name, str(out)],
            [g2p_model_name, str(wl), str(out)],
        )
        for args in arg_orders:
            out.unlink(missing_ok=True)
            try:
                subprocess.run(
                    [mfa, "g2p", *args, "--clean", "--num_jobs", "1"],
                    capture_output=True, text=True, timeout=1800, env=env,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return {}
            if out.exists() and out.stat().st_size > 0:
                result: Dict[str, List[str]] = {}
                for line in out.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    fields = line.split("\t") if "\t" in line else line.split()
                    if len(fields) < 2:
                        continue
                    w = fields[0].lower()
                    # phones = everything after the last field that parses as
                    # a float (probability columns), else fields[1:]
                    phones: List[str] = []
                    for cut in range(len(fields) - 1, 0, -1):
                        try:
                            float(fields[cut])
                            phones = fields[cut + 1:]
                            break
                        except ValueError:
                            continue
                    if not phones:
                        phones = fields[1:]
                    if "\t" in line and len(fields) == 2:
                        phones = fields[1].split()
                    if w not in result and phones:
                        result[w] = phones
                if result:
                    return result
    return {}

# Base IPA vowel letters (stripped of length marks/diacritics/tone/stress).
# Used to automatically classify a dictionary's phone inventory into
# vowels/consonants WITHOUT needing to know MFA's exact phone set in
# advance -- this is what makes syllabification (`syllabic.py`) work
# automatically for whatever phone set your downloaded dictionary actually
# uses, rather than relying on a hand-typed (and possibly wrong) vowel list.
_IPA_VOWEL_BASE_LETTERS = set("aeiouyœøɪʊɛɔəɐɑæɜɵʉɨʏ")


def _base_letter(phone: str) -> str:
    """Strip IPA length marks (ː, ˑ), stress marks (ˈ, ˌ), and combining
    diacritics from a phone symbol, returning its core letter(s) in
    lowercase, for vowel/consonant classification."""
    # decompose and drop combining marks (category "Mn")
    decomposed = unicodedata.normalize("NFD", phone)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    stripped = stripped.replace("ː", "").replace("ˑ", "").replace("ˈ", "").replace("ˌ", "")
    return stripped.lower()


# Syllabic-consonant diacritics (combining vertical line below U+0329,
# combining vertical line above U+030D). MFA's German dictionary writes the
# reduced -en/-el/-em endings as syllabic consonants ("n̩", "l̩", "m̩"), which
# ARE syllable nuclei -- e.g. "aufgeben" = /aw f ɡ eː b n̩/ is 3 syllables,
# not 2. Without this they'd be classed as ordinary consonants and undercount
# syllables for the very common German words ending that way.
_SYLLABIC_MARKS = ("̩", "̍")


def infer_vowel_phones(all_phones: Sequence[str]) -> FrozenSet[str]:
    """Given every distinct phone symbol appearing in a dictionary, return
    the subset that act as syllable nuclei: any phone whose base IPA letter
    is a vowel letter (diphthongs written as two characters count if ANY
    base letter is a vowel), PLUS any phone carrying a syllabic-consonant
    diacritic ("n̩", "l̩", "m̩")."""
    vowels = set()
    for p in all_phones:
        if any(m in p for m in _SYLLABIC_MARKS):
            vowels.add(p)
            continue
        base = _base_letter(p)
        if any(ch in _IPA_VOWEL_BASE_LETTERS for ch in base):
            vowels.add(p)
    return frozenset(vowels)


class MFAPronDict:
    """`phonemic.PronouncingDictionary` backed by a real MFA-format
    pronunciation dictionary file, with automatic G2P fallback for
    out-of-vocabulary words via the `mfa g2p` CLI.

    Parameters
    ----------
    dict_path : path to the downloaded MFA dictionary file (e.g. the path
        `mfa model inspect dictionary german_mfa` prints).
    g2p_model_name : name of a downloaded MFA G2P model (e.g. "german_mfa")
        to fall back on for words missing from `dict_path`. Pass None to
        disable the fallback (OOV words then just return no phonemes, same
        as any other PronouncingDictionary implementation).
    """

    def __init__(self, dict_path: str, g2p_model_name: Optional[str] = "german_mfa"):
        self._phonemes: Dict[str, List[str]] = {}
        self._probs: Dict[str, float] = {}
        self._load(Path(dict_path))
        self._g2p_model_name = g2p_model_name
        self._g2p_cache: Dict[str, List[str]] = {}
        self._refresh_inventory()

    def _refresh_inventory(self) -> None:
        """Nucleus phones (auto-inferred) + the legal onsets attested in this
        dictionary (+ compound boundaries, once enabled), packed into one
        `syllabic.PhoneInventory` so every syllabification downstream uses
        them."""
        all_phones = {p for phons in self._phonemes.values() for p in phons}
        vowels = infer_vowel_phones(all_phones)
        onsets = attested_onsets(self._phonemes.values(), vowels)
        self.vowel_phones = PhoneInventory(vowels, legal_onsets=onsets,
                                           morph_boundaries=getattr(self, "_morph", None))

    def enable_compound_boundaries(self, count) -> int:
        """Split every dictionary word (and every G2P-resolved word -- call
        this after `precompute_oov`) into compound parts / prefix + stem with
        `compounds_de.CompoundSplitter`, using `count(word)` (e.g. SUBTLEX-DE)
        for the frequency criterion. Returns the number of words split."""
        from .compounds_de import CompoundSplitter
        sp = CompoundSplitter(self._phonemes, count, infer_vowel_phones(
            {p for phons in self._phonemes.values() for p in phons}))
        morph = {}
        for w, ph in self._phonemes.items():
            cuts = sp.boundaries(w)
            if cuts:
                morph[tuple(ph)] = tuple(cuts)
        self._morph = morph
        self._refresh_inventory()
        return len(morph)

    def _load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                fields = line.split("\t") if "\t" in line else line.split()
                if len(fields) < 2:
                    continue
                word = fields[0].lower()
                if not _is_lexical(word):
                    # meta-entries such as "[bracketed]" (phone "spn") are not
                    # words and must not enter any cohort
                    continue
                # MFA's plain format is "word<TAB>phones..."; its
                # probabilistic format inserts numeric columns (pronunciation
                # probability, silence probabilities) between the word and
                # the phones. Whichever it is, the phones are always
                # whatever comes after the last field that parses as a
                # float -- simplest robust rule: try to read fields[1] as a
                # probability; if it parses, phones start at fields[-1]'s
                # tab-split (phones are space-separated within that last
                # tab field); otherwise fields[1:] (space-joined) are the
                # phones directly.
                prob = 1.0
                if len(fields) > 2:
                    try:
                        prob = float(fields[1])
                        phones = fields[-1].split()
                    except ValueError:
                        phones = " ".join(fields[1:]).split()
                else:
                    phones = fields[1].split()

                # keep the highest-probability pronunciation seen per word
                # (real German words can have >1 listed pronunciation, e.g.
                # regional variants; we need exactly one target
                # transcription per word for the cohort model)
                if word not in self._phonemes or prob > self._probs.get(word, -1):
                    self._phonemes[word] = phones
                    self._probs[word] = prob

    def _g2p_fallback(self, word: str) -> List[str]:
        if self._g2p_model_name is None:
            return []
        if word in self._g2p_cache:
            return self._g2p_cache[word]
        # `mfa g2p` spins up a fair amount of machinery per invocation, so a
        # real run should resolve every OOV word up front in ONE call via
        # `precompute_oov()`; this single-word path is only a convenience
        # fallback for ad-hoc lookups.
        if not _is_lexical(word):
            return []
        phones = _run_mfa_g2p(self._g2p_model_name, [word]).get(word.lower(), [])
        self._g2p_cache[word] = phones
        if phones:
            self.vowel_phones = self.vowel_phones | infer_vowel_phones(phones)
        return phones

    @staticmethod
    def canonical(word: str) -> str:
        """The dictionary's spelling of a word: lowercase, and "ss" for "ß" --
        german_mfa contains no "ß" at all ("strasse", "weiss", "sass"), so a
        standard-spelled "Straße" would otherwise never be found and fall
        through to G2P. This is also the item key the cohort model uses."""
        return word.lower().replace("ß", "ss")

    def precompute_oov(self, words: Sequence[str]) -> int:
        """Batch-run G2P for every word in `words` not already in the
        dictionary, using ONE `mfa g2p` call instead of one-per-word (much
        faster for a real study's vocabulary). Call this once up front on
        your full text's vocabulary before processing. Returns the number
        of words that needed G2P."""
        oov = sorted({self.canonical(w) for w in words
                      if _is_lexical(w) and self.canonical(w) not in self._phonemes})
        if not oov or self._g2p_model_name is None:
            return 0
        resolved = _run_mfa_g2p(self._g2p_model_name, oov)
        if not resolved:
            print(
                f"[MFAPronDict] G2P produced no output for {len(oov)} out-of-vocabulary "
                "word(s) -- they will have NO phoneme/syllable values. Check that "
                "`mfa` is on PATH and `mfa model download g2p german_mfa` has been run; "
                "try `mfa g2p german_mfa <wordlist.txt> <out.txt>` by hand to see the error."
            )
        for w, phones in resolved.items():
            self._phonemes[w] = phones
            self._g2p_cache[w] = phones
        self._refresh_inventory()
        return len(oov)

    def phonemes(self, word: str) -> List[str]:
        word = self.canonical(word)
        if word in self._phonemes:
            return list(self._phonemes[word])
        return list(self._g2p_fallback(word))

    def vocabulary(self) -> Sequence[str]:
        return list(self._phonemes.keys())

    def coverage_report(self, words: Sequence[str]) -> Dict[str, object]:
        """Diagnostic: for a given word list (e.g. your study's actual
        vocabulary), report how many are in the static dictionary vs. need
        G2P vs. have no pronunciation at all. Run this before trusting a
        real analysis -- a high G2P-fallback rate on ordinary German words
        would suggest something is wrong with the dictionary path/format."""
        vocab = {self.canonical(w) for w in words if _is_lexical(w)}   # punctuation has no pronunciation by design
        in_g2p_cache = {w for w in vocab if w in self._g2p_cache and self._g2p_cache[w]}
        in_dict = {w for w in vocab if w in self._phonemes and w not in in_g2p_cache}
        missing = vocab - in_dict - in_g2p_cache
        return {
            "total_words": len(vocab), "in_static_dictionary": len(in_dict),
            "via_g2p_fallback": len(in_g2p_cache), "unresolved": len(missing),
            "unresolved_examples": sorted(missing)[:20],
        }
