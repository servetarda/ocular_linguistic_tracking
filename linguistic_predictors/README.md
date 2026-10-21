# linguistic_predictors

Word-by-word predictability of German text at five levels — lexical,
part of speech, semantic, phonemic and syllabic — as surprisal and entropy,
for regression against EEG/MEG or eye-tracking data (mTRF). The method
follows Heilbron, Armeni, Schoffelen, Hagoort & de Lange (2022), *A hierarchy
of linguistic predictions during natural language comprehension*, PNAS
119(32), e2201968119; the syllabic level is an extension.

The full method, the errors found and fixed in the original implementation,
and the validation are in **[docs/methods_report.html](docs/methods_report.html)**.

## Quick start

```bash
# check that everything runs, offline (tiny stand-in models)
python run_story.py --text "Die kleine Katze rannte unter den Baum." --backend toy --out results_toy/

# the real run
python run_story.py --text-file stimuli/story_part1.txt \
    --mfa-dict ~/Documents/MFA/pretrained_models/dictionary/german_mfa.dict \
    --device mps --out results/ --progress
```

Run from the conda environment with the requirements installed (`nlp_env`
on this machine). Expect ≈9 s per word on an Apple M4 with the full
143k-word dictionary; `--vocab-size 20000` is ≈5× faster.

## Output

| file | contents |
|---|---|
| `words.csv` | one row per unit (word or punctuation mark), every level |
| `phonemes.csv` | one row per phoneme — the mTRF input (+ `onset`/`offset`/`duration` with `--textgrid`) |
| `syllables.csv` | one row per syllable, same |
| `qa_sample.csv` | spot-check sheet for a native speaker (phonemes, syllables, POS in context) |
| `run_config.json` | settings, models, pronunciation coverage, cohort health |

Main columns of `words.csv` (bits):

| column | meaning |
|---|---|
| `lexical_surprisal` | −log₂ P(word \| preceding text), the word as written (Heilbron et al.), with the end-of-word correction |
| `lexical_surprisal_raw` | the same without the end-of-word correction (plain token product) |
| `lexical_surprisal_spoken` | summed over spellings a listener cannot tell apart (casing; digits for numerals) |
| `lexical_entropy`, `entropy_full`, `entropy_words` | entropy of the complete-word nucleus (the one POS and semantics use); full next-token entropy; entropy over the scored vocabulary |
| `unigram_surprisal`, `in_subtlex` | SUBTLEX-DE frequency (Laplace-smoothed); whether SUBTLEX has the word |
| `pos`, `pos_surprisal`, `pos_entropy` | predicted part of speech |
| `semantic_prediction_error`, `semantic_distance` | predicted-vs-actual meaning (content words); Broderick et al. control |
| `phon_*`, `syll_*` | per-phoneme / per-syllable surprisal and entropy, `hier` (language-model prior) and `freq` (frequency prior) |
| `lexical_surprisal__gerpt2-large` … | the same word-level values from a comparison model |
| `word`, `word_model` | the unit as written / as the models read it (Swiss "ss" → "ß") |

Onset times come from forced alignment of the audio (`mfa align`); pass the
TextGrid with `--textgrid` to add them to `phonemes.csv` / `syllables.csv`.

## Method in brief

- **Lexical** — GerPT2 (`benjamin/gerpt2`) reads the text as written. Word
  probability includes the end-of-word correction (Pimentel & Meister 2024),
  and the line-break-vs-space choice is conditioned away. Long texts use
  ≈800-token windows that start at sentence boundaries.
- **Candidates** — at every position, the probability of every dictionary
  word (both casings, standard spelling, numerals as digits), scored exactly
  with a shared-context token trie. Skipped subtrees are proven to change no
  cohort probability by more than 1%. This one distribution feeds the four
  levels below.
- **Lexical entropy** — entropy of the nucleus of this distribution (the most
  probable words up to 90% probability, at least 40).
- **Syntactic** — each nucleus candidate is tagged by spaCy with left context
  only, and so is the presented word (no look-ahead); POS surprisal is −log₂
  of the probability of the presented word's tag.
- **Semantic** — probability-weighted mean fastText vector of the nucleus vs
  the presented word (cosine distance), for content words.
- **Phonemic / syllabic** — cohort model over the MFA dictionary, with an
  explicit end-of-word outcome. The prior is the language model's
  probability (`hier`) or SUBTLEX-DE frequency (`freq`). Syllables use
  legal-onset maximisation and treat compound and prefix boundaries as
  syllable edges.

## Validation

- `python tests/run_all.py`: offline suites plus a real-model suite. The
  real-model suite compares every fast path against brute-force Hugging Face
  computations.
- `python scripts/validate_model.py --mfa-dict …`: minimal sentence pairs
  (semantic incongruity, cloze, word-class violation). 13/13 checks come out
  in the expected direction; see `docs/validation_results.csv`.
- `python scripts/benchmark_models.py`: language-model choice. It compares
  perplexity on the story with the held-out fit to German eye-tracking
  reading times (PoTeC). GerPT2 fits best; see `docs/benchmark_results.csv`.
- `python scripts/audit_results.py results/`: flags implausible values in an
  output folder.
- `python scripts/story_statistics.py results/`: statistical tests of the
  story's predictors (anomalous words vs other nouns, frequency, content vs
  function words, context vs frequency in phoneme prediction) and the
  correlations between predictors.

## Layout

```
run_story.py                 command-line entry point
linguistic_predictors/
  pipeline.py                compute_predictors: all levels for one text
  components_de.py           loads the real German models and resources
  toy.py                     offline stand-ins (--backend toy, tests)
  text.py                    text units and their positions in the text
  lexical.py                 GPT-2 word probability, entropy, context windows
  vocab_scorer.py            fast exact scoring of the whole vocabulary
  candidates.py              the candidate distribution per position
  cohort.py                  sequential cohort model (phonemes, syllables)
  phonemic.py, syllabic.py   pronunciations, frequency prior, syllabification
  compounds_de.py            compound/prefix boundaries
  lexicon_mfa.py             MFA dictionary + G2P
  frequency_de.py            SUBTLEX-DE
  normalize_de.py            spelling normalisation, German numerals
  syntactic.py, semantic.py  POS and semantic levels
  mtrf_export.py, textgrid_align.py, qa_export.py   outputs
scripts/                     validate_model, benchmark_models, audit_results
tests/                       run_all.py and the suites
stimuli/story_part1.txt      the study text (from the PDF; "ɑ" → "a")
docs/                        methods report, validation and benchmark results
data/                        downloaded resources (not in git)
```

## Resources

| resource | where it is expected | source |
|---|---|---|
| MFA `german_mfa` dictionary + G2P | `--mfa-dict`; MFA's own conda env is found automatically | `mfa model download dictionary german_mfa` |
| SUBTLEX-DE | `data/subtlex_de/` | https://osf.io/py9ba/ (CC BY-NC-SA 4.0) |
| fastText `cc.de.300.bin` | `~/NLP/data/fasttext/` or `--fasttext-path` | https://fasttext.cc/docs/en/crawl-vectors.html |
| GPT-2 checkpoints | Hugging Face cache (downloaded on first use) | `benjamin/gerpt2`, `benjamin/gerpt2-large` |
| PoTeC reading measures (benchmark only) | `data/potec/rm_merged/` | https://osf.io/download/3ywhz/ |

## Options worth knowing

`--vocab-size N` (speed), `--prune-eps` (error bound, default 0.01),
`--gpt2-size {small,large,xl,dbmdz}` / `--gpt2-model`,
`--extra-lexical-models` (none to skip), `--no-eow-correction`
(Heilbron-style token products), `--no-spelling-normalization`,
`--no-compounds`, `--score-punctuation`, `--textgrid`. `python run_story.py -h`
lists them all.

## Limitations

- POS surprisal cannot see an open-class category violation that the tagger
  re-tags to fit its slot (e.g. *einen laufen Mantel* is tagged ADJ).
- The model choice rests on reading data (PoTeC), with preliminary support
  from 8 listeners' MEG data. Check it on the study's own data if possible
  (`--extra-lexical-models benjamin/gerpt2-large` adds comparison columns).
- With `--textgrid`, every aligned sound must be the same phone as the
  dictionary pronunciation the predictors used (not just the same number of
  sounds); give MFA the same pronunciations, or declare justified
  equivalences.
- Compound splitting and the spelling rule are frequency-based heuristics.
  Words missing from the dictionary rely on G2P; check `qa_sample.csv`.

## Citation

Cite Heilbron et al. (2022) for the method, and the resources used:
MFA (McAuliffe et al. 2017), SUBTLEX-DE (Brysbaert et al. 2011), fastText
(Grave et al. 2018), GerPT2 (Minixhofer), spaCy.
