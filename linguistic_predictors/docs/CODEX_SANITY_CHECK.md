# Codex sanity check: findings and suggestions for Claude

Saved: 18 October 2026. Review performed: 13 October 2026.
Reviewed repository reference: latest commit reported as `9b79534` at review time.

## Scope and authorization

The user requested a read-only sanity check of `docs/SUMMARY.md` and relevant project files. The user subsequently authorized creation of this separate note only. No existing project files were edited, no project code or tests were run, and no results were regenerated during the review. This note does not authorize implementing changes; obtain the user's approval before doing so.

These findings describe the files inspected on 13 October. They have not been rechecked against any subsequent changes. This was a static review, not an independent reproduction, a PDF report audit, or a verification against Heilbron’s original paper and supplementary methods. References below are repository-relative; line numbers are those at review time.

## Overall assessment

There is substantial implementation work behind the summary, and the saved results support several numerical claims. However, “finished, validated, and ready for the paper” is stronger than this review can support. Preserve the current results as provisional while resolving the issues below.

## What checked out

- CSV row counts matched 1,810 units, 6,719 phonemes, and 2,671 syllables. The run configuration reported 1,553 words.
- The reported odd-versus-normal word statistics matched `results/statistics.csv`.
- GerPT2 had the best reading-time scores in `docs/benchmark_results.csv`.
- The inspected code implemented end-of-word scoring, end-of-word competitors in phoneme cohorts, and preservation of uncorrected lexical scores.

Matching saved tables confirms reporting consistency, not independent correctness of their generation.

## 1. Lexical entropy differs from the documented complete-word measure

Evidence: `linguistic_predictors/lexical.py`, `_stats_from_logprobs` (around line 409); `linguistic_predictors/candidates.py` (around lines 191 and 218); `linguistic_predictors/pipeline.py`.

`lexical_entropy` is derived from a masked distribution over word-initial language-model tokens. Those tokens can be fragments, rather than complete words. The complete-word candidate distribution used for other levels is a different distribution. Consequently, the summary's claim that all measures come from one complete-word distribution is inaccurate.

There is also a nucleus-selection issue: `keep = cdf <= top_p` omits the token that crosses the probability threshold, unless the minimum-count rule includes it. This need not retain the smallest set reaching 90% probability as documented.

Suggested follow-up:
- Decide and document whether the intended measure is word-start-token entropy or complete-word entropy.
- Trace the intended definition against the original paper and supplementary methods before changing it.
- If complete-word entropy is intended, review how the scored vocabulary, normalization, truncation, and missing probability mass should enter that calculation.
- Verify the threshold-crossing behavior with a small explicit probability distribution.
- Estimate the effect on saved predictors before deciding which outputs require regeneration.

## 2. POS prediction uses later-word information through the actual-word override

Evidence: `linguistic_predictors/syntactic.py`, `pos_surprisal_entropy` (around lines 38–52); corresponding test in `tests/test_offline.py`.

Candidates are initially tagged using preceding context only. The actual word's candidate tag is then overwritten with its full-sentence tag: `tags[actual] = actual_pos`. Later words can therefore change the probability distribution used for POS surprisal and entropy.

Using full-sentence information to label the observed outcome is distinguishable from using it to modify the predictive distribution. The existing test explicitly endorses the overwrite; passing that test does not establish methodological correctness.

Suggested follow-up:
- Compare this exact behavior with the original methodology.
- Specify how ambiguity between a word's possible POS categories is handled.
- Check whether identical prefixes receive different predictive distributions when only the continuation changes.
- Justify or revise the override with user approval. Avoid treating removal of an extreme “zu” value as sufficient validation.

## 3. Alignment checks do not establish pronunciation or boundary agreement

Evidence: `linguistic_predictors/textgrid_align.py`, `merge_onsets_into_phoneme_table` (around lines 174–205) and `merge_onsets_into_syllable_table` (around lines 246–249).

Phoneme alignment checks counts but does not compare phone identities before attaching times. Different pronunciations with equal phone counts could pass. Syllable alignment re-syllabifies the aligned phones and checks syllable counts without proving that boundaries match those used for predictor calculation.

This is a prospective risk for audio alignment, not demonstrated corruption of the existing untimed outputs.

Suggested follow-up:
- Define explicit normalization between dictionary and aligner phone symbols.
- Check actual pronunciation sequences, allowing only justified equivalences.
- Confirm that the phonemes grouped into each timed syllable match the syllables used to compute its predictor.
- Test equal-count pronunciation mismatches, pronunciation variants, and compound boundaries before trusting aligned outputs.

## 4. Listening results are preliminary evidence, not demonstrated equivalence

Evidence: `docs/listening_benchmark_results.csv`; `scripts/listening_benchmark/compare.py` and `summary.py`.

Eight participants provide a preliminary comparison. A nonsignificant difference does not establish that GerPT2 is “as good as” GerPT2-large. The reported six-of-eight wins need to be identified as the selected-region analysis rather than an unqualified result across analyses.

Suggested follow-up:
- Use wording such as “preliminary listening evidence consistent with retaining GerPT2.”
- Report effect sizes, uncertainty, the region-selection procedure, and the distinction between whole-cortex and selected-region results.
- Avoid “confirmed” or equivalence claims unless an appropriate analysis supports them.
- Verify the reproduction against the source study before claiming that its analysis was followed exactly.

## 5. Statistical diagnostics do not establish comprehensive validation

Evidence: `scripts/story_statistics.py` (around lines 128–146); `scripts/benchmark_models.py`, `cv_delta_ll` (around lines 144–157).

The phoneme tests use thousands of observations grouped within words and a single story without accounting for that dependence. The reading benchmark randomly splits words instead of holding out complete texts. Its standard-error calculation treats overlapping repeated-cross-validation scores as independent. These choices limit inferential claims, even when the descriptive patterns are useful.

The statement that a maximum pairwise correlation of 0.68 establishes suitability for joint regression is also too strong. Pairwise correlations alone do not establish stable attribution among multiple predictors, particularly after constructing time-lagged neural-response models.

Suggested follow-up:
- Separate descriptive sanity checks from confirmatory inference.
- Consider grouping or resampling at defensible units such as words or passages, recognizing the limits of a single story.
- Assess reading-model rankings with text-held-out evaluation and uncertainty appropriate to dependent folds.
- Inspect the actual lagged regression design and held-out model comparisons before making claims about unique predictor contributions.

## 6. Reporting and test-coverage discrepancies

### Semantic measures

The semantic prediction error is cosine distance, not surprisal in bits. There is no semantic entropy output. The summary's “each level gives surprisal and entropy” statement should be qualified.

### Run duration

`results/run_config.json` recorded `seconds: 31590.5`, approximately 8.8 hours, while the summary said about two hours. These may describe different runs. Reconcile their provenance rather than assuming either duration is the definitive one.

### Meaning of “all tests pass”

`tests/test_real_gpt2.py` targets `dbmdz/german-gpt2`, rather than the selected GerPT2. It catches model-loading exceptions and exits successfully when skipping tests. `tests/run_all.py` treats a zero exit code as successful, although its displayed final line can disclose the skip.

Record which tests actually ran, which were skipped, the selected model, and the relevant versions before claiming full real-model coverage. The reviewer did not run these tests.

## Suggested priority, subject to user approval

1. Resolve the lexical-entropy definition and nucleus-selection behavior.
2. Establish the intended POS methodology and evaluate the actual-word override.
3. Verify pronunciation and syllable matching before audio-aligned analysis.
4. Reconcile documentation with saved runs and qualify validation claims.
5. Perform targeted verification and any justified regeneration only after approval.

No fixes are applied by this note.
