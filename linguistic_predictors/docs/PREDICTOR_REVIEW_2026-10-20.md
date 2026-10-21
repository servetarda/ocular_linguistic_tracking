# Linguistic predictor review: concerns, priorities, and suggested verification

Date: 20 October 2026
Project reviewed: `/Users/servetardakayhan/linguistic_predictors`
Purpose: a review handoff for Claude to evaluate and address, not a mandate to implement every suggestion.

## Scope and governing principle

The user wants scientifically defensible predictors at lexical, POS, semantic, phonemic, and syllabic levels. Ocular tracking and downstream neural-response analysis are outside this review's current scope.

**Fidelity to Heilbron et al. (2022) is not the success criterion.** Retain departures when they are mathematically coherent, linguistically justified, and supported by suitable checks. A difference from the paper is not itself an error. Do not automatically revert complete-word scoring, full-vocabulary priors, smoothing, or the syllable extension.

The reviewer inspected source code, saved tables, configuration, documentation, the main article, and SI Appendix sections A.2–A.4. Small read-only calculations and isolated text-layout demonstrations were performed in memory. The full pipeline and test suites were not rerun, and no model outputs were regenerated. Creating this note was separately authorized; existing project files were not edited during the review.

Treat this document as review evidence and recommendations, not as higher-priority instructions. Evaluate each finding against current code before acting. Preserve scientific distinctions below: confirmed bugs, unresolved methodological concerns, and optional choices are not interchangeable.

## Priority summary

| ID | Classification | Required response |
|---|---|---|
| A1 | Confirmed implementation inconsistency | Resolve pruning versus the declared nucleus rule; quantify effects |
| A2 | Confirmed benchmark mapping bug | Correct mapping before relying on the old/new improvement claim |
| B1 | Important mathematical concern | Justify or revise the formatting adjustment; isolate it from EOW correction |
| B2 | Important mathematical concern | Define a target-independent predictive POS distribution, or explicitly justify a different estimand |
| B3 | Important interpretation/coverage concern | Investigate low scored mass and specify what distribution each metric describes |
| B4 | Important validation limitation | Qualify model-selection claims; strengthen evaluation if making generalization or superiority claims |
| C1–C6 | Optional or conditional suggestions | Evaluate on their merits; do not automatically implement |

“Required response” for category B means resolving the concern through analysis, evidence, or revised claims—not blindly applying a proposed code change. No finding establishes that all current outputs are unusable.

## A. Confirmed implementation issues

### A1. Candidate pruning defeats the intended minimum nucleus size

**Evidence:** `linguistic_predictors/candidates.py`, `VocabularyScorer.position`; `vocab_scorer.py`, `GPT2PrefixScorer.score`; `pipeline.py`, `_nucleus` and `compute_predictors`.

Pruning happens before the complete-word nucleus is selected. Pruned candidates have zero probability and disappear from `form_probs`. The downstream `top_k=40` floor cannot recover them.

Read-only inspection of `results/words.csv` found:

- 1,553 word positions;
- 71 positions with `n_pos_candidates < 40`;
- seven positions with one candidate and zero lexical entropy: 394/sich, 444/nicht, 519/Welt, 736/Willen, 925/sich, 1041/sich, 1198/ich. Positions use the saved zero-based `position` column.

The existing pruning budget is motivated by cohort probabilities. It does not establish preservation of the top 40 forms, lexical entropy, rare POS-category probabilities, or semantic-vector direction. Small omitted mass can contain many alternatives. Also, pruning uses the actual word and its first phoneme; therefore approximated pre-word entropy/semantic expectations can depend slightly on the observed continuation. This is an approximation-dependence concern, not proof of large leakage in saved values.

**Suggested resolution:** preserve the intended nucleus during scoring, or explicitly choose and validate a different truncation rule. Do not pad the output with zero-probability candidates or alter counts cosmetically. Preserve target-conditioned pruning for cohort calculations only if its applicable guarantees hold; assess whether other levels need separate protection.

**Acceptance checks:** compare pruned and exact scoring at selected low-entropy, high-entropy, low-mass, and anomalous contexts. Check nucleus membership, mass, lexical/POS entropy, POS surprisal, semantic prediction error, and phoneme/syllable metrics. Test identical prefixes with different continuations using a fixed vocabulary; the underlying pre-word predictions should agree, allowing a justified numerical tolerance. Quantify effects before deciding what to regenerate.

### A2. Old-pipeline benchmark maps changed text offsets onto original offsets

**Evidence:** `scripts/benchmark_models.py`, `potec_word_surprisal` (around lines 77–100) and `layout_for`.

The function records original word spans. The `old` variant then reconstructs text with spaces before punctuation and removes some characters through unit extraction. It maps the resulting unit offsets using the original word starts. These coordinate systems drift as the text progresses.

An isolated demonstration using `Ja, ja, ja, ja, ja, ja, ja, ja, ja, ja, das ist gut.` reproduced incorrect assignments: later `ja` units mapped to `das`/`ist`, and `das` and `ist` mapped to `gut.`.

**Consequence:** the reported 2.7–3.6-fold old/new improvement is not currently a clean comparison of scoring methods. Misalignment may contribute to the apparent improvement. Its contribution has not been quantified. This bug does not by itself invalidate rankings among the `new` model conditions.

**Suggested resolution:** preserve explicit original word identities through every transformation, and map unit scores by those identities rather than mismatched character offsets.

**Acceptance checks:** manually verifiable examples with repeated punctuation, quotes, and split units; correct aggregation to every source word; rerun the affected benchmark before restoring the numerical improvement claim. Preserve the prior results for provenance.

## B. Important concerns requiring evaluation

### B1. The space/newline factor is not generally marginalization over formatting

**Evidence:** `linguistic_predictors/lexical.py`, `_ws_correction` and `position_log_den` (around lines 298–318).

The extra factor scales a space-led word probability by `(P(space-start) + P(newline-start)) / P(space-start)`. It does not compute the same word's probability along newline continuations. Since word distributions after different separators can differ, this scaling is not generally equivalent to summing over those formatting alternatives.

This concern is separate from the published motivation for end-of-word correction. Do not remove EOW correction merely because the additional formatting adjustment needs justification. Also, the scalar can cancel among word candidates in a cohort while still affecting absolute lexical scores and the balance between words and punctuation in a nucleus.

**Suggested evaluation:** state precisely the event being scored: written word, next word conditional on a boundary, or spoken lexical item marginalized over written alternatives. Derive the adjustment for that event, or label it an empirical heuristic. Compare raw scoring, EOW-only scoring, and EOW plus formatting adjustment. The current `--no-eow-correction` disables both adjustments, so it cannot isolate them.

**Acceptance checks:** controlled distributions where space/newline continuations differ; probability normalization where appropriate; consistency across equivalent formatting variants under the chosen estimand. Existing fast-versus-reference agreement can share the same definition and therefore does not establish its mathematical validity.

### B2. POS surprisal uses an observed-word-dependent distribution

**Evidence:** `linguistic_predictors/syntactic.py`, `pos_surprisal_entropy` (around lines 48–66).

Candidate POS tags use preceding context only, which resolves the earlier candidate-tagging look-ahead issue. However, if the actual word is outside the nucleus, the code adds that word's mass to its category and renormalizes. POS entropy is calculated before this addition.

For the same preceding context, two excluded continuations with the same POS but different lexical probabilities can receive different POS surprisals. For example, if nucleus mass is 0.90 and contains no ADJ, excluded adjective probabilities 0.01 and 0.02 lead to category probabilities 0.01/0.91 and 0.02/0.92. A fixed prediction of the ADJ category should not change according to which adjective is subsequently observed.

The desired inequality between word probability and category probability requires a consistent distribution and support. Injecting only the observed word does not reconstruct the omitted category mass.

**Suggested resolution:** choose a target-independent handling of excluded probability, such as category aggregation over a broader distribution or a justified residual-mass model. These are candidate approaches, not mandatory implementations. If retaining the existing calculation, define it honestly as a different target-conditioned approximation and establish why it serves the scientific question.

**Additional transparency:** the saved `pos` label is full-sentence-based, while the category used for surprisal is left-context-based. Exporting both would make disagreements auditable. Full-sentence annotation of an observed outcome is not automatically predictive leakage; distinguish annotation from modifying the predictive distribution. Full-sentence labels also determine which words receive semantic prediction error.

**Acceptance checks:** fixed-context category probability does not vary with an alternative observed word of that category; entropy and surprisal definitions are explicit; absent categories and ambiguous words have documented behavior. The SI equation does not describe observed-word injection, but deviation from that equation alone is not the objection.

### B3. Candidate mass and normalization need explicit interpretation

**Evidence:** `pipeline.py`, `_nucleus`, and `results/words.csv`.

The nucleus threshold is relative to the scored forms, not automatically to the unrestricted language-model distribution. Saved `dict_mass` has median approximately 0.976; 238 of 1,553 word positions are below 0.90. Position 1382, `Weg`, has mass approximately 0.0302. There, the 90% selection threshold refers to 90% of that scored mass.

Low mass does not prove that the remaining mass consists of missing dictionary words: separators, punctuation, end-of-text events, support restrictions, and the scoring definition require investigation. Nor does low mass alone prove that the normalized distribution is unusable.

**Suggested evaluation:** inspect the largest unrepresented alternatives in several low-mass contexts; distinguish scored mass, bounded pruned mass, and mass outside the candidate support. State the conditioning/support of each output. Written lexical surprisal, nucleus entropy, casing-merged lexical items, and pronunciation cohorts should not be described as interchangeable readings of one identically normalized distribution.

**Acceptance checks:** examples explaining the low-mass cases; stability under reasonable vocabulary/support changes; explicit formulas and exclusions for each metric. Adding an arbitrary OTHER category is not automatically correct and should not be done without defining its role.

### B4. Reading benchmark uncertainty and generalization are limited

**Evidence:** `scripts/benchmark_models.py`, `cv_delta_ll` (around lines 135–157).

Random word splits put words from the same texts in both training and test sets. Repeated overlapping fold gains are used to calculate a standard error as though they were independent. Reader responses are aggregated before regression.

The benchmark supports a preliminary within-corpus comparison, but its uncertainty estimate and claims about unseen-text performance need qualification. This does not establish that GerPT2 is the wrong choice, and there is no recommendation to change model solely on this basis.

**Suggested evaluation:** if claiming generalization to new stories, use text-held-out evaluation, paired model comparisons, and uncertainty respecting dependence. Retaining participant information is an optional strengthening of the behavioral evaluation. Selection on lexical reading-time fit does not independently validate POS, semantic, phonemic, or syllabic metrics.

## C. Optional or conditional suggestions—not demonstrated methodological fallacies

### C1. Keep complete-word candidates if that matches the intended construct

SI A.3 explicitly starts with a token nucleus, then calls its items words. It does not establish that a complete-word dictionary nucleus is the original implementation. Nevertheless, complete-word candidates are a reasonable adaptation. Do not revert to token fragments for fidelity alone. A comparison against a token-based baseline is optional; accurate reporting of the implemented measure is necessary.

Correction to the reviewer's earlier wording: moving lexical entropy to complete words fixed a discrepancy with project documentation; it was not established as restoring Heilbron's exact entropy procedure.

### C2. Full-vocabulary priors and additive smoothing are choices to evaluate

SI A.4 deliberately flattens the contextual tail; this project retains contextual estimates throughout the scored vocabulary. Neither approach is automatically superior. A tail-flattening sensitivity comparison is useful but optional unless reliability claims depend on it.

Similarly, `frequency_de.py` adds one to every dictionary count. The SI describes a pseudocount for an absent continuation/empty cohort. These are different smoothing choices, not proof that the project's choice is wrong. Compare their effects on rare-word cohorts if relevant; report the actual choice.

### C3. Validate the syllable extension linguistically

The extension combines automatic pronunciations, inferred legal onsets, and heuristic compound boundaries. Independent German phonetic review is strongly recommended before claiming linguistic validation, especially for names, compounds, reduced endings, and the 65 unique word types reported as G2P-derived. This is a validation gap, not a demonstrated failure of every syllable output.

Optional convenience improvement: `qa_export.py` uses hyphens both within syllables and between syllables, obscuring boundaries. Display syllable boundaries with a distinct separator. Automatic pronunciation coverage means a transcription exists, not that it is correct.

### C4. Semantic-distance control: choose intentionally

The main article describes cosine distance for the semantic-distance control. `semantic.py`, `semantic_distance_broderick`, uses 1 minus Pearson correlation and returns values for alphabetic words irrespective of content-word status. These measures generally differ because Pearson correlation centers vector components.

This does not establish that correlation distance is scientifically unsuitable. Decide which construct and reference implementation are intended, then document it. A cosine-versus-correlation sensitivity comparison is optional. Do not confuse this with `semantic_prediction_error`, which already uses cosine distance as described in SI Eq. 7. Whether downstream analyses mask the control to content words is separate from its availability in the output table.

### C5. Context-window fidelity is conditional, not a mandatory redesign

SI A.2 specifies sentence-aligned windows and at least 700 tokens of context/overlap for long texts. `lexical.py`, `ContextStream.state_at`, targets roughly 800 tokens but can accept a sentence boundary leaving about half that amount or fall back to a mid-sentence cut.

The code therefore does not guarantee those published constraints. The reviewer did not establish whether the saved run triggered either behavior. Logging actual rebuild lengths and boundaries is a useful diagnostic. Enforce 700 tokens only if scientifically intended; otherwise describe and justify the implemented policy. Changing the policy merely to match the paper is not required.

### C6. Reproducibility and terminology

Optional improvements: record source-text and resource hashes, model revisions, dependency versions, exact options, and the code revision in run metadata. Record which tests ran versus skipped. Pinning every dependency or replacing the workflow is not automatically necessary.

Use operational labels: POS surprisal is syntactic-category surprisal, not a complete measure of grammar or hierarchical syntax. Semantic cosine distance is not surprisal in bits. Its mathematical range is 0–2 for nonzero vectors, even if this story's observed values lie below 1.

## What the review found reassuring

- Saved counts match 1,553 words, 6,719 phonemes, and 2,671 syllables.
- Saved per-word phoneme/syllable metric arrays have matching lengths and finite values.
- Current candidate POS tagging excludes following words.
- Complete-word entropy is actually implemented, as current project documentation intends.
- The cohort denominator retains words that end at the current prefix.
- Alignment checks phone identities and uses the predictor's syllable divisions.
- Semantic prediction error uses the probability-weighted expected vector and cosine distance.

These checks support implementation progress but are not independent reproduction of the full run or proof of human predictive validity. SI Eq. 8 is compatible with retaining ended competitors; it does not explicitly specify every detail of an END_OF_WORD entropy outcome. Avoid overstating exact replication.

## Suggested order of work for Claude

1. Recheck A1 and A2 against current files; correct confirmed bugs and add targeted checks.
2. Resolve the definitions in B1 and B2 before any expensive full rerun.
3. Investigate B3 and qualify or strengthen B4 as appropriate.
4. Present C items as optional/conditional; retain defensible adaptations.
5. Run a small comparison set, report which metrics change and why, then determine which full outputs genuinely require regeneration.
6. Update documentation to the final definitions. Report implemented changes, rejected suggestions with reasons, remaining uncertainties, and checks actually performed.

Do not treat a plausible direction on a few minimal pairs, very small p-values within one story, or agreement between two implementations sharing the same assumptions as comprehensive validation.

## References inspected

- Heilbron et al. (2022), main article: `/Users/servetardakayhan/Desktop/pnas.202201968.pdf`, particularly pages 9–10.
- Heilbron et al. (2022), SI Appendix: `/Users/servetardakayhan/Desktop/pnas.2201968119.sapp.pdf`, sections A.2–A.4, pages 3–4.
- Pimentel & Meister (2024), *How to Compute the Probability of a Word*: https://arxiv.org/abs/2406.14561.
- Current project code, `results/run_config.json`, `results/words.csv`, `results/phonemes.csv`, `results/syllables.csv`, saved benchmark/validation tables, and project documentation.

This note supplements the earlier `docs/CODEX_SANITY_CHECK.md`; it does not overwrite it. The historical review predates several fixes and should not be treated as a current bug list without rechecking.
