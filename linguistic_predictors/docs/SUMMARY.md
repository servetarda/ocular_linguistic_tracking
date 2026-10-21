# Project summary: German word-predictability predictors for an EEG/eye-tracking listening study

*Status as of 18 October 2026, after the review in `docs/CODEX_SANITY_CHECK.md` (phase 8). Project folder: `~/linguistic_predictors` (a git repository).*

## 1. Goal

The study measures brain activity (EEG) and the eyes (pupil size, eye movements) while people **listen** to a German story. To analyse that data, you need numbers saying, for every word and sound, **how predictable it was** given everything heard before.

The program computes these numbers. It follows the method of **Heilbron et al. (2022, PNAS), "A hierarchy of linguistic predictions during natural language comprehension"**, carried over from English to German.

## 2. Current state in one paragraph

The program follows Heilbron's method, with only small technical refinements, and passes its automated tests. It has been run on the full story; the results are saved and behave as expected in descriptive checks (these are sanity checks on one story, not proof about listeners). An outside review (18 October) found two places where the code did not match the method; both are fixed and the story was re-run (phase 8). A 14-page PDF report is in `docs/methods_report_heilbron_german.pdf`, **but it still has the wording from before phase 8 and must be updated before use**. The **next step belongs to the user**: record or obtain the story audio, align it with the text (forced alignment), then use the predictors in brain-response (TRF) models.

## 3. What the program computes

For every word, a language model (GerPT2) estimates the probability of every one of about 143,000 German words appearing at that point. All five kinds of predictability are read from that one distribution of complete words:

| Level | What is predicted | Example question |
|---|---|---|
| Lexical | the next word | How expected is "Vorruhestand" after "Marlene sollte in den …"? |
| Syntactic (part of speech) | the word type (noun, verb, …) | Was a noun expected here? |
| Semantic | the word's meaning | How far is the meaning from what was expected? |
| Phonemic | the next sound within the word | After /f oː ʁ/, how expected is /uː/? |
| Syllabic (extra, not in Heilbron) | the next syllable within the word | After "Vor-", how expected is "-ru-"? |

The levels give:
- **Surprisal** (lexical, word type, sound, syllable): how unexpected the actual input was, in bits. High means surprising.
- **Entropy** (the same levels): how uncertain the prediction was beforehand.
- **Semantic prediction error** (meaning): a distance between the expected and the actual meaning, from 0 to about 1. It is not in bits, and there is no meaning entropy (as in Heilbron).

The sound and syllable levels come in two versions:
- **"hier":** uses context.
- **"freq":** uses word frequency only. This is the context-free comparison, as in Heilbron.

## 4. Work done, in order

### Phase 1: Audit and rebuild of the original program

I tested every part of the original code against the real models, and found and fixed errors at every level:

**Lexical (word) level**
- The language model read the text re-joined with spaces ("überraschend ." instead of "überraschend."), which made punctuation look very surprising.
- The text started with a lone space, so the first word got 15.5 bits instead of 3.7.
- Word probability was computed as the product of the model's text pieces. That measures "the next word *starts with* these letters", so the fragment "Vor" looked more likely than "Vorruhestand". It's now corrected (end-of-word correction, Pimentel & Meister 2024).
- The model's choice between a space and a line break was counted as word surprisal. "Marlene" got 26 bits only because the model expected a new paragraph. It's now removed.
- Umlauts were garbled in the candidate list.

**Phonemic and syllabic levels**
- Words ending at the current sound were dropped from the competitors; "Tisch" was missing while "Tischler" was being heard.
- Only 10,000 words were used, and it was very slow (74 minutes for 36 words). It now uses all 143,000 dictionary words at about 4–9 seconds per word.
- The pronunciation dictionary writes ß as "ss", so 751 words (weiß, groß, Straße…) were scored in an unusual spelling.
- Every word came out with exactly one syllable, and the syllable rules were wrong.
- The pronunciation generator for unknown words never ran.

**Semantic level**
- Word vectors were looked up in lowercase. German vectors are case-sensitive, so many words got no vector.
- Punctuation was mixed into the semantic-distance context.

**Syntactic level**
- Candidates were lowercased.
- The presented word was counted twice.
- Candidate words were tagged with the rest of the sentence visible, i.e. with information from the future.

**Frequency:** I replaced the mixed-source frequency list (`wordfreq`) with SUBTLEX-DE (German film subtitles), which is closer to spoken German.

### Phase 2: Choosing the language model

I compared five German GPT-2 models on how well their surprisal predicts German **reading times** (eye-tracking corpus PoTeC). **GerPT2 (`benjamin/gerpt2`) fitted best.** On the same model, the fixes alone made the original pipeline fit reading times 2.7–3.6× better.

### Phase 3: Validation, first report, clean-up

- **Minimal sentence pairs** that differ in one word (e.g. "Milch und Zucker" vs "Milch und Socken"): 13 of 13 checks come out in the expected direction.
- **Reports and clean-up:** I wrote a technical HTML report (`docs/methods_report.html`) and cleaned up the code into one pipeline.

### Phase 4: Reviewing ChatGPT's changes, and returning to Heilbron's method

ChatGPT had added two measures that are not part of Heilbron's method: a "grammar-form" measure (case, number, gender) and a step that overrides the tagger's word types. **Because the user wants Heilbron's methodology unchanged, I removed both.** They are kept on a separate git branch, `chatgpt-extensions`, in case they're wanted later.

I also fixed three things in this phase:
- **Main word-surprisal column:** now the word as written, as in the paper. The listener-oriented version, which adds up spellings a listener can't tell apart (capitals, digits vs words), is kept as an extra column.
- **POS bug:** infinitival "zu" ("zu nehmen") got the maximum word-type surprisal (39.9 bits), because the tagger labelled it differently with and without the following verb. A word type can never be less likely than a word of that type; this is now guaranteed. *(The fix chosen here let later words influence the value; it was replaced in phase 8.)*
- **Word vectors:** after the big fastText file was deleted, the program would silently have used different word vectors (spaCy's). It now uses the saved fastText cache, and stops with a clear error if a word is missing.

### Phase 5: The full story

**Text corrections** (no words changed):
- joined 4 line breaks in the middle of sentences, left over from the PDF;
- added quotation marks to 2 pieces of speech that lacked them, and added a missing space after a closing quote;
- "Bio gemüse" → "Biogemüse", "quadratmeter" → "Quadratmeter", "Brennesseln" → "Brennnesseln".

**The 15 deliberately odd words**, which are never changed: Unkraut, Anlage, Paradies, Zeitung, Anstrichs, Vollbart, Schuppen, Schubkarre, Wiese, Garten (twice), Handgelenk, Fortkommen, Besuch, Humor.
- Each also appears elsewhere in the story in a normal sentence.
- They are listed with their context in `stimuli/violations.csv`.

**Bugs found during the long run:**
- **Memory leak:** Apple's graphics chip kept memory it no longer needed (10.7 GB after 100 words). The run slowed from 2 to 25 seconds per word and was killed by macOS after 10 hours. Fixed; memory now stays at about 1.6 GB.
- **Punctuation:** four punctuation marks showed "infinite" surprisal instead of an empty value. Fixed.
- **Syllable value:** one syllable was capped at the 39.9-bit maximum instead of its real value of about 44 bits. Fixed.

**Final run:** 1,810 units (1,553 words), 6,719 sounds and 2,671 syllables. That run took 8.8 hours (it also computed the GerPT2-large comparison); the phase-8 re-run without the comparison took 1.9 hours. The automatic audit found no problems.

**Statistical tests:** see section 6.

### Phase 6: PDF report and the listening check

- **PDF report:** I wrote `docs/methods_report_heilbron_german.pdf`, with a plain explanation, a draft Methods section for the paper, the differences from Heilbron, the validation, and the limitations.
- **The user's concern:** the model was chosen on *reading* data, but the study is about *listening*.
- **The listening check:** I used the public listening data of Gehmacher et al. (2026, bioRxiv): German listeners' MEG brain recordings while they heard German stories (OSF project 4m2st). With their own analysis model, I tested which language model's surprisal best predicts brain activity, using 8 participants.
  - GerPT2 and the 6× larger GerPT2-large did not differ significantly. With 8 people this does **not** show they are equally good.
  - In the selected-region analysis, both beat dbmdz (the model those authors used) in 6 of 8 participants; this trend is not significant.
  - In short: preliminary listening evidence consistent with keeping GerPT2.
- **Code and report:** the code is in `scripts/listening_benchmark/`, and the PDF report was updated with this result.

### Phase 7: Guidance and housekeeping

- **Guidance given:** how to do forced alignment with MFA, and how to turn the predictors into brain-model inputs (section 8).
- **Files deleted:** 8.9 GB of files no longer needed were moved to the Trash: the fastText full model, the MEG download, two extra language models, the reading data and ChatGPT's leftovers. **The user must empty the Trash to get the space back.**
- **Option fixed:** `--extra-lexical-models none` now correctly skips the GerPT2-large comparison.

### Phase 8: Outside review and fixes (18 October)

ChatGPT reviewed the project (`docs/CODEX_SANITY_CHECK.md`). Each point was checked against the code and the paper:

- **Word uncertainty (lexical entropy) — fixed.** It was computed from the model's text pieces ("Vor" counted as a guess), not from complete words as documented. It now uses the same top guesses of complete words as the word-type and meaning levels. The 90% cut-off now also keeps the word that crosses 90%.
- **Word type (POS) — fixed.** Later words could change a word's value: the heard word was tagged with the whole sentence. Now it is tagged like every candidate, with only the words before it (the paper: tagging "every potential sentence"). Example: in "ein Bad zu …", "zu" counts as a preposition, because "nehmen" hasn't been heard yet.
- **Same kind of leak, two more places — fixed.** The heard word was always added to the top guesses, which slightly changed word-type uncertainty and the expected meaning. The guesses are now only what the model predicted.
- **Audio alignment checks — fixed.** Alignment now checks that each aligned sound is the same sound as in the dictionary (not just the same number), and syllable times follow the exact syllables the values were computed on.
- **Statistics — fixed.** Sound tests now count each word once (not 6,700 sounds as independent). Overlap between predictors is now also checked with VIF.
- **Wording — fixed.** Listening result described as preliminary; semantic measure described as a distance; run times corrected; the real-model tests now use GerPT2 (they had been skipped silently, testing a deleted model).

**What changed in the results:** word surprisal, semantic distance, word frequency and all sound and syllable values are identical; meaning prediction error moved by less than 0.005. Word uncertainty changed for every word (correlation with the old values 0.98). Word-type surprisal changed for 80 of 1,553 words (correlation 0.91).
- Some changes fix odd values: "Verdammter" 19.3 → 2.3 bits, "Theos" 22.9 → 6.5 bits.
- Some show the tagger's weakness on unfinished sentences, which now enters the value: "ein Bad" is read as a name (1.8 → 9.4 bits), "Wie krieg" as a noun (1.4 → 13.2 bits). This is a property of Heilbron's procedure, not a new error.

## 5. Differences from Heilbron et al. (2022)

| Type | Difference | Why |
|---|---|---|
| Language | German tools instead of English: GerPT2, spaCy German tagger, fastText German vectors, MFA German pronunciation dictionary, SUBTLEX-DE frequencies | The study is in German. GerPT2 was chosen on reading data and confirmed on listening data. |
| Language | Swiss "ss" read as "ß" by the models (10 words) | The models know the standard spelling better |
| Refinement | End-of-word correction of word probability | Stops word fragments ("Vor") from inheriting the probability of longer words. Paper-style value kept in `lexical_surprisal_raw`. It barely changes the fit to humans. |
| Refinement | Space vs line-break choice removed | Listeners can't hear page layout |
| Refinement | Sound predictions use all 143,000 dictionary words | The paper approximated this with the top guesses plus a small made-up value for the rest |
| Extension | Syllable level | Extra measure; doesn't change the others |

**Same as the paper:**
- the definitions of surprisal and entropy;
- the number of top guesses (90% probability, at least 40 words);
- how word types are predicted;
- the semantic measure;
- the sound model with an "end of word" option;
- the reading window over long texts.

**Shared limitation:** word-type surprisal can't detect odd words that keep their word type. All 15 story violations are nouns in noun positions.

## 6. Key results

All results below are from the phase-8 re-run.

**Odd words vs the same words in their normal sentences** (15 pairs; one-sided tests; p corrected for running 12 tests):

| Level | Odd | Normal | p | Result |
|---|---|---|---|---|
| Word surprisal | 24.4 bits | 5.2 bits | 0.0002 | significant |
| Meaning prediction error (distance) | 0.72 | 0.40 | 0.001 | significant |
| First-sound surprisal | 5.5 bits | 3.6 bits | 0.004 | significant |
| Word-type surprisal | 0.47 bits | 0.30 bits | 0.55 | not significant (expected: nouns in noun slots) |

**General checks** (descriptive, one story), all significant:
- rarer words are more surprising;
- content words are more surprising than function words;
- context halves sound surprisal compared with frequency alone (per word: 1.2 vs 2.6 bits per sound);
- the first sound of a word is more surprising than its later sounds (per word: 3.0 vs 0.4 bits).

**Overlap between predictors:** the highest correlation between two word-level predictors is 0.67 (word surprisal ~ meaning error), and all VIF values are below 2.7. That means no gross overlap. Whether each predictor explains something of its own can only be shown in the TRF models themselves.

## 7. Important files

| File or folder | What it is |
|---|---|
| `run_story.py` | Runs everything on a text |
| `linguistic_predictors/` | The program code |
| `stimuli/story_part1.txt` | The full story (corrected text) |
| `stimuli/violations.csv` | The 15 odd words with their context |
| `results/words.csv` | One row per word or punctuation mark, with all measures |
| `results/phonemes.csv` | One row per sound |
| `results/syllables.csv` | One row per syllable |
| `results/statistics.csv` | The statistical tests |
| `docs/methods_report_heilbron_german.pdf` | Main report; **needs updating for phase 8** (source file no longer available) |
| `docs/methods_report.html` | Older technical report, including the bug history |
| `docs/differences_from_heilbron.md` | Short plain-language list of the differences |
| `scripts/story_statistics.py` | The statistical tests |
| `scripts/listening_benchmark/` | The listening (MEG) model check |
| `tests/run_all.py` | Automated tests (4 suites, including the real GerPT2); all pass, and any skipped suite is reported |

**Resources the program needs:**
- the environments `nlp_env` and `mfa` (in `/opt/anaconda3/envs/`);
- GerPT2 (downloaded automatically by Hugging Face);
- the saved fastText cache (`~/NLP/data/fasttext/cc.de.300.table.*`);
- the MFA dictionary and pronunciation model (`~/Documents/MFA/pretrained_models/`);
- SUBTLEX-DE (`data/subtlex_de/`).

## 8. What comes next (for the user)

1. **Forced alignment.** Download MFA's German acoustic model, put `story.wav` and `story.txt` in one folder, and run `mfa align`. The text must match the recording exactly. Check the result in Praat.
2. **Add the times.** Re-run the program with `--textgrid` to give every sound and syllable an onset time.
3. **Build the brain-model inputs.** Turn each measure into a time series at the EEG sampling rate: zeros, with the measure's value at each word or sound onset.
4. **Fit TRF models.** Include these controls in every model: sound envelope, sound onsets, pitch, word onsets, word frequency, semantic distance, phoneme onsets, and the frequency-only sound measures. Test each context measure by how much prediction accuracy drops when it's removed.
5. **A new story** needs the 7 GB fastText file downloaded once (about 3 minutes) for its new words.

## 9. Rules to keep

- Never "correct" the 15 deliberate odd words.
- Keep Heilbron's methodology; only small technical details may differ.
- Run `tests/run_all.py` after any code change.
- Long runs need the Mac awake, with the lid open and the charger plugged in.
