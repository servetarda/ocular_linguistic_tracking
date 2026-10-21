# How this pipeline differs from Heilbron et al. (2022)

Heilbron, Armeni, Schoffelen, Hagoort & de Lange (2022), *A hierarchy of
linguistic predictions during natural language comprehension*, PNAS 119(32),
e2201968119 (and SI Appendix).

The differences fall into three groups:

1. differences that come from studying German;
2. deliberate changes made for accuracy (in every case the paper-style value
   is still in the output);
3. what is kept the same.

Full technical details are in [methods_report.html](methods_report.html).

---

## 1. Differences because the study is in German

The paper studied English. Every resource was replaced by a German one.

| Purpose | This pipeline |
|---|---|
| Next-word prediction | GerPT2 (`benjamin/gerpt2`), a German GPT-2. It was chosen out of five German models because it best predicted German reading times (PoTeC eye-tracking data). GerPT2-large is reported as comparison columns. |
| Part of speech | spaCy `de_core_news_lg` (Universal Dependencies tags) |
| Word meaning | fastText `cc.de.300` German word vectors |
| Pronunciations | MFA `german_mfa` dictionary; MFA German G2P for words it lacks |
| Word frequency | SUBTLEX-DE (German film subtitles) |

**Spelling:** the story uses Swiss spelling ("regelmässig"). The models
read the standard spelling ("regelmäßig"), which they know better. The
original spelling is kept in the output column `word`, and the spelling
the models read is in `word_model`.

---

## 2. Deliberate changes (for accuracy)

### 2.1 Word probability: end-of-word correction

**Paper:** a word's probability is the product of the probabilities of the
model's text pieces (tokens). Each piece carries the space *before* a word,
so the model marks where a word starts but not where it ends. The product
therefore measures "the next word *starts with* these letters".

**Here:** after the word's pieces, the model is also asked whether a new
word starts next (space, punctuation or line break) or more letters follow.
The word's probability is multiplied by the probability that it ends there.
A small bookkeeping step prevents the previous word's ending from being
counted twice. The method is from Pimentel & Meister (2024).

Example, after "Marlene sollte in den …" (GerPT2):

| | Paper's method | Chance the word ends here | This pipeline |
|---|---|---|---|
| Vor | 0.110 % | 1.5 % | 0.002 % |
| Vorruhestand | 0.048 % | 99.9 % | 0.048 % |

The paper's method makes the fragment "Vor" more likely than the real word.
Complete words are almost unaffected. The correction matters for words that
are also the start of longer words (Haus → Hausaufgaben), which are common
in German. It also removes fragments from the candidate list that feeds the
part-of-speech, semantic, phonemic and syllabic levels.

**Why this is right for listening:** listeners do hear words bit by bit, but
the model's pieces are not what they hear. The pieces are spelling chunks
(" Vor" + "ruh" + "estand"), not syllables or sounds. Hearing bit by bit is
modelled by the phonemic and syllabic levels. Those levels include the
possibility that the word ends at the current sound. Example: suppose the
model strongly expects "Hausaufgaben" and the speaker says only "Haus". The
paper's method says "Haus" is not surprising. This pipeline says it is,
which matches the listener.

- Paper-style value: column `lexical_surprisal_raw`.
- Switch off: `--no-eow-correction`.

### 2.2 Spoken-word column (optional) and page layout

The main column, `lexical_surprisal`, is the word as written, as in the
paper. An extra column, `lexical_surprisal_spoken`, adds together the
spellings a listener cannot tell apart:

- capital vs lowercase initial;
- a number written in digits vs in words ("100" = "hundert").

Page layout is also inaudible. The model's choice between a space and a line
break is removed from the word's probability. Without this, "Marlene" cost
26 bits only because the model expected a new paragraph.

- Spoken form: column `lexical_surprisal_spoken` (not used by default).

### 2.3 Sound predictions use the whole dictionary

**Paper:** the context-based (hierarchical) phoneme model took the language
model's top candidates (the nucleus). Every other word got the same small
pseudo-count.

**Here:** the language-model probability of every one of the 143,248
dictionary words is computed at every position. Words are summed over
casings and spellings. A search shortcut skips words only when it can prove
that no cohort probability changes by more than 1%.

### 2.4 Frequency-based model uses SUBTLEX-DE counts

The single-level (frequency-only) phoneme model and the unigram surprisal
use SUBTLEX-DE counts + 1 for every dictionary word.

### 2.5 Syllable level (extension)

Syllabic surprisal and entropy are new. They are not in the paper. They use
the same cohort model as the phonemic level, with German syllable rules:

- legal word onsets;
- r-glide handling;
- compound and prefix boundaries treated as syllable edges.

### 2.6 How the language model was chosen

The paper did not select its model this way. Here, the model was chosen by
how well its surprisal predicts human reading times (held-out fit on PoTeC).

For the story itself:

- a preliminary check on independent German listening data (MEG, 8
  participants; `scripts/listening_benchmark/`) is consistent with keeping
  GerPT2, but it does not show that GerPT2 and GerPT2-large are equally good;
- the fixes in the pipeline alone improved the reading-time fit 2.7–3.6-fold
  for the same model.

---

## 3. What is the same as the paper

- **Nucleus:** the smallest set of complete-word candidates reaching 90%
  probability (including the word that crosses 90%), and at least 40 words
  (SI A.3). The same nucleus is used for lexical entropy, part of speech and
  semantics. It is the prediction made before the word is heard: the
  presented word is in it only if it is one of the top candidates.
- **Part of speech:** every candidate is tagged in "the context plus the
  candidate" (the paper: tagging "every potential sentence"). The presented
  word is tagged the same way, with the preceding words only, so nothing
  after the word changes its value. Surprisal is −log₂ of the probability of
  its tag. If the presented word is not in the nucleus, its own probability
  is added to its tag, so the tag is never less likely than the word.
  Ambiguous words are decided by the preceding words alone: "ein Bad zu …"
  counts "zu" as a preposition, because the verb that makes it "zu nehmen"
  has not been heard yet.
- **Semantic prediction error:** the cosine distance between the
  probability-weighted mean vector of the candidates and the presented word.
- **Semantic distance:** the control from Broderick et al. (2018).
- **Phoneme cohort model:** includes an explicit "the word ends here"
  outcome (SI Eq. 8). It comes in a hierarchical version (language-model
  prior) and a single-level version (frequency prior).
- **Long texts:** the model's context window is rebuilt at sentence
  starts, with at least 700 tokens of overlap.

---

## 4. Shared limitation

Part-of-speech surprisal depends on a tagger, which can "repair" an odd word
to fit its slot. For example, *einen laufen
Mantel* is tagged adjective. The story's odd words (e.g. *zu einer dieser
Unkraut*) are nouns in noun positions with the wrong grammatical form. Word
class alone cannot detect them. This limitation comes from the paper's
method, not from the changes above. The lexical and semantic levels do
detect these words.

---

## For a methods section

1. Report the changes in section 2 as deviations from Heilbron et al. (2022).
2. The paper-style values are always in the output (`lexical_surprisal_raw`
   and `--no-eow-correction`). Showing that the main results hold with them
   costs nothing.
