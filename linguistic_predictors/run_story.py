#!/usr/bin/env python3
"""Compute every linguistic predictor for a German text.

    python run_story.py --text-file stimuli/story_part1.txt \\
        --mfa-dict ~/Documents/MFA/pretrained_models/dictionary/german_mfa.dict --out results/

    python run_story.py --text "Die kleine Katze rannte unter den Baum." --backend toy --out results_toy/

Writes into --out/:

  words.csv        one row per unit (word or punctuation mark), all five levels
  phonemes.csv     one row per phoneme (+ onset/offset/duration with --textgrid) -- the mTRF input
  syllables.csv    one row per syllable (same)
  qa_sample.csv    spot-check sheet for a native speaker (phonemes, syllables, POS in context)
  run_config.json  settings, models, pronunciation coverage, cohort health

Onset times come from forced alignment of your audio (`mfa align`); pass the
resulting TextGrid with --textgrid.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from linguistic_predictors import config
from linguistic_predictors.components_de import GERMAN_GPT2_CHECKPOINTS, load_components
from linguistic_predictors.mtrf_export import flatten_phoneme_level, flatten_syllable_level
from linguistic_predictors.pipeline import PipelineOptions, compute_predictors
from linguistic_predictors.qa_export import build_qa_sample
from linguistic_predictors.text import TextLayout


def options_from_args(args) -> PipelineOptions:
    return PipelineOptions(
        top_p=args.top_p, top_k=args.top_k, prune_eps=args.prune_eps,
        eow_correction=not args.no_eow_correction, vocab_size=args.vocab_size,
        case_fix=not args.no_case_fix, spelling_normalization=not args.no_spelling_normalization,
        compounds=not args.no_compounds, score_punctuation=args.score_punctuation,
        pos_max_candidates=args.pos_max_candidates,
        extra_lexical_models=[m for m in args.extra_lexical_models if m.lower() != "none"],
        batch_size=args.batch_size, progress=args.progress)


def components_from_args(args) -> dict:
    if args.backend == "toy":
        from linguistic_predictors.toy import toy_components
        return toy_components()
    if not args.mfa_dict:
        sys.exit("ERROR: --mfa-dict is required unless --backend toy "
                 "(its path: `mfa model inspect dictionary german_mfa`).")
    return load_components(
        args.mfa_dict, gpt2_model=args.gpt2_model or GERMAN_GPT2_CHECKPOINTS[args.gpt2_size],
        device=args.device, g2p_model=(None if args.no_g2p else args.g2p_model),
        spacy_model=args.spacy_model, embeddings=args.embeddings, fasttext_path=args.fasttext_path,
        frequency=args.frequency, subtlex_path=args.subtlex_path)


def write_outputs(df, meta, comps, out_dir: Path, args) -> None:
    pron_dict, vowel_set = comps["pron_dict"], meta["vowel_set"]
    df.to_csv(out_dir / "words.csv", index=False)
    flat_phon = flatten_phoneme_level(df, pron_dict)
    flat_syll = flatten_syllable_level(df, pron_dict, vowel_set)
    if args.textgrid:
        from linguistic_predictors.textgrid_align import (group_phones_by_word,
                                                          merge_onsets_into_phoneme_table,
                                                          merge_onsets_into_syllable_table,
                                                          parse_textgrid)
        print(f"      merging onsets from {args.textgrid}")
        groups = group_phones_by_word(parse_textgrid(args.textgrid), word_tier=args.textgrid_word_tier,
                                      phone_tier=args.textgrid_phone_tier)
        strict = not args.textgrid_lenient
        flat_phon = merge_onsets_into_phoneme_table(flat_phon, groups, strict=strict)
        flat_syll = merge_onsets_into_syllable_table(flat_syll, groups, vowel_set, strict=strict)
    flat_phon.to_csv(out_dir / "phonemes.csv", index=False)
    flat_syll.to_csv(out_dir / "syllables.csv", index=False)
    if args.qa_sample not in ("0", "none"):
        n = None if args.qa_sample == "all" else int(args.qa_sample)
        qa = build_qa_sample(df["word_model"].tolist(), df["pos"].tolist(), pron_dict, vowel_set=vowel_set, n=n)
        qa.to_csv(out_dir / "qa_sample.csv", index=False, encoding="utf-8-sig")

    lm, words = comps["lm"], meta["words"]
    cov = pron_dict.coverage_report(words) if hasattr(pron_dict, "coverage_report") else {}
    subtlex = comps.get("subtlex")
    (out_dir / "run_config.json").write_text(json.dumps({
        "backend": args.backend,
        "gpt2_model": getattr(lm, "model_name", None),
        "extra_lexical_models": [m for m in args.extra_lexical_models if m.lower() != "none"] if meta["is_gpt2"] else [],
        "frequency_source": subtlex.source if subtlex else "wordfreq",
        "embeddings": type(comps["embeddings"]).__name__,
        "spelling_normalised": [list(c) for c in meta["spelling_changes"]],
        "compound_boundaries": not args.no_compounds,
        "n_units": len(words), "n_words": sum(1 for w in words if w[:1].isalpha()),
        "candidate_vocab_size": meta["cand_vocab_size"], "candidate_forms": meta["n_forms"],
        "vocab_size_arg": args.vocab_size, "case_fix": not args.no_case_fix,
        "eow_correction": bool(getattr(lm, "eow_correction", False)), "prune_eps": args.prune_eps,
        "context_seed": [lm.decode_token(t) for t in lm.seed_ids] if meta["is_gpt2"] else None,
        "top_p": args.top_p, "top_k": args.top_k, "pos_max_candidates": args.pos_max_candidates,
        "log_base": config.LOG_BASE, "pronunciation_coverage": cov, "cohort_health": meta["cohort_health"],
        "textgrid": args.textgrid, "seconds": meta["seconds"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      words.csv ({len(df)}), phonemes.csv ({len(flat_phon)}), syllables.csv ({len(flat_syll)})")
    if cov.get("unresolved"):
        print(f"  NOTE: {cov['unresolved']} word(s) have no pronunciation ({cov.get('unresolved_examples')}); "
              "they get no phoneme/syllable rows.")


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    text = args.text if args.text else Path(args.text_file).read_text(encoding="utf-8")
    if not TextLayout.from_text(text).words:
        sys.exit("ERROR: no words found in the input text.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/3] loading components (backend={args.backend}) ...")
    comps = components_from_args(args)
    print("[2/3] computing predictors ...")
    df, meta = compute_predictors(text, comps, options_from_args(args))
    print("[3/3] writing output ...")
    write_outputs(df, meta, comps, out_dir, args)
    print(f"\nDone in {time.time() - t0:.0f}s. Files in {out_dir}/")
    if not args.textgrid:
        print("  phonemes.csv / syllables.csv have no onset times yet: run `mfa align` on the audio, "
              "then re-run with --textgrid.")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="German text, given directly.")
    src.add_argument("--text-file", help="Path to a UTF-8 .txt file.")
    p.add_argument("--out", default="results", help="Output directory (default: results/).")

    g = p.add_argument_group("models")
    g.add_argument("--backend", choices=["torch", "toy"], default="torch",
                   help="'torch': the real models (default). 'toy': tiny offline stand-ins, to check "
                        "that everything runs.")
    g.add_argument("--gpt2-size", choices=sorted(GERMAN_GPT2_CHECKPOINTS), default="small",
                   help="Main language model: small = benjamin/gerpt2 (default, best fit to German "
                        "reading times), large, xl, dbmdz. See scripts/benchmark_models.py.")
    g.add_argument("--gpt2-model", default=None, help="Any GPT-2 checkpoint id (overrides --gpt2-size).")
    g.add_argument("--extra-lexical-models", nargs="*", default=["benjamin/gerpt2-large"],
                   help="Comparison models for word-level columns (lexical_surprisal__<name>); "
                        "'none' (or no names) to skip.")
    g.add_argument("--device", default="auto", help="auto (Apple GPU / NVIDIA GPU / CPU), mps, cuda or cpu.")
    g.add_argument("--mfa-dict", default=None, help="The german_mfa pronunciation dictionary file.")
    g.add_argument("--g2p-model", default="german_mfa", help="MFA G2P model for words missing from the dictionary.")
    g.add_argument("--no-g2p", action="store_true", help="No G2P: unlisted words get no pronunciation.")
    g.add_argument("--spacy-model", default="de_core_news_lg")
    g.add_argument("--embeddings", choices=["auto", "fasttext", "spacy"], default="auto",
                   help="auto: fastText cc.de.300.bin if found, else spaCy vectors.")
    g.add_argument("--fasttext-path", default=None, help="Path to cc.de.300.bin.")
    g.add_argument("--frequency", choices=["subtlex", "wordfreq"], default="subtlex",
                   help="Frequency source for unigram surprisal and the single-level cohort.")
    g.add_argument("--subtlex-path", default=None, help="SUBTLEX-DE file (default: data/subtlex_de/).")

    m = p.add_argument_group("method")
    m.add_argument("--vocab-size", type=int, default=0,
                   help="Candidate vocabulary: the N most frequent dictionary words (0 = all, the default; "
                        "20000 is ~5x faster).")
    m.add_argument("--prune-eps", type=float, default=0.01,
                   help="Error bound of the candidate scorer: every cohort probability within this relative "
                        "error (0.01 = 1%%). 0 = exact, several times slower.")
    m.add_argument("--top-p", type=float, default=config.TOP_P)
    m.add_argument("--top-k", type=int, default=config.TOP_K)
    m.add_argument("--no-eow-correction", action="store_true",
                   help="Plain token products as word probabilities (Heilbron et al. 2022).")
    m.add_argument("--no-case-fix", action="store_true", help="Score candidates in lowercase only.")
    m.add_argument("--no-spelling-normalization", action="store_true",
                   help="Let the models read Swiss 'ss' spellings as written.")
    m.add_argument("--no-compounds", action="store_true",
                   help="Ignore compound/prefix boundaries when syllabifying.")
    m.add_argument("--score-punctuation", action="store_true",
                   help="Also score the candidate vocabulary at punctuation rows.")
    m.add_argument("--pos-max-candidates", type=int, default=0,
                   help="Tag at most N candidates per position (0 = the whole nucleus).")
    m.add_argument("--batch-size", type=int, default=128, help="Scorer batch size (x16 tokens).")

    o = p.add_argument_group("output")
    o.add_argument("--textgrid", default=None, help="Forced-alignment TextGrid: adds onset/offset times.")
    o.add_argument("--textgrid-word-tier", default=None)
    o.add_argument("--textgrid-phone-tier", default=None)
    o.add_argument("--textgrid-lenient", action="store_true",
                   help="Warn and drop mismatched words instead of stopping.")
    o.add_argument("--qa-sample", default="80", help="Rows in qa_sample.csv: a number, 'all', or 0.")
    o.add_argument("--progress", action="store_true", help="Print progress and ETA.")
    return p


if __name__ == "__main__":
    main()
