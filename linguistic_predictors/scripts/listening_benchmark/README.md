# Listening benchmark: which language model best predicts listeners' MEG?

Independent German listening data: Gehmacher, Kaltenmaier, Schubert, Weisz &
Press (2026), bioRxiv, doi:10.64898/2026.06.12.731866; data at
https://osf.io/4m2st (source-parcellated MEG, 360 cortical parcels, 50 Hz;
German audiobook stories with forced-aligned word onsets).

For each language model, word surprisal is added to the paper's own
encoding model (envelope, acoustic onsets, pitch, word onset, unigram
surprisal, semantic distance; TRF lags -0.2..1.2 s; RidgeCV; leave-one-block-
out; Spearman per parcel). The score is the held-out gain over the
controls-only model.

```bash
python scripts/listening_benchmark/download_data.py data/gehmacher2026 P3,P4,P5,P6,P9,P15,P18,P19
python scripts/listening_benchmark/surprisal.py data/gehmacher2026/surprisal          # nlp_env
python scripts/listening_benchmark/compare.py P3 data/gehmacher2026/meg_scores        # mne env, per participant
python scripts/listening_benchmark/summary.py data/gehmacher2026/meg_scores
```

Result (8 participants): docs/listening_benchmark_results.csv.
