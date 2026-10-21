"""Which language model's surprisal best predicts listeners' MEG?

Data: Gehmacher et al. (2026), OSF 4m2st (source-parcellated MEG, 360
cortical parcels, 50 Hz; 4 story blocks per participant). The encoding model
follows their a02_rf_scores.py: TRF with lags -0.2..1.2 s, RidgeCV
(alphas 1e1..1e10, per target), predictors scaled without centring,
leave-one-block-out cross-validation, Spearman correlation per parcel.

Controls in every model (as in the paper): envelope, acoustic onsets, pitch,
word onset, unigram surprisal, semantic distance. Each model's lexical
surprisal is then added on its own; its contribution is the gain in held-out
correlation (Fisher z) over the controls-only model.

    python meg_compare.py <participant> <out_dir>
"""
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

mne.set_log_level("WARNING")
ROOT = Path(__file__).resolve().parents[2] / "data/gehmacher2026"
CNT = ROOT / "data/predictors/continuous/target"
NLP = ROOT / "data/predictors/word_level_gpt/target"
SURP = ROOT / "surprisal"
VARIANTS = ["theirs", "dbmdz", "gerpt2", "gerpt2_raw", "gerpt2_large"]
ALPHAS = np.logspace(1, 10, 20)
TMIN, TMAX = -0.2, 1.2


def block_name(info):
    a, b, g = (int(x) for x in np.asarray(info).ravel()[:3])
    return f"{'female' if g == 1 else 'male'}/{a}_{b}_{'f' if g == 1 else 'm'}"


def load(subject):
    """Per block: MEG (samples x 360 parcels), controls (samples x 6), surprisal variants."""
    d = pickle.load(open(ROOT / f"data/parcellated_meg/{subject}.pkl", "rb"))
    sfreq = float(d["fsample"])
    blocks = []
    for trial, info in zip(d["trial"], np.atleast_2d(d["trialinfo"])):
        name = block_name(info)
        c = np.load(CNT / f"{name}.npz")
        acoustic = np.vstack([c["envelope"], c["acoustic_onsets"], c["pitch"]])
        acoustic = mne.filter.resample(acoustic, down=float(c["fsample"]) / sfreq, npad="auto")
        y = np.asarray(trial)[:360]
        n = min(y.shape[1], acoustic.shape[1])
        words = pd.read_csv(NLP / f"{name}.csv")
        surp = pd.read_csv(SURP / f"{name}.csv")
        assert list(surp["word"].astype(str)) == list(words["word"].astype(str)), name
        idx = np.round(words["onset"].to_numpy(float) * sfreq).astype(int)
        ok = (idx >= 0) & (idx < n)
        idx = idx[ok]
        controls = np.zeros((6, n))
        controls[:3] = acoustic[:, :n]
        controls[3, idx] = 1.0
        controls[4, idx] = words.loc[ok, "unigram_surprisal"].to_numpy(float)
        controls[5, idx] = words.loc[ok, "semantic_distance"].to_numpy(float)
        surprisal = {}
        for v in VARIANTS:
            s = surp.loc[ok, v].to_numpy(float)
            s = np.where(np.isfinite(s), s, np.nanmean(s))
            x = np.zeros(n); x[idx] = s
            surprisal[v] = x
        blocks.append(dict(y=y[:, :n].T, controls=controls.T, surprisal=surprisal))
    return blocks, sfreq


def fold_scores(train, test, make_X, sfreq):
    """Held-out Spearman rho per parcel for one design."""
    X_tr = np.concatenate([make_X(b) for b in train]); X_te = np.concatenate([make_X(b) for b in test])
    y_tr = np.concatenate([b["y"] for b in train]); y_te = np.concatenate([b["y"] for b in test])
    sy = StandardScaler().fit(y_tr)
    sx = StandardScaler(with_mean=False).fit(X_tr)
    rf = mne.decoding.ReceptiveField(TMIN, TMAX, sfreq, estimator=RidgeCV(alphas=ALPHAS, alpha_per_target=True),
                                     n_jobs=1)
    rf.fit(sx.transform(X_tr), sy.transform(y_tr))
    pred = rf.predict(sx.transform(X_te))
    valid = rf.valid_samples_
    yt, yp = sy.transform(y_te)[valid], pred[valid]
    return np.array([spearmanr(yt[:, c], yp[:, c])[0] for c in range(yt.shape[1])])


def main(subject, out):
    blocks, sfreq = load(subject)
    designs = {"controls": lambda b: b["controls"]}
    for v in VARIANTS:
        designs[v] = (lambda v: lambda b: np.column_stack([b["controls"], b["surprisal"][v]]))(v)
    z = {k: [] for k in designs}
    for k in range(len(blocks)):
        train = [b for i, b in enumerate(blocks) if i != k]
        for name, make_X in designs.items():
            z[name].append(np.arctanh(fold_scores(train, [blocks[k]], make_X, sfreq)))
    res = pd.DataFrame({name: np.mean(v, axis=0) for name, v in z.items()})   # parcels x designs
    res.insert(0, "subject", subject)
    res.insert(1, "parcel", range(360))
    Path(out).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(out) / f"{subject}.csv", index=False)
    gains = {v: float((res[v] - res["controls"]).mean()) for v in VARIANTS}
    print(subject, "controls z =", round(float(res["controls"].mean()), 4),
          "| mean gain:", {k: round(g, 5) for k, g in gains.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
