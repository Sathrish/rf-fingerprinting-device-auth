"""Run every evaluation protocol and save results to results/.

Protocols (training distances 2, 8, 14, 20, 26 ft unless stated):
  random_split   original-style: random 80/20 split of chunks from the same recordings
  cross_run      train on run1, validate on a time-separated part of run1, test on run2
                 (a recording session the model never saw), 2 seeds
  cross_distance the cross_run model tested on run2 at unseen distances 32-62 ft
  noise          cross_run model vs a noise-augmented model, tested at falling SNR
  open_set       authentication: train on 12 devices, reject 4 never-seen "rogue" devices
                 (2 folds of 4 rogue devices; CPU budget)
  baseline       hand-crafted impairment features + random forest, random split vs cross run

Usage:  python scripts/run_experiments.py --data D:/cyber/neu_m044q5210/KRI-16Devices-RawData
Each stage writes results/<stage>.json and is skipped if that file already exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rfauth.data import build_split, device_labels, index_dataset, read_chunks  # noqa: E402
from rfauth.evaluate import centroid_scores, closed_set, open_set, softmax  # noqa: E402
from rfauth.features import add_awgn, impairment_features, spectrograms  # noqa: E402
from rfauth.train import predict, train_cnn  # noqa: E402

TRAIN_DISTS = [2, 8, 14, 20, 26]
UNSEEN_DISTS = [32, 38, 44, 50, 56, 62]
CHUNK = 4096
N_TRAIN, N_VAL, N_TEST = 100, 25, 50          # chunks per recording
TRAIN_REGION, VAL_REGION = (0.0, 0.75), (0.8, 1.0)  # time-separated parts of run1
SNRS = [30, 20, 15, 10, 5, 0]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(RES, "log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save(name: str, obj: dict) -> None:
    with open(os.path.join(RES, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def done(name: str) -> bool:
    return os.path.exists(os.path.join(RES, f"{name}.json"))


class Data:
    """Lazily built, cached splits."""

    def __init__(self, root: str):
        self.recs = index_dataset(root)
        self.labels = device_labels(self.recs)
        self.devices = sorted(self.labels, key=self.labels.get)
        self._cache: dict = {}

    def sel(self, dists, run):
        return [r for r in self.recs if r.distance in dists and r.run == run]

    def get(self, key):
        if key not in self._cache:
            t = time.time()
            if key == "train":
                v = build_split(self.sel(TRAIN_DISTS, 1), self.labels, N_TRAIN, CHUNK, TRAIN_REGION, 1)
            elif key == "val":
                v = build_split(self.sel(TRAIN_DISTS, 1), self.labels, N_VAL, CHUNK, VAL_REGION, 2)
            elif key == "test":
                v = build_split(self.sel(TRAIN_DISTS, 2), self.labels, N_TEST, CHUNK, (0.0, 1.0), 3)
            elif key == "unseen":
                v = build_split(self.sel(UNSEEN_DISTS, 2), self.labels, N_TEST, CHUNK, (0.0, 1.0), 4)
            elif key == "random_pool":
                # exactly the original notebooks: the first 50 consecutive chunks of every file
                # (run1 and run2), later shuffled into train/test
                recs = self.sel(TRAIN_DISTS, 1) + self.sel(TRAIN_DISTS, 2)
                starts = np.arange(50) * CHUNK
                v = (np.concatenate([read_chunks(r, starts, CHUNK) for r in recs]),
                     np.repeat([self.labels[r.device] for r in recs], 50),
                     np.repeat([r.distance for r in recs], 50))
            elif key == "later_same_files":
                # same recordings as random_pool, but from later in each file (after 20%)
                recs = self.sel(TRAIN_DISTS, 1) + self.sel(TRAIN_DISTS, 2)
                v = build_split(recs, self.labels, 25, CHUNK, (0.2, 1.0), 6)
            else:
                raise KeyError(key)
            self._cache[key] = v
            log(f"loaded {key}: {v[0].shape[0]} chunks in {time.time() - t:.0f}s")
        return self._cache[key]


def per_distance(y, pred, dist) -> dict:
    return {int(d): float((pred[dist == d] == y[dist == d]).mean()) for d in np.unique(dist)}


def model_path(name: str) -> str:
    return os.path.join(MODELS, f"{name}.pt")


def get_model(D: Data, name: str, iq_tr, y_tr, iq_va, y_va, n_classes, **kw):
    """Train a model, or load it if it was trained in an earlier run."""
    from rfauth.models import RFCNN
    if os.path.exists(model_path(name)):
        m = RFCNN(n_classes)
        m.load_state_dict(torch.load(model_path(name)))
        m.eval()
        log(f"loaded saved model {name}")
        return m, None
    log(f"training {name}")
    m, hist = train_cnn(iq_tr, y_tr, iq_va, y_va, n_classes, log=log, **kw)
    torch.save(m.state_dict(), model_path(name))
    with open(os.path.join(RES, f"history_{name}.json"), "w") as f:
        json.dump(hist, f)
    return m, hist


def stage_random_split(D: Data):
    X, y, dist = D.get("random_pool")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    # original method: the "test" chunks are also used to choose the best epoch
    m, _ = get_model(D, "random_split", X[tr], y[tr], X[te], y[te], 16, seed=0)
    pred = predict(m, spectrograms(X[te])).argmax(1)
    Xl, yl, _ = D.get("later_same_files")
    pred_later = predict(m, spectrograms(Xl)).argmax(1)
    save("random_split", {
        "description": "Original notebook method: first 50 consecutive chunks of every file (first "
                       "~41 ms of each 4 s recording, run1+run2), random 80/20 split, test set also "
                       "used for epoch selection. Then the same model on chunks from later in the "
                       "same recordings.",
        "random_split_test": closed_set(y[te], pred),
        "same_model_later_in_same_files": closed_set(yl, pred_later)})


def stage_cross_run(D: Data):
    Xtr, ytr, _ = D.get("train")
    Xva, yva, _ = D.get("val")
    Xte, yte, dte = D.get("test")
    Ste = spectrograms(Xte)
    runs = []
    for seed in (0, 1):
        m, hist = get_model(D, f"cross_run_seed{seed}", Xtr, ytr, Xva, yva, 16, seed=seed)
        logits = predict(m, Ste)
        pred = logits.argmax(1)
        r = closed_set(yte, pred)
        r.update(seed=seed, per_distance=per_distance(yte, pred, dte))
        if seed == 0:
            r["confusion"] = confusion_matrix(yte, pred).tolist()
        runs.append(r)
        log(f"cross_run seed {seed}: acc {r['accuracy']:.4f}")
    accs = [r["accuracy"] for r in runs]
    save("cross_run", {
        "description": "Train run1 (first 75% of each file), validate run1 (last 20%), test run2.",
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
        "runs": runs, "devices": D.devices})


def stage_cross_distance(D: Data):
    from rfauth.models import RFCNN
    m = RFCNN(16)
    m.load_state_dict(torch.load(model_path("cross_run_seed0")))
    Xu, yu, du = D.get("unseen")
    pred = predict(m, spectrograms(Xu)).argmax(1)
    save("cross_distance", {
        "description": "cross_run seed-0 model (trained on 2-26 ft) tested on run2 at unseen distances.",
        "overall": closed_set(yu, pred), "per_distance": per_distance(yu, pred, du)})


def stage_noise(D: Data):
    from rfauth.models import RFCNN
    Xtr, ytr, _ = D.get("train")
    Xva, yva, _ = D.get("val")
    Xte, yte, _ = D.get("test")
    clean = RFCNN(16)
    clean.load_state_dict(torch.load(model_path("cross_run_seed0")))
    aug, _ = get_model(D, "cross_run_noise_aug", Xtr, ytr, Xva, yva, 16, seed=0, augment_snr=(0, 30))
    out = {"description": "Run2 test chunks with added white noise; augmented model trained with "
                          "random SNR in 0-30 dB.", "snr_db": SNRS, "clean_model": [], "augmented_model": []}
    for snr in SNRS:
        S = spectrograms(add_awgn(Xte, snr, np.random.default_rng(100 + snr)))
        for key, mdl in (("clean_model", clean), ("augmented_model", aug)):
            out[key].append(float((predict(mdl, S).argmax(1) == yte).mean()))
        log(f"SNR {snr:>3} dB  clean {out['clean_model'][-1]:.3f}  augmented {out['augmented_model'][-1]:.3f}")
    S0 = spectrograms(Xte)
    out["no_added_noise"] = {"clean_model": float((predict(clean, S0).argmax(1) == yte).mean()),
                             "augmented_model": float((predict(aug, S0).argmax(1) == yte).mean())}
    save("noise", out)


def stage_open_set(D: Data):
    Xtr, ytr, _ = D.get("train")
    Xva, yva, _ = D.get("val")
    Xte, yte, _ = D.get("test")
    Ste = spectrograms(Xte)
    folds, scores_dump = [], {}
    for fold in (0, 2):
        rogue = list(range(4 * fold, 4 * fold + 4))
        known = [c for c in range(16) if c not in rogue]
        remap = {c: i for i, c in enumerate(known)}
        ktr, kva = np.isin(ytr, known), np.isin(yva, known)
        y_tr = np.array([remap[c] for c in ytr[ktr]])
        y_va = np.array([remap[c] for c in yva[kva]])
        m, _ = get_model(D, f"open_set_fold{fold}", Xtr[ktr], y_tr, Xva[kva], y_va, 12, seed=0)
        kte = np.isin(yte, known)
        logits = predict(m, Ste)
        msp = softmax(logits).max(1)
        emb_te = predict(m, Ste, emb=True)
        emb_tr = predict(m, spectrograms(Xtr[ktr]), emb=True)
        cos = centroid_scores(emb_tr, y_tr, emb_te)
        y_known_te = np.array([remap[c] for c in yte[kte]])
        r = {"rogue_devices": [D.devices[c] for c in rogue],
             "known_closed_set": closed_set(y_known_te, logits[kte].argmax(1)),
             "max_softmax": open_set(msp[kte], msp[~kte]),
             "centroid_cosine": open_set(cos[kte], cos[~kte])}
        folds.append(r)
        scores_dump[f"fold{fold}_cos_known"], scores_dump[f"fold{fold}_cos_rogue"] = cos[kte], cos[~kte]
        scores_dump[f"fold{fold}_msp_known"], scores_dump[f"fold{fold}_msp_rogue"] = msp[kte], msp[~kte]
        log(f"open_set fold {fold}: AUROC msp {r['max_softmax']['auroc']:.3f} "
            f"cos {r['centroid_cosine']['auroc']:.3f}")
    np.savez_compressed(os.path.join(RES, "open_set_scores.npz"), **scores_dump)
    summary = {k: {m: float(np.mean([f[k][m] for f in folds])) for m in ("auroc", "eer", "far_at_95_tpr")}
               for k in ("max_softmax", "centroid_cosine")}
    summary["known_closed_set_accuracy"] = float(np.mean([f["known_closed_set"]["accuracy"] for f in folds]))
    save("open_set", {"description": "Train on 12 devices (run1); test run2 chunks of the 12 enrolled "
                                     "devices plus 4 never-seen rogue devices. 2 folds (rogues = devices 0-3 and 8-11).",
                      "mean_over_folds": summary, "folds": folds})


def stage_baseline(D: Data):
    X, y, _ = D.get("random_pool")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    F = impairment_features(X)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(F[idx[:cut]], y[idx[:cut]])
    rand = closed_set(y[idx[cut:]], rf.predict(F[idx[cut:]]))
    Xtr, ytr, _ = D.get("train")
    Xte, yte, _ = D.get("test")
    rf2 = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(impairment_features(Xtr), ytr)
    cross = closed_set(yte, rf2.predict(impairment_features(Xte)))
    log(f"baseline RF: random split {rand['accuracy']:.3f}, cross run {cross['accuracy']:.3f}")
    save("baseline", {"description": "Hand-crafted impairment features (DC offset, IQ imbalance, "
                                     "phase drift, amplitude stats, 32-bin PSD) + random forest.",
                      "random_split": rand, "cross_run": cross})


def stage_baseline_extended(D: Data):
    """The random-forest baseline under the same harder tests as the CNN."""
    Xtr, ytr, _ = D.get("train")
    Xte, yte, _ = D.get("test")
    Xu, yu, du = D.get("unseen")
    Ftr = impairment_features(Xtr)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Ftr, ytr)
    unseen_pred = rf.predict(impairment_features(Xu))
    noise = []
    for snr in SNRS:
        F = impairment_features(add_awgn(Xte, snr, np.random.default_rng(100 + snr)))
        noise.append(float((rf.predict(F) == yte).mean()))
        log(f"baseline RF SNR {snr:>3} dB: {noise[-1]:.3f}")
    # noise-augmented random forest: train features from chunks with random SNR in 0-30 dB
    rng = np.random.default_rng(0)
    Ftr_aug = impairment_features(add_awgn(Xtr, rng.uniform(0, 30, len(Xtr)), rng))
    rf_aug = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(
        np.vstack([Ftr, Ftr_aug]), np.r_[ytr, ytr])
    noise_aug = []
    for snr in SNRS:
        F = impairment_features(add_awgn(Xte, snr, np.random.default_rng(100 + snr)))
        noise_aug.append(float((rf_aug.predict(F) == yte).mean()))
    Fte = impairment_features(Xte)
    folds = []
    for fold in (0, 2):
        rogue = list(range(4 * fold, 4 * fold + 4))
        known = np.isin(ytr, rogue, invert=True)
        rfk = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Ftr[known], ytr[known])
        p = rfk.predict_proba(Fte).max(1)
        kte = np.isin(yte, rogue, invert=True)
        r = open_set(p[kte], p[~kte])
        r["known_closed_set_accuracy"] = float((rfk.predict(Fte[kte]) == yte[kte]).mean())
        r["rogue_devices"] = [D.devices[c] for c in rogue]
        folds.append(r)
        log(f"baseline RF open-set fold {fold}: AUROC {r['auroc']:.3f}")
    save("baseline_extended", {
        "description": "Random forest on impairment features: trained on run1 at 2-26 ft, tested on "
                       "run2 at unseen distances, under added noise, and for rogue-device rejection.",
        "unseen_distances": {"overall": closed_set(yu, unseen_pred),
                             "per_distance": per_distance(yu, unseen_pred, du)},
        "noise": {"snr_db": SNRS, "clean_model": noise, "augmented_model": noise_aug},
        "open_set_max_probability": {
            "mean_over_folds": {k: float(np.mean([f[k] for f in folds]))
                                for k in ("auroc", "eer", "far_at_95_tpr", "known_closed_set_accuracy")},
            "folds": folds}})


STAGES = {"baseline": stage_baseline, "baseline_extended": stage_baseline_extended, "random_split": stage_random_split, "cross_run": stage_cross_run,
          "cross_distance": stage_cross_distance, "noise": stage_noise, "open_set": stage_open_set}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to KRI-16Devices-RawData")
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(RES, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    D = Data(a.data)
    log(f"{len(D.recs)} recordings, {len(D.labels)} devices")
    for name in a.stages:
        if done(name):
            log(f"skip {name} (results exist)")
            continue
        log(f"=== {name} ===")
        STAGES[name](D)
    log("all stages finished")


if __name__ == "__main__":
    main()
