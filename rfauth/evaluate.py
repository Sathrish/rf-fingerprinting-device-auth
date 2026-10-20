"""Metrics: closed-set identification and open-set authentication (rejecting unknown devices)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


def closed_set(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def centroid_scores(emb_known_train: np.ndarray, y_train: np.ndarray, emb_test: np.ndarray) -> np.ndarray:
    """Cosine similarity of each test embedding to its nearest enrolled-device centroid."""
    def norm(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    cents = np.stack([emb_known_train[y_train == c].mean(0) for c in np.unique(y_train)])
    return (norm(emb_test) @ norm(cents).T).max(1)


def open_set(score_known: np.ndarray, score_unknown: np.ndarray) -> dict:
    """Higher score = more likely an enrolled device. Returns AUROC, EER and the
    false-accept rate of unknown devices when 95% of genuine chunks are accepted."""
    y = np.r_[np.ones(len(score_known)), np.zeros(len(score_unknown))]
    s = np.r_[score_known, score_unknown]
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    k = np.nanargmin(np.abs(fnr - fpr))
    far_at_95 = float(fpr[np.searchsorted(tpr, 0.95)])
    return {"auroc": float(roc_auc_score(y, s)), "eer": float((fpr[k] + fnr[k]) / 2),
            "far_at_95_tpr": far_at_95}
