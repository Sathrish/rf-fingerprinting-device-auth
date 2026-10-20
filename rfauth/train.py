"""Training with a separate validation split for model selection (the test set is never used
to pick epochs), a one-cycle learning-rate schedule, and optional on-the-fly noise augmentation.

Validation accuracy of this CNN fluctuates strongly from epoch to epoch, so training runs the full
schedule by default (patience = epochs) and keeps the epoch with the best validation accuracy."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from .features import add_awgn, spectrograms
from .models import RFCNN


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: RFCNN, S: np.ndarray, batch: int = 256, emb: bool = False) -> np.ndarray:
    """Logits (or embeddings) for spectrograms ``S`` of shape (n, 256, T)."""
    model.eval()
    outs = []
    for k in range(0, len(S), batch):
        xb = torch.from_numpy(S[k:k + batch]).unsqueeze(1)
        outs.append((model.embedding(xb) if emb else model(xb)).numpy())
    return np.concatenate(outs)


def train_cnn(iq_tr: np.ndarray, y_tr: np.ndarray, iq_va: np.ndarray, y_va: np.ndarray,
              n_classes: int, seed: int = 0, epochs: int = 40, patience: int = 40, batch: int = 64,
              lr: float = 3e-3, width: int = 8, augment_snr: tuple[float, float] | None = None,
              log=print) -> tuple[RFCNN, dict]:
    """Train the CNN; select the epoch with the best *validation* accuracy.

    ``augment_snr=(lo, hi)`` adds white noise at a random SNR in [lo, hi] dB to every training
    chunk, freshly drawn each epoch.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = RFCNN(n_classes, width=width)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = (len(iq_tr) + batch - 1) // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    S_tr_clean = None if augment_snr else spectrograms(iq_tr)
    S_va = spectrograms(iq_va)
    hist = {"train_acc": [], "val_acc": [], "epoch_s": []}
    best, best_acc, bad = None, -1.0, 0
    for ep in range(epochs):
        t0 = time.time()
        if augment_snr:
            snr = rng.uniform(*augment_snr, size=len(iq_tr))
            S_tr = spectrograms(add_awgn(iq_tr, snr, rng))
        else:
            S_tr = S_tr_clean
        model.train()
        order = rng.permutation(len(S_tr))
        correct = 0
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            xb = torch.from_numpy(S_tr[idx]).unsqueeze(1)
            yb = torch.from_numpy(y_tr[idx]).long()
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            correct += (out.argmax(1) == yb).sum().item()
        tr_acc = correct / len(S_tr)
        va_acc = float((predict(model, S_va).argmax(1) == y_va).mean())
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(va_acc)
        hist["epoch_s"].append(time.time() - t0)
        log(f"  epoch {ep + 1:2d}  train {tr_acc:.3f}  val {va_acc:.3f}  ({hist['epoch_s'][-1]:.0f}s)")
        if va_acc > best_acc:
            best_acc, best, bad = va_acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop at epoch {ep + 1}")
                break
    model.load_state_dict(best)
    hist["best_val_acc"] = best_acc
    hist["epochs_run"] = len(hist["val_acc"])
    return model, hist
