"""Fast tests on small synthetic recordings (no real dataset needed)."""
import os

import numpy as np
import pytest
import torch

from rfauth.data import build_split, chunk_starts, device_labels, index_dataset
from rfauth.evaluate import centroid_scores, open_set
from rfauth.features import add_awgn, impairment_features, spectrograms
from rfauth.models import RFCNN
from rfauth.train import train_cnn

N = 4096 * 40


@pytest.fixture
def fake_root(tmp_path):
    """Two devices x two distances x two runs, each a tone at a device-specific offset."""
    rng = np.random.default_rng(0)
    for dist in (2, 8):
        d = tmp_path / f"{dist}ft"
        d.mkdir()
        for k, dev in enumerate(("3123D52", "3123D54")):
            for run in (1, 2):
                t = np.arange(N)
                x = np.exp(2j * np.pi * (0.05 + 0.1 * k) * t) + 0.1 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
                x.astype(np.complex128).tofile(d / f"WiFi_air_X310_{dev}_{dist}ft_run{run}.sigmf-data")
                (d / f"WiFi_air_X310_{dev}_{dist}ft_run{run}.sigmf-meta").write_text("{}")
    return str(tmp_path)


def test_index_parses_names(fake_root):
    recs = index_dataset(fake_root)
    assert len(recs) == 8
    assert {r.distance for r in recs} == {2, 8} and {r.run for r in recs} == {1, 2}
    assert recs[0].n_samples == N
    assert device_labels(recs) == {"3123D52": 0, "3123D54": 1}


def test_chunks_do_not_overlap_and_regions_are_disjoint():
    rng = np.random.default_rng(0)
    tr = chunk_starts(N, 20, 4096, (0.0, 0.75), rng)
    va = chunk_starts(N, 5, 4096, (0.8, 1.0), rng)
    assert np.all(np.diff(tr) >= 4096)          # no overlap inside a split
    assert tr.max() + 4096 <= 0.75 * N          # train stays in its region
    assert va.min() >= 0.8 * N                  # validation starts after a gap
    with pytest.raises(ValueError):
        chunk_starts(N, 1000, 4096, (0.0, 0.1), rng)


def test_build_split_shapes_and_labels(fake_root):
    recs = [r for r in index_dataset(fake_root) if r.run == 1]
    X, y, d = build_split(recs, device_labels(recs), 5, 4096, (0.0, 0.75), 0)
    assert X.shape == (20, 4096) and X.dtype == np.complex64
    assert sorted(set(y)) == [0, 1] and set(d) == {2, 8}


def test_spectrogram_is_standardised_and_power_invariant():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((3, 4096)) + 1j * rng.standard_normal((3, 4096))).astype(np.complex64)
    S = spectrograms(x)
    assert S.shape == (3, 128, 63)
    assert np.allclose(S.mean(axis=(1, 2)), 0, atol=1e-4)
    diff = np.abs(S - spectrograms(10 * x))                  # absolute power is removed
    assert diff.mean() < 1e-3 and diff.max() < 0.1           # (tiny float32 effects in near-empty bins)


def test_awgn_hits_requested_snr():
    rng = np.random.default_rng(2)
    x = np.exp(2j * np.pi * 0.1 * np.arange(4096))[None].repeat(200, 0).astype(np.complex64)
    noisy = add_awgn(x, 10, rng)
    snr = 10 * np.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(noisy - x) ** 2))
    assert abs(snr - 10) < 0.2


def test_impairment_features_detect_dc_offset():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((4, 4096)) + 1j * rng.standard_normal((4, 4096))).astype(np.complex64)
    F0, F1 = impairment_features(x), impairment_features(x + 0.5)
    assert F0.shape[0] == 4 and np.all(np.isfinite(F0))
    assert np.all(F1[:, 0] > F0[:, 0] + 0.2)            # mean-I feature moves with DC offset


def test_open_set_metrics():
    r = open_set(np.array([0.9, 0.8, 0.95]), np.array([0.1, 0.2, 0.3]))
    assert r["auroc"] == 1.0 and r["eer"] == 0.0
    emb = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1]], dtype=float)
    s = centroid_scores(emb, np.array([0, 0, 1, 1]), np.array([[1, 0], [-1, -1]], dtype=float))
    assert s[0] > 0.99 and s[1] < 0


def test_cnn_learns_separable_toy_data(fake_root):
    recs = index_dataset(fake_root)
    lab = device_labels(recs)
    tr = [r for r in recs if r.run == 1]
    te = [r for r in recs if r.run == 2]
    Xtr, ytr, _ = build_split(tr, lab, 28, 4096, (0.0, 0.75), 0)
    Xva, yva, _ = build_split(tr, lab, 6, 4096, (0.8, 1.0), 1)
    m, hist = train_cnn(Xtr, ytr, Xva, yva, 2, epochs=8, batch=16, log=lambda *_: None)
    Xte, yte, _ = build_split(te, lab, 8, 4096, (0.0, 1.0), 2)
    with torch.no_grad():
        m.eval()
        pred = m(torch.from_numpy(spectrograms(Xte)).unsqueeze(1)).argmax(1).numpy()
    assert (pred == yte).mean() >= 0.9
    assert hist["epochs_run"] >= 1 and 0 <= hist["best_val_acc"] <= 1


def test_model_shapes():
    m = RFCNN(16)
    x = torch.zeros(2, 1, 128, 63)
    assert m(x).shape == (2, 16) and m.embedding(x).shape == (2, 128)
