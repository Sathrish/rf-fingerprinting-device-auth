"""Dataset indexing and chunk reading for the ORACLE KRI-16Devices-RawData recordings.

Each recording is one transmitter at one distance in one run (run1 / run2, ~9.5 min apart).
The SigMF metadata says ``cf32``, but the files are really complex128: reading them as
complex64 gives garbage values, while complex128 gives the stated 20,006,400 samples.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

DTYPE = np.complex128
FILE_RE = re.compile(r"WiFi_air_X310_(?P<dev>[0-9A-F]+)_(?P<dist>\d+)ft_run(?P<run>\d)\.sigmf-data$")


@dataclass(frozen=True)
class Recording:
    device: str
    distance: int  # feet
    run: int
    path: str

    @property
    def n_samples(self) -> int:
        return os.path.getsize(self.path) // np.dtype(DTYPE).itemsize


def index_dataset(root: str) -> list[Recording]:
    """Return every recording under ``root`` (one folder per distance)."""
    recs = []
    for dist_dir in sorted(os.listdir(root)):
        full = os.path.join(root, dist_dir)
        if not os.path.isdir(full):
            continue
        for name in sorted(os.listdir(full)):
            m = FILE_RE.match(name)
            if m:
                recs.append(Recording(m["dev"], int(m["dist"]), int(m["run"]), os.path.join(full, name)))
    return recs


def device_labels(recs: list[Recording]) -> dict[str, int]:
    return {d: i for i, d in enumerate(sorted({r.device for r in recs}))}


def chunk_starts(n_samples: int, n_chunks: int, chunk_len: int, region: tuple[float, float],
                 rng: np.random.Generator) -> np.ndarray:
    """Non-overlapping chunk start offsets drawn from a fractional ``region`` of the file.

    Drawing train and validation chunks from disjoint regions (e.g. 0-0.75 and 0.8-1.0)
    keeps them separated in time, so neighbouring samples never land on both sides.
    """
    lo, hi = int(region[0] * n_samples), int(region[1] * n_samples)
    slots = (hi - lo) // chunk_len
    if slots < n_chunks:
        raise ValueError(f"region holds {slots} chunks, {n_chunks} requested")
    picked = np.sort(rng.choice(slots, size=n_chunks, replace=False))
    return lo + picked * chunk_len


def read_chunks(rec: Recording, starts: np.ndarray, chunk_len: int) -> np.ndarray:
    """Read IQ chunks (n, chunk_len) complex64 from a recording via memory mapping."""
    mm = np.memmap(rec.path, dtype=DTYPE, mode="r")
    out = np.stack([np.asarray(mm[s:s + chunk_len]) for s in starts])
    return out.astype(np.complex64)


def build_split(recs: list[Recording], labels: dict[str, int], n_chunks: int, chunk_len: int,
                region: tuple[float, float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IQ chunks, labels and distances for a list of recordings."""
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for r in recs:
        starts = chunk_starts(r.n_samples, n_chunks, chunk_len, region, rng)
        X.append(read_chunks(r, starts, chunk_len))
        y += [labels[r.device]] * n_chunks
        d += [r.distance] * n_chunks
    return np.concatenate(X), np.array(y), np.array(d)
