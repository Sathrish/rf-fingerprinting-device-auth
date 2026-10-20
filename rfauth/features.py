"""Signal features: STFT spectrograms for the CNN, hand-crafted impairment features for the
classical baseline, and additive white Gaussian noise for robustness tests."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

FS = 5e6  # receiver sample rate from the SigMF metadata


def spectrograms(iq: np.ndarray, nfft: int = 128) -> np.ndarray:
    """(n, L) complex IQ -> (n, nfft, T) standardised log-power spectrograms.

    Same window as the original notebooks (nperseg=128, noverlap=64, two-sided); nfft=128
    instead of 256 halves the input size so the CNN trains on a CPU (the original nfft=256
    only zero-pads, interpolating the same 128 frequency bins).
    Each spectrogram is standardised on its own, so absolute received power (which depends on
    distance, not on the device) cannot be used as a shortcut.
    """
    _, _, S = sps.spectrogram(iq, fs=FS, nperseg=128, noverlap=64, nfft=nfft,
                              return_onesided=False, axis=-1)
    S = np.fft.fftshift(S, axes=-2)
    db = 10 * np.log10(np.abs(S) + 1e-12)
    mu = db.mean(axis=(1, 2), keepdims=True)
    sd = db.std(axis=(1, 2), keepdims=True) + 1e-8
    return ((db - mu) / sd).astype(np.float32)


def add_awgn(iq: np.ndarray, snr_db, rng: np.random.Generator) -> np.ndarray:
    """Add complex white Gaussian noise at ``snr_db`` (scalar or one value per chunk)."""
    snr = np.broadcast_to(np.asarray(snr_db, dtype=np.float64), (iq.shape[0],))[:, None]
    p_sig = np.mean(np.abs(iq) ** 2, axis=1, keepdims=True)
    p_noise = p_sig / 10 ** (snr / 10)
    noise = (rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape)) * np.sqrt(p_noise / 2)
    return (iq + noise).astype(np.complex64)


def impairment_features(iq: np.ndarray, n_psd: int = 32) -> np.ndarray:
    """Hand-crafted features of hardware impairments, one row per chunk.

    DC offset, IQ amplitude/phase imbalance, phase drift between neighbouring samples (a proxy
    for carrier frequency offset), amplitude statistics and a coarse power spectrum.
    """
    x = iq / (np.sqrt(np.mean(np.abs(iq) ** 2, axis=1, keepdims=True)) + 1e-12)
    i, q = x.real, x.imag
    feats = [
        i.mean(1), q.mean(1),                                   # DC offset
        i.std(1) / (q.std(1) + 1e-12),                          # IQ amplitude imbalance
        np.mean((i - i.mean(1, keepdims=True)) * (q - q.mean(1, keepdims=True)), 1)
        / (i.std(1) * q.std(1) + 1e-12),                        # IQ phase imbalance (correlation)
    ]
    for lag in (1, 2, 4, 8):                                    # phase drift / CFO proxy
        ac = np.mean(x[:, lag:] * np.conj(x[:, :-lag]), axis=1)
        feats += [np.angle(ac), np.abs(ac)]
    a = np.abs(x)
    am, asd = a.mean(1, keepdims=True), a.std(1, keepdims=True) + 1e-12
    feats += [a.std(1), (((a - am) / asd) ** 3).mean(1), (((a - am) / asd) ** 4).mean(1)]
    psd = np.abs(np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)) ** 2
    psd = psd.reshape(psd.shape[0], n_psd, -1).mean(2)
    psd = 10 * np.log10(psd / psd.sum(1, keepdims=True) + 1e-12)
    return np.column_stack(feats + [psd]).astype(np.float32)
