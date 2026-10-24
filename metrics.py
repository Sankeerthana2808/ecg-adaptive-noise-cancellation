"""Performance metrics."""
from __future__ import annotations

import numpy as np


def snr_db(signal: np.ndarray, estimate: np.ndarray) -> float:
    """SNR (dB) of ``estimate`` relative to the ground-truth ``signal``."""
    noise = estimate - signal
    return 10 * np.log10(np.sum(signal ** 2) / (np.sum(noise ** 2) + 1e-20))


def snr_improvement_db(clean: np.ndarray, noisy: np.ndarray, cleaned: np.ndarray,
                       skip: int = 0) -> float:
    """SNR improvement (dB) after filtering; ``skip`` drops the initial convergence transient."""
    sl = slice(skip, None)
    return snr_db(clean[sl], cleaned[sl]) - snr_db(clean[sl], noisy[sl])


def convergence_sample(e: np.ndarray, clean: np.ndarray, window: int = 180,
                       threshold_ratio: float = 0.1) -> int:
    """First sample where the moving-average residual interference power falls below
    ``threshold_ratio`` of the clean-ECG power (a practical convergence-time proxy)."""
    resid = (e - clean) ** 2
    ma = np.convolve(resid, np.ones(window) / window, mode="valid")
    limit = threshold_ratio * np.mean(clean ** 2)
    below = np.where(ma < limit)[0]
    return int(below[0] + window) if below.size else len(e)
