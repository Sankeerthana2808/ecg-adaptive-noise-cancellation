"""Signal generation: synthetic ECG, powerline interference, and a
two-input adaptive-noise-cancellation (ANC) scenario.

ANC setup (Widrow et al., 1975):
    primary   d[n] = s[n] + v[n]      s = clean ECG, v = interference seen through an unknown path
    reference x[n]                    correlated with v[n] but not with s[n]
    The adaptive filter estimates v[n] from x[n]; the error e[n] = d[n] - y[n]
    is the ECG estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (centre as fraction of beat, width as fraction of beat, amplitude in mV)
# Sum-of-Gaussians P-QRS-T model, a simplified form of the McSharry dynamical model.
_WAVES = {
    "P": (0.20, 0.025, 0.15),
    "Q": (0.355, 0.008, -0.12),
    "R": (0.375, 0.010, 1.20),
    "S": (0.395, 0.009, -0.25),
    "T": (0.60, 0.050, 0.30),
}


def synthetic_ecg(duration_s: float, fs: float = 360.0, heart_rate_bpm: float = 72.0,
                  hrv: float = 0.03, seed: int | None = 0) -> np.ndarray:
    """Synthetic ECG in mV with beat-to-beat RR variability (``hrv`` is the std of RR in s)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    ecg = np.zeros(n)
    mean_rr = 60.0 / heart_rate_bpm
    t_beat = 0.0
    t = np.arange(n) / fs
    while t_beat < duration_s + mean_rr:
        rr = max(0.4, rng.normal(mean_rr, hrv))
        for centre, width, amp in _WAVES.values():
            mu = t_beat + centre * rr
            sigma = width * rr
            ecg += amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
        t_beat += rr
    return ecg


def powerline_interference(n: int, fs: float, f0: float = 50.0, amplitude: float = 0.5,
                           phase: float = 0.3, drift_hz: float = 0.0) -> np.ndarray:
    """Mains interference. ``drift_hz`` adds a slow frequency drift (non-stationarity)."""
    t = np.arange(n) / fs
    inst_phase = 2 * np.pi * f0 * t + phase
    if drift_hz:
        inst_phase += 2 * np.pi * drift_hz * 0.5 * t ** 2 / t[-1]
    return amplitude * np.sin(inst_phase)


@dataclass
class Scenario:
    fs: float
    clean: np.ndarray       # s[n]  ground truth
    primary: np.ndarray     # d[n]  noisy ECG
    reference: np.ndarray   # x[n]  noise reference
    interference: np.ndarray  # v[n] true interference component in primary


def build_scenario(duration_s: float = 20.0, fs: float = 360.0, f0: float = 50.0,
                   interference_amp: float = 0.5, white_noise_std: float = 0.02,
                   ref_noise_std: float = 0.01, amplitude_step: float | None = None,
                   seed: int = 0) -> Scenario:
    """Build a primary/reference pair.

    The interference reaches the primary sensor through an unknown path
    (gain + phase shift), so the filter must learn that path. If
    ``amplitude_step`` is given, interference amplitude is multiplied by it at the
    half-way point to test tracking (non-stationary environment).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    clean = synthetic_ecg(duration_s, fs, seed=seed)
    v = powerline_interference(n, fs, f0, interference_amp, phase=0.9)
    if amplitude_step is not None:
        v[n // 2:] *= amplitude_step
    primary = clean + v + rng.normal(0, white_noise_std, n)
    reference = powerline_interference(n, fs, f0, 1.0, phase=0.0) + rng.normal(0, ref_noise_std, n)
    return Scenario(fs, clean, primary, reference, v)
