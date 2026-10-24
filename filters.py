"""Adaptive filters implemented from scratch with NumPy.

All functions share one signature and return ``(y, e, w_final)``:
    d : primary input (desired signal + interference)
    x : reference input
    y : filter output (interference estimate)
    e : error = d - y  (this is the cleaned ECG estimate in ANC)
"""
from __future__ import annotations

import numpy as np


def _delay_line(x: np.ndarray, n: int, order: int) -> np.ndarray:
    """Return [x[n], x[n-1], ..., x[n-order+1]] with zero padding at the start."""
    start = n - order + 1
    if start >= 0:
        return x[start:n + 1][::-1]
    buf = np.zeros(order)
    buf[: n + 1] = x[: n + 1][::-1]
    return buf


def lms(d: np.ndarray, x: np.ndarray, order: int = 8, mu: float = 0.01):
    """Least-mean-squares: w <- w + mu * e * u.  Stable for 0 < mu < 2 / (order * E[x^2])."""
    n_samples = len(d)
    w = np.zeros(order)
    y = np.zeros(n_samples)
    e = np.zeros(n_samples)
    for n in range(n_samples):
        u = _delay_line(x, n, order)
        y[n] = w @ u
        e[n] = d[n] - y[n]
        w += mu * e[n] * u
    return y, e, w


def nlms(d: np.ndarray, x: np.ndarray, order: int = 8, mu: float = 0.5, eps: float = 1e-6):
    """Normalised LMS: step size divided by input power. Stable for 0 < mu < 2."""
    n_samples = len(d)
    w = np.zeros(order)
    y = np.zeros(n_samples)
    e = np.zeros(n_samples)
    for n in range(n_samples):
        u = _delay_line(x, n, order)
        y[n] = w @ u
        e[n] = d[n] - y[n]
        w += (mu / (eps + u @ u)) * e[n] * u
    return y, e, w


def rls(d: np.ndarray, x: np.ndarray, order: int = 8, lam: float = 0.999, delta: float = 100.0):
    """Exponentially weighted recursive least squares.

    lam   forgetting factor (0.9 < lam <= 1); effective memory ~ 1 / (1 - lam) samples
    delta initialises P = delta * I (large delta = fast initial convergence)
    """
    n_samples = len(d)
    w = np.zeros(order)
    P = delta * np.eye(order)
    y = np.zeros(n_samples)
    e = np.zeros(n_samples)
    for n in range(n_samples):
        u = _delay_line(x, n, order)
        Pu = P @ u
        k = Pu / (lam + u @ Pu)          # gain vector
        y[n] = w @ u                      # a priori output
        e[n] = d[n] - y[n]                # a priori error
        w = w + k * e[n]
        P = (P - np.outer(k, Pu)) / lam
        P = 0.5 * (P + P.T)               # keep P symmetric (numerical stability)
    return y, e, w
