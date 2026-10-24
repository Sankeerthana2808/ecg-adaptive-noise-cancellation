"""Run all experiments and save figures + a metrics table into ./results.

Usage:  python -m ecganc.experiments [--out results]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .filters import lms, nlms, rls
from .metrics import convergence_sample, snr_db, snr_improvement_db
from .signals import build_scenario

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"LMS": "#d95f02", "NLMS": "#1b9e77", "RLS": "#7570b3"}


def run_all(sc, order=4):
    out = {}
    for name, fn, kw in [("LMS", lms, dict(mu=0.02)),
                         ("NLMS", nlms, dict(mu=0.05)),
                         ("RLS", rls, dict(lam=0.999))]:
        t0 = time.perf_counter()
        y, e, w = fn(sc.primary, sc.reference, order=order, **kw)
        out[name] = dict(y=y, e=e, w=w, runtime_s=time.perf_counter() - t0)
    return out


def fig_waveforms(sc, res, path):
    t = np.arange(len(sc.clean)) / sc.fs
    win = (t >= 8) & (t <= 11)
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t[win], sc.clean[win], "k"); ax[0].set_title("Clean ECG (ground truth)")
    ax[1].plot(t[win], sc.primary[win], color="#999"); ax[1].set_title("Primary input: ECG + 50 Hz interference + noise")
    for a, nm in zip(ax[2:], ["NLMS", "RLS"]):
        a.plot(t[win], res[nm]["e"][win], color=COL[nm], label=f"{nm} output")
        a.plot(t[win], sc.clean[win], "k--", lw=0.7, alpha=0.6, label="truth")
        a.set_title(f"{nm} filtered output"); a.legend(loc="upper right", fontsize=8)
    ax[-1].set_xlabel("Time (s)")
    for a in ax: a.set_ylabel("mV")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_learning_curves(sc, res, path):
    win = 90
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nm, r in res.items():
        mse = np.convolve((r["e"] - sc.clean) ** 2, np.ones(win) / win, mode="valid")
        ax.semilogy(np.arange(len(mse)) / sc.fs, mse, label=nm, color=COL[nm])
    ax.set_xlim(0, 6); ax.set_xlabel("Time (s)"); ax.set_ylabel("Residual MSE (mV$^2$)")
    ax.set_title("Learning curves (moving-average residual error vs. true ECG)")
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_stepsize_sweep(sc, path):
    mus = np.logspace(-3.5, -0.9, 14)
    skip = int(5 * sc.fs)
    gains, conv = [], []
    for mu in mus:
        _, e, _ = lms(sc.primary, sc.reference, order=4, mu=mu)
        gains.append(snr_improvement_db(sc.clean, sc.primary, e, skip))
        conv.append(convergence_sample(e, sc.clean) / sc.fs)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogx(mus, gains, "o-", color=COL["LMS"]); ax[0].set_xlabel("LMS step size $\\mu$")
    ax[0].set_ylabel("Steady-state SNR gain (dB)"); ax[0].set_title("Accuracy vs. step size")
    ax[1].loglog(mus, conv, "o-", color=COL["LMS"]); ax[1].set_xlabel("LMS step size $\\mu$")
    ax[1].set_ylabel("Convergence time (s)"); ax[1].set_title("Speed vs. step size")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return mus.tolist(), gains, conv


def fig_snr_sweep(path):
    amps = [0.1, 0.25, 0.5, 1.0, 2.0]
    rows = {k: [] for k in COL}; snr_in = []
    for a in amps:
        sc = build_scenario(duration_s=15, interference_amp=a, seed=3)
        skip = int(5 * sc.fs)
        snr_in.append(snr_db(sc.clean[skip:], sc.primary[skip:]))
        for nm, r in run_all(sc).items():
            rows[nm].append(snr_db(sc.clean[skip:], r["e"][skip:]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(snr_in, snr_in, "k--", label="no filter")
    for nm, v in rows.items(): ax.plot(snr_in, v, "o-", label=nm, color=COL[nm])
    ax.set_xlabel("Input SNR (dB)"); ax.set_ylabel("Output SNR (dB)")
    ax.set_title("Output SNR vs. input SNR"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return snr_in, rows


def fig_tracking(path):
    sc = build_scenario(duration_s=20, amplitude_step=3.0, seed=5)
    res = run_all(sc)
    win = 90; half = len(sc.clean) // 2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nm, r in res.items():
        mse = np.convolve((r["e"] - sc.clean) ** 2, np.ones(win) / win, mode="valid")
        ax.semilogy(np.arange(len(mse)) / sc.fs, mse, label=nm, color=COL[nm])
    ax.axvline(half / sc.fs, color="k", ls=":"); ax.text(half / sc.fs + 0.2, 0.3, "interference\namplitude x3", fontsize=8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Residual MSE (mV$^2$)")
    ax.set_title("Tracking a sudden change in interference amplitude"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    # Recovery time: first time after the step that the moving-average residual MSE is
    # within 2x of the filter's own pre-step steady-state level (median over 5-9.5 s).
    rec = {}
    for nm, r in res.items():
        mse = np.convolve((r["e"] - sc.clean) ** 2, np.ones(win) / win, mode="valid")
        base = np.median(mse[int(5 * sc.fs): int(9.5 * sc.fs)])
        after = mse[half:]
        ok = np.where(after < 2 * base)[0]
        rec[nm] = round(float(ok[0] / sc.fs), 2) if ok.size else float("inf")
    return rec


def fig_rls_lambda(sc, path):
    lams = [0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
    skip = int(5 * sc.fs); g = []
    for lam in lams:
        _, e, _ = rls(sc.primary, sc.reference, order=4, lam=lam)
        g.append(snr_improvement_db(sc.clean, sc.primary, e, skip))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(lams, g, "o-", color=COL["RLS"]); ax.set_xlabel("Forgetting factor $\\lambda$")
    ax.set_ylabel("Steady-state SNR gain (dB)"); ax.set_title("RLS: effect of forgetting factor")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return lams, g


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results")
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)

    sc = build_scenario(duration_s=20, seed=0)
    res = run_all(sc)
    skip = int(5 * sc.fs)
    table = {}
    for nm, r in res.items():
        table[nm] = dict(
            input_snr_db=round(snr_db(sc.clean[skip:], sc.primary[skip:]), 2),
            output_snr_db=round(snr_db(sc.clean[skip:], r["e"][skip:]), 2),
            snr_gain_db=round(snr_improvement_db(sc.clean, sc.primary, r["e"], skip), 2),
            convergence_s=round(convergence_sample(r["e"], sc.clean) / sc.fs, 3),
            runtime_s=round(r["runtime_s"], 3),
        )
    fig_waveforms(sc, res, out / "waveforms.png")
    fig_learning_curves(sc, res, out / "learning_curves.png")
    mus, gains, conv = fig_stepsize_sweep(sc, out / "lms_stepsize_sweep.png")
    snr_in, rows = fig_snr_sweep(out / "snr_sweep.png")
    rec = fig_tracking(out / "tracking.png")
    lams, lam_g = fig_rls_lambda(sc, out / "rls_lambda.png")

    summary = dict(main_table=table, tracking_recovery_s=rec,
                   lms_sweep=dict(mu=mus, snr_gain_db=gains, convergence_s=conv),
                   snr_sweep=dict(input_snr_db=snr_in, output_snr_db=rows),
                   rls_lambda=dict(lambda_=lams, snr_gain_db=lam_g),
                   config=dict(fs=sc.fs, duration_s=20, filter_order=4, f0_hz=50))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(table, indent=2)); print("tracking recovery (s):", rec)


if __name__ == "__main__":
    main()
