# Adaptive Noise Cancellation for ECG Signals (LMS, NLMS, RLS)

Removal of 50 Hz powerline interference from ECG using adaptive filters implemented **from scratch in NumPy**, with a comparative study of convergence speed, steady-state accuracy, noise robustness and tracking ability.

> Course/graduate-level DSP project (Adaptive Signal Processing / Biomedical Signal Processing).

![Filtered waveforms](results/waveforms.png)

## 1. Problem

Mains interference (50 Hz in India/EU, 60 Hz in the US) corrupts ECG recordings and can completely mask the P and T waves. A fixed notch filter also distorts the ECG around 50 Hz and cannot follow changes in the interference path. An **adaptive noise canceller** (Widrow et al., 1975) uses a second *reference* input correlated with the interference to estimate and subtract it.

```
 primary   d[n] = s[n] + v[n]  ----------------(+)---> e[n] = d[n] - y[n]  ≈  s[n] (clean ECG)
                                                 ^ -
 reference x[n] ---> [ adaptive filter w ] ---> y[n] ≈ v[n]
                              ^                        |
                              +------ update rule -----+
```

## 2. Methods

| Algorithm | Update | Cost / sample | Key parameter |
|---|---|---|---|
| LMS | `w ← w + μ e u` | O(M) | step size μ |
| NLMS | `w ← w + μ e u / (ε + uᵀu)` | O(M) | normalised step μ ∈ (0, 2) |
| RLS | gain vector `k = Pu / (λ + uᵀPu)`, `P ← (P − k uᵀP)/λ` | O(M²) | forgetting factor λ |

*Test signal*: synthetic P-QRS-T ECG (sum-of-Gaussians model, 72 bpm with RR variability, fs = 360 Hz) plus 0.5 mV, 50 Hz interference arriving through an unknown gain/phase path and 0.02 mV white noise. The reference is a 50 Hz tone with small additive noise. Filter order M = 4. Ground truth is available, so all metrics are exact.

## 3. Results (20 s record, input SNR = −6.1 dB)

| Filter | Output SNR (dB) | SNR gain (dB) | Convergence (s) | Runtime (s) |
|---|---|---|---|---|
| LMS (μ = 0.02) | 18.3 | **24.4** | 0.64 | 0.02 |
| NLMS (μ = 0.05) | 15.0 | 21.1 | 0.79 | 0.03 |
| RLS (λ = 0.999) | 18.7 | **24.7** | **0.50** | 0.08 |

SNR is measured after discarding the first 5 s so only steady-state performance is counted. Convergence = first time the moving-average residual power drops below 10 % of the ECG power. Runtime is pure-Python loops on one CPU core and indicates relative cost only.

### Findings

1. **RLS converges fastest and reaches the lowest steady-state error**, but costs ~4× the compute of LMS even at M = 4 (it is O(M²)).
2. **Step-size trade-off in LMS** (`results/lms_stepsize_sweep.png`): small μ is accurate but slow (≈17 s to converge at μ ≈ 3·10⁻⁴); large μ converges fast but loses accuracy (gain falls from 24.8 dB to 18.7 dB at the largest μ tested).
3. **NLMS needs a much smaller μ than the textbook default here.** At μ = 0.5 it only achieved 4.6 dB of gain on a 12 s test record; μ = 0.05 gave 18 dB. The ECG itself acts as measurement noise during adaptation, so a large normalised step causes large misadjustment. This is why μ = 0.05 is used above.
4. **Robustness to interference strength** (`results/snr_sweep.png`): from 7.5 dB down to −18 dB input SNR, LMS and RLS lose only ~2 dB of output SNR; NLMS degrades much more (9.4 dB output at −18 dB input).
5. **Tracking trade-off** (`results/tracking.png`): when the interference amplitude suddenly triples, LMS and NLMS recover in 0.4 s and 0.16 s, but RLS with λ = 0.999 takes ≈ 9.6 s. This matches theory: RLS weight error decays with a time constant of about 1/(1−λ) = 1000 samples. Lower λ tracks faster but is noisier in steady state (`results/rls_lambda.png`: gain drops from 24.7 dB at λ = 1 to 19.5 dB at λ = 0.95).

## 4. Run it

```bash
git clone https://github.com/sankeerthana2808/ecg-adaptive-noise-cancellation.git
cd ecg-adaptive-noise-cancellation
pip install -r requirements.txt
PYTHONPATH=src python -m ecganc.experiments --out results   # regenerates all figures + metrics.json
PYTHONPATH=src python -m pytest -q                          # unit tests
```

Tests check that every filter improves SNR by > 10 dB, that NLMS/RLS recover a known FIR system exactly from noiseless data, that RLS converges faster than LMS, and that `e = d − y`.

## 5. Project layout

```
src/ecganc/signals.py       synthetic ECG, interference, scenario builder
src/ecganc/filters.py       LMS, NLMS, RLS (from scratch)
src/ecganc/metrics.py       SNR, SNR improvement, convergence time
src/ecganc/experiments.py   all experiments, figures, metrics.json
tests/test_filters.py       unit tests
results/                    generated figures and metrics
```

## 6. Limitations and future work

- Evaluation uses **synthetic** ECG so ground truth is exact; validating on real recordings (e.g. MIT-BIH Arrhythmia Database via the `wfdb` package, with added mains noise) is the natural next step.
- The reference is an ideal 50 Hz tone. A real system would take it from a mains pick-up electrode, or synthesise it from a PLL.
- Next steps: frequency-domain / subband LMS, variable step-size LMS to remove the speed-accuracy trade-off, baseline-wander and EMG removal, and a fixed-point / FPGA implementation of the NLMS core.

## References

1. B. Widrow et al., "Adaptive noise cancelling: principles and applications," *Proc. IEEE*, 63(12), 1975.
2. S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2013.
3. P. McSharry et al., "A dynamical model for generating synthetic ECG signals," *IEEE Trans. Biomed. Eng.*, 50(3), 2003.

## License

MIT
