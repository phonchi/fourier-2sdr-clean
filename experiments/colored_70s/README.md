# Colored-noise 70S: Fourier whitening and corrected holdout selection

This study adds separable stationary colored noise to the clean synthetic 70S
stack used by `experiments/seventy_s_same_sample` and asks two questions
(Section 6.5 of the main text, Tables 5 and S3):

1. Does known-spectrum whitening let the same-sample 2SDR pipeline recover
   its white-noise behaviour?
2. Can the corrected holdout score choose among complete fitted estimators on
   the original pixel scale without clean targets?

## Design

The noise is separable periodic AR(1) with lag-one correlation `rho`. It is
diagonal in the separable real Fourier coordinates, so whitening is the
row/column scaling `Z -> Z / S` (`fourier_2sdr.whitening`), which keeps the
matrix structure. The noise variance is `sigma2 = (base / s0) * kappa`, where
`base` is the noise level of the white-noise 70S stacks (estimated from the
coupled stacks without clean targets) and
`kappa = tr(C^{-1} Sigma_S) / tr(Sigma_S)` matches the mean Mahalanobis signal
energy of the white-noise reference at label `s0`; `s0` in `{.04,.05,.06}` is
reported as the white-reference SNR. Clean images are used only to generate
observations and to evaluate reconstructions.

**Part A (no split, all 5,000 particles).** The same-sample pipeline (nested
upper-envelope Tu SURE for `(p,q)`, HHI GIC for `r`) is applied to whitened
coordinates (known spectrum, Tu noise variance 1), to raw coordinates (Tu
noise variance `sigma2`, a misspecified white-noise model), and to
coordinates whitened with a separable spectrum estimated from 500 independent
noise-only images. It reports pixel MSE for 2SDR, HHI-GIC PCA and full-core
MPCA, and the selected ranks.

**Part B (split).** Within each of the 50 views the particles are split into
fit/score/test parts (60/20/20). In raw and in whitened coordinates one upper
MPCA envelope `64 x 64` is fitted on the fit part; every nested `(p,q)` on
the grid `{4,8,12,16,20,24,32,40,48,56,64}^2` gives a complete 2SDR estimator
with `r` from HHI GIC on the fit-part core, and a full-core MPCA estimator.
Vector PCA with HHI-GIC rank, vector PCA with `r` in `{5,10,...,100}`, and
the mean-only estimator complete the candidate set. The corrected holdout
score on the score part (original pixel scale, known diagonal noise
covariance) ranks all candidates; the test part gives their realized MSE.
With `--partb-estimated-psd` the estimated spectrum replaces the known one
both for whitening and in the score covariance.

All parameters and the seed rule are in `protocol.json`.

## Reported runs

The reported design is 10 realizations per `s0` at `rho = 0.6`, and 5 per
`s0` at `rho = 0` (white-noise anchor), `0.3` and `0.8`. Seeds depend only on
`(rho, s0, k)`, so the realizations below are reproduced by any run that
covers them. The frozen outputs in `results/colored-70s/<run>/` come from:

| result directory | command arguments | used rows |
|---|---|---|
| `anchor_rho00` | `--rho 0.0 --realizations 5` | all |
| `formal_rho06` | `--rho 0.6 --realizations 30` | `s0 = .04, .05`, `k < 10` |
| `formal_rho06_s006` | `--rho 0.6 --snrs 0.06 --realizations 10` | all |
| `sensitivity_rho03` | `--rho 0.3 --realizations 5` | `rho = 0.3` |
| `sensitivity_rho08` | `--rho 0.8 --realizations 5` | all |
| `formal_rho06_partb_estpsd` | `--rho 0.6 --realizations 5 --skip-part-a --partb-estimated-psd` | all |

`formal_rho06` holds 30 realizations at `s0 = .04` (10 at `s0 = .05`); the
tables use the first 10. Each command is

```bash
python experiments/colored_70s/run_experiment.py --output-root runs/colored_70s/<name> <arguments>
```

and creates `attempt-<UTC>/` with `protocol.json`, `command.json`, `run.log`,
`status.json` and `outputs/` (`part_a_pixel_mse.csv`,
`part_a_whitened_mse.csv`, `part_a_selections.csv`, `part_b_candidates.csv`,
and `montage.png` when the run covers `s0 = .05`). One realization (Parts A
and B) takes about five minutes on CPU. The 70S stacks are expected under
`data/70s/` (see `protocol.json`); `--smoke` runs a tiny configuration on
1,000 particles for checking the installation.

## Tables and summaries from the frozen results

```bash
python experiments/colored_70s/make_tables.py --results-dir results --output-dir OUT
python experiments/colored_70s/envelope_reduction.py --results-dir results
python experiments/colored_70s/estimated_spectrum_summary.py --results-dir results
python experiments/colored_70s/summarize.py OUT/summary --max-realizations 10 results/colored-70s/formal_rho06
```

- `make_tables.py` writes Table S2 (calibration, from
  `results/whitened-calibration/`), Table 5 (Part B) and Table S3 (Part A).
- `envelope_reduction.py` reports the reduction in test MSE of the
  holdout-selected estimator relative to the 2SDR candidate at the grid point
  nearest the same-sample envelope (35.6-45.8% under white noise, 43.1-68.1%
  under colored noise).
- `estimated_spectrum_summary.py` compares Part B with the estimated and the
  known spectrum on paired realizations: 9 of 15 selections share family and
  envelope (all with `r = 49`), the mean test MSE rises by 0.76%, 0.37% and
  0.55% at `s0 = .04, .05, .06`, and every estimated-spectrum selection is
  within 0.026% of the lowest test MSE among its candidates.
- `summarize.py` gives per-condition means with 95% paired t intervals.
