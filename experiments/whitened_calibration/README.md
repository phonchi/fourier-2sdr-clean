# Calibration of the corrected holdout score with Fourier whitening

This study checks, with a known risk minimizer, that the corrected holdout
score is unbiased for the risk of complete estimators on the original scale
when the noise is colored and the estimator whitens it (Table S2 of the supplementary
text).

## Design

Images are `32 x 32` real Fourier coordinate matrices. The signal has four
directions of variance 16 in a random `4 x 4` Kronecker envelope (geometry
seed `20260807`) and is generated in whitened coordinates together with unit
white noise; observations are `z = S z_w`, where `S^2` is the separable
periodic AR(1) spectrum with lag-one correlation `rho`
(`fourier_2sdr.whitening.periodic_ar1_spectrum`). For ideal nested
projectors the whitened risk is `pq + 16[4 - min(p,q,4)]`, so `(4,4)` is the
unique risk minimizer and lies inside the grid `{1,2,4,8,16,32}`.

Each replicate draws 500 fitting and 500 holdout observations. The 73
candidates are the mean-only estimator and MPCA fits for every `(p,q)` on the
grid, fitted either to whitened data (estimator `S P S^{-1}`, idempotent but
nonsymmetric) or to raw data (estimator `P`). For each candidate the runner
records

- `oracle_risk`: the analytic conditional risk per coordinate, given the
  fitted mean and basis;
- `corrected_split`: the corrected holdout score
  `mean ||z - f(z)||^2 + 2 tr(Sigma H) - tr(Sigma)` per coordinate, with the
  trace computed as `sum_j v_j^T Sigma v_j` (no dense `D x D` matrix);
- `uncorrected_holdout`: the holdout residual alone;
- `fit_sample_plugin`: the same correction applied to the fitting residual;
- `tu_whitened`: the Tu working score in whitened coordinates (whitened
  candidates only).

The reported design is 100 replicates for each `rho` in `{0.3, 0.6, 0.8}`
(seed rule in `protocol.json`; replicates are independent, so any prefix of a
run reproduces the corresponding rows of a full run).

## Run

```bash
python experiments/whitened_calibration/run_experiment.py --output-root runs/whitened_calibration_rho06 --rho 0.6
python experiments/whitened_calibration/run_experiment.py --output-root runs/whitened_calibration_rho03 --rho 0.3
python experiments/whitened_calibration/run_experiment.py --output-root runs/whitened_calibration_rho08 --rho 0.8
```

Each command creates `attempt-<UTC>/` with `protocol.json`, `command.json`,
`run.log`, `status.json` and `outputs/candidate_scores.csv`. A replicate takes
about 5 seconds with 8 CPU threads. `--replicates N` runs the first `N`
replicates, `--threads` sets the torch thread count (the reported runs used
8), and `--smoke` runs a tiny configuration for checking the installation.

The frozen outputs are in `results/whitened-calibration/rho{03,06,08}/`;
`python experiments/colored_70s/make_tables.py --results-dir results
--output-dir OUT` writes Table S2 from them.
