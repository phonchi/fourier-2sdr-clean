# Controlled PCA--2SDR--MPCA projection-family experiment

## Scientific question

For a fixed `24 x 8` matrix observation, this experiment checks two distinct
statements. First, unrestricted rank-matched vector PCA is the full-mode
boundary of

```text
K(A, B, G) = (A kron B) G,
```

and full-core MPCA is its `r = p q` boundary. Second, when the population signal
has a smaller row--column envelope and uses only four of its sixteen core
directions, the experiment asks whether estimating the interior structured
projector has lower finite-sample prediction risk than rank-matched vector PCA
and full-core MPCA.

The comparison concerns structured estimation and coordinate concentration.
It does **not** claim that the 2SDR projector family is larger than unrestricted
PCA. PCA can represent every rank-matched projector at the full-mode boundary.

## Fixed design

- ambient matrix: `24 x 8`, vector dimension 192;
- scenarios: `(24,8,4)`, `(2,8,4)`, `(4,4,4)`, `(8,2,4)`, and `(4,4,16)`;
- physical sample sizes: 250, 1000, and 5000;
- global signal-to-noise ratios: 0.04, 0.05, and 0.06;
- 100 paired replicates per experimental combination;
- independent paired held-out sample size: 1000;
- model and replicate master seeds: 20260721.
- fixed-envelope alternating updates: at most 300 iterations with projector
  tolerance `1e-8`.

`protocol.json` is the complete analysis definition. Oracle terminology refers
only to the supplied dimensions. Except at algebraic full-mode boundaries, row,
column, and core subspaces are estimated from the training scatter.

## Methods and measures

The three fitted methods are rank-matched vector PCA, fixed-dimension 2SDR, and
fixed-dimension full-core MPCA. All receive the same centered Wishart training
scatter and the same held-out signal/noise draw within a replicate.

The primary outcome is normalized population prediction risk,

```text
{tr[(I - Pi_hat) Sigma_signal] + tr[Pi_hat Sigma_noise]}
----------------------------------------------------------------.
                       tr(Sigma_signal)
```

The output also records normalized signal-projector error, clean-signal capture,
retained noise energy, and held-out prediction risk. Boundary identities are
checked at explicit numerical tolerances before scientific conclusions are
allowed.

## Running

From this repository root:

```bash
fourier-2sdr-experiment controlled-projection-family \
  --mode smoke --output-root runs/controlled-family-smoke
fourier-2sdr-experiment controlled-projection-family \
  --mode formal --output-root runs/controlled-family-formal
```

The output directory is always explicit.
Every run writes the resolved analysis definition, exact command, source hashes,
software versions, seed manifest, complete log, raw and summarized CSV/JSON,
manuscript-ready PDF/PNG figures, a TeX table, and `SUCCESS.json` only after all
artifacts and decision checks have been computed.

The replicate seed encodes the numeric SNR value rather than the SNR's position
in the design list. Reordering or extending the SNR list therefore does not
silently change the paired replicates for an existing SNR.

The population-risk mechanism can be reconstructed from any completed run's
raw rows without rerunning its simulation:

```bash
python experiments/controlled_projection_family/analyze_risk_decomposition.py \
  --run-dir RUN_DIR
```

where `RUN_DIR` is the output directory of a completed run. This writes raw
and summarized missed-signal/retained-noise components, an identity-checked
figure, provenance, and `SUCCESS.json` below the run's `risk_decomposition/`
directory. Use `--output-dir` to write elsewhere without adding files to that
run directory.
