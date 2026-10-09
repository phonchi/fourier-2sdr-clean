# Fourier 2SDR

Reference implementation and reproduction package for the paper
*Fourier-Domain Two-Stage Dimension Reduction for Image Applications*.

The `fourier_2sdr` package builds the nonredundant real Fourier matrix
`Z = Q_h^T X Q_w` of a real image from a single real-input FFT
(`torch.fft.rfft2` with `norm="ortho"` followed by fixed conjugate-pair
blocks; no dense basis is formed), fits two-stage dimension reduction (2SDR)
at fixed ranks `(p, q, r)`, a `p x q` row--column envelope followed by PCA
inside the core, and implements the dimension-selection tools of the paper:
the Tu working criterion for the envelope, the HHI generalized information
criterion (GIC) for the core rank, and, for separable periodic colored noise,
Fourier whitening (a row/column rescaling of `Z`) together with the corrected
holdout score that compares complete fitted estimators on the original scale.
`experiments/` contains the code and settings of the paper's numerical
studies (Section 6), `results/` the frozen outputs behind every reported
number, and `scripts/make_paper_assets.py` regenerates the tables and figures
from those outputs.

## Install and test

Python 3.11 or newer is required.

```bash
python -m pip install -e '.[experiments]'
python -m unittest discover -s tests
```

The core package depends only on `numpy` and `torch`; the `experiments`
extra adds `scipy`, `Pillow`, and `matplotlib`, and the `notebook` extra adds
JupyterLab. `requirements-repro.txt` records the exact versions of the
environment in which the frozen results were verified (Python 3.13.12,
numpy 2.4.3, torch 2.11.0+cu128, scipy 1.17.1, Pillow 12.1.1,
matplotlib 3.10.9). All package code, experiments, and tests run on the CPU;
a GPU is used only by the optional timing sections of the benchmark notebook.

The test suite (33 tests, a few seconds) checks the Fourier coordinates
against the dense basis and their exact inversion, the fixed-rank fit and both
selection routes, the whitening identities and the unbiasedness of the
corrected holdout score, the SHA-256 manifests of `results/`, and the
byte-identical regeneration of the paper's tables and figures.

## Using the package

All functions take real `torch` tensors; the examples below run on the CPU
with random data.

**Real Fourier coordinates.** `real_fourier_transform` maps `(..., h, w)`
images to the complete real matrix `Q_h^T X Q_w`; the transform is
orthogonal (Parseval) and `real_fourier_inverse` inverts it exactly up to
floating-point error.

```python
import torch
from fourier_2sdr import real_fourier_transform, real_fourier_inverse

images = torch.randn(200, 32, 32, dtype=torch.float64)   # (n, h, w) real images
Z = real_fourier_transform(images)                        # (n, h, w) real Fourier matrices
assert torch.allclose(real_fourier_inverse(Z), images)
assert torch.allclose(Z.square().sum(), images.square().sum())
```

**Fixed-rank 2SDR.** `fit_2sdr` fits the row basis, column basis, and core
basis at ranks `(p, q, r)` by alternating updates (at most `max_iter`
iterations, projector tolerance `tol`). The returned `TwoSDRFit` exposes
`ranks`, `transform` (core scores), `reconstruct` (affine reconstruction in
Fourier coordinates), and `ambient_basis` (the orthonormal `(h*w, r)` basis
of the fitted subspace).

```python
from fourier_2sdr import fit_2sdr

# plant a (4, 4, 4) signal in unit white noise
A, _ = torch.linalg.qr(torch.randn(32, 4, dtype=torch.float64))
B, _ = torch.linalg.qr(torch.randn(32, 4, dtype=torch.float64))
xi = 8.0 * torch.randn(200, 4, dtype=torch.float64)
Z = torch.einsum("nk,pk,qk->npq", xi, A, B) + torch.randn(200, 32, 32, dtype=torch.float64)

fit = fit_2sdr(Z, (4, 4, 4))                   # ranks (p, q, r)
scores = fit.transform(Z)                      # (n, r) core scores
denoised = real_fourier_inverse(fit.reconstruct(Z))   # back to pixels
print(fit.ranks, fit.ambient_basis.shape)      # (4, 4, 4) torch.Size([1024, 4])
```

**Same-sample selection (Tu working criterion, then HHI GIC).**
`select_tu_hhi` fits one upper envelope that covers all `(p, q)` candidates,
scores every candidate as a nested truncation of that fit with the Tu working
criterion at noise variance `sigma2`, and then applies HHI GIC to the selected
core. The returned `RankSelection` carries the fit, the candidate scores, and
the censoring flags that say whether a selection hit the edge of its
candidate set.

```python
from fourier_2sdr import select_tu_hhi

selection = select_tu_hhi(
    Z,
    stage1_candidates=[(p, q) for p in (2, 4, 8) for q in (2, 4, 8)],
    sigma2=1.0,                                # known or estimated noise variance
)
print(selection.ranks)                         # (4, 4, 4) for the planted signal
print(selection.stage1_right_censored, selection.core_right_censored)
reconstruction = selection.fit.reconstruct(Z)
```

**Whitening and the corrected holdout score.** For separable periodic AR(1)
noise, `separable_noise_scale` returns the scale `S` with `S[k, l]^2 = e_k f_l`
(mean-one spectrum); whitening is the elementwise division `Z / S`. A fit in
whitened coordinates acts on the original scale as
`f(z) = mu + S V V^T S^{-1} (z - mu)`, applied by `whitened_projection`, and
`corrected_holdout_score` evaluates `mean ||z - f(z)||^2 + 2 tr(Sigma H) - tr(Sigma)`
per coordinate on an independent holdout sample, which is unbiased for the
risk of `f` under the known diagonal noise covariance `Sigma = diag(omega)`.

```python
from fourier_2sdr import (
    separable_noise_scale, separable_periodic_ar1_spectrum,
    whitened_projection, corrected_holdout_score,
)

rho, sigma2 = 0.6, 1.0
scale = torch.as_tensor(separable_noise_scale((32, 32), rho)) * sigma2 ** 0.5  # S, (h, w)
omega = torch.as_tensor(separable_periodic_ar1_spectrum((32, 32), rho)).reshape(-1) * sigma2
colored = torch.einsum("nk,pk,qk->npq", xi, A, B) + scale * torch.randn(200, 32, 32, dtype=torch.float64)
Z_fit, Z_score = colored[:150], colored[150:]

whitened_fit = fit_2sdr(Z_fit / scale, (4, 4, 4))       # fit in whitened coordinates
basis = whitened_fit.ambient_basis                       # (h*w, r), orthonormal
mean = Z_fit.mean(dim=0).reshape(-1)
estimate = whitened_projection(Z_score.reshape(len(Z_score), -1), mean, basis, scale.reshape(-1))
score = corrected_holdout_score(Z_score, estimate, basis, omega)   # per-coordinate risk estimate
```

The package also exports `select_corrected_holdout`, a simplified holdout
route: it fits each `(p, q)` candidate on one allocation, scores it with the
covariance-corrected holdout residual on a second, and selects `r` by HHI GIC
on an optional third allocation (if the third allocation is omitted, the score
allocation is reused and the Stage-II choice is not independently validated).
It does not whiten and compares envelope candidates only. The complete-estimator
selection reported in Section 6.5, which scores 2SDR candidates in raw and
whitened coordinates together with MPCA, PCA, and mean-only candidates on the
original pixel scale, is implemented in `experiments/colored_70s/run_experiment.py`.

## Reproducing the paper

Every study is run from the repository root through one dispatcher
(installed as `fourier-2sdr-experiment`, equivalently
`python -m fourier_2sdr.experiments`):

```bash
fourier-2sdr-experiment STUDY --mode smoke  --output-root DIR   # installation check
fourier-2sdr-experiment STUDY --mode formal --output-root DIR   # reported design
```

For the four protocol-driven studies an optional `--protocol FILE` replaces
the registered `protocol.json` (the controlled study's design is fixed in its
runner). Smoke mode runs a tiny configuration (a few seconds for the three
data-free studies) and is not a reported result. Each run writes its resolved
protocol, command, log, and outputs below `DIR`.

| Paper | Output | `STUDY` (runner) | Formal command | Data | Approximate formal runtime |
|---|---|---|---|---|---|
| Section 6.2 | Figure 2, Table S1 | `controlled-projection-family` (`experiments/controlled_projection_family/run_experiment.py`) | `--mode formal` (100 paired replicates per condition) | none | about 2 h on one CPU process |
| Section 6.3 | Table 3 | `same-sample-selection` (`experiments/same_sample_selection/run_experiment.py`) | `--mode formal` (100 replicates per cell; the direct runner can resume an interrupted attempt with `--resume-attempt`) | none | many hours on the CPU |
| Section 6.4 | Table 4, Figure 3 | `seventy-s-same-sample` (`experiments/seventy_s_same_sample/run_experiment.py`) | `--mode formal` (all 5,000 particles at SNR .04, .05, .06) | `data/70s` | about a minute |
| Section 6.5, calibration | Table S2 | `whitened-calibration` (`experiments/whitened_calibration/run_experiment.py`) | `--mode formal` runs `rho = 0.6`; the runner's `--rho 0.3` and `--rho 0.8` give the other rows (100 replicates each, `--threads 8`) | none | about 9 min per `rho` with 8 threads |
| Section 6.5, 70S | Table 5, Table S3 | `colored-70s` (`experiments/colored_70s/run_experiment.py`) | `--mode formal` runs `--rho 0.6 --realizations 30`; the other reported runs are listed in `experiments/colored_70s/README.md` | `data/70s` | about 5 min per realization (Parts A and B); the reported runs took hours |

The direct runner commands, with their complete flag sets, are documented in
each `experiments/*/README.md`. The synthetic 70S particle stacks (5,000
particles of `130 x 130` pixels at three SNRs, about 1.9 GB) are downloaded
and verified by

```bash
./data/fetch_data.sh 70s
```

which fetches the archives from this repository's `datasets-v1` release with
the GitHub CLI (`gh`), checks each against its recorded SHA-256, and extracts
them into `data/70s/`. Everything else runs without external data.

Reruns of the frozen designs were checked against the committed outputs: a
full rerun of Section 6.4 reproduced the selected `(p, q, r)` and PCA ranks
exactly, with per-view MSE within `1.2e-6` relative error and a byte-identical
Table 4; 24 rerun replicates of Section 6.3 matched the frozen risks exactly;
and the Section 6.5 runners reproduced the frozen candidate tables bit for bit
(one realization at `rho = 0.6`, `s0 = .04`, and five calibration replicates
at `rho = 0.6` with `--threads 8`).

## Regenerating the tables and figures

Nothing is refitted; every asset is built from `results/`:

```bash
python scripts/make_paper_assets.py            # writes build/paper_assets/
python scripts/make_paper_assets.py --check    # compares with results/paper-assets/ byte for byte
```

The script writes `table3_same_sample_risk.tex` (Table 3),
`paper1_70s_table.tex` (Table 4), `paper_70s_montage.png` (Figure 3),
`supplement_projection_family_risk.pdf` (Figure 2),
`supplement_projection_family_table.tex` (Table S1), and, through
`experiments/colored_70s/make_tables.py`, `paper1_colored_calibration.tex`
(Table S2), `paper1_colored_70s.tex` (Table 5), and
`paper1_colored_70s_partA.tex` (Table S3).

`results/` holds one folder per study with the frozen outputs of the formal
run named in its `result_manifest.json` (see
`scripts/build_result_manifests.py`, which rebuilds the manifests after an
intentional update):

| Folder | Paper | Source |
|---|---|---|
| `controlled-projection-family` | Section 6.2 | formal run `controlled_projection_family_snr004_006_seed20260721_r100` |
| `same-sample-selection` | Section 6.3 | formal run `formal_nested_v1/attempt-20260804T024153Z` |
| `seventy-s-same-sample` | Section 6.4 | formal run `formal_blockalgebra_v2/attempt-20260803T081656Z` |
| `whitened-calibration` | Section 6.5, Table S2 | formal runs `formal_rho03`, `formal_rho06`, `formal_rho08` |
| `colored-70s` | Section 6.5, Tables 5 and S3 | one subfolder per formal run (`experiments/colored_70s/README.md` maps each to its command) |
| `paper-assets` | all | reference copies of the generated tables and figures |

`tests/test_results.py` verifies the size and SHA-256 of every file in
`results/` against its manifest and runs `make_paper_assets.py --check`;
`tests/test_colored_70s.py` checks the three Section 6.5 tables against the
hashes of the manuscript copies. The in-text numbers of Section 6.5 come from
three scripts in `experiments/colored_70s/`:

```bash
python experiments/colored_70s/envelope_reduction.py --results-dir results        # MSE reduction over same-sample envelopes
python experiments/colored_70s/estimated_spectrum_summary.py --results-dir results  # estimated versus known spectrum
python experiments/colored_70s/summarize.py OUT --max-realizations 10 results/colored-70s/formal_rho06  # per-condition means with 95% intervals
```

## Notebooks

- `notebooks/01_real_fourier_coordinates.ipynb` is a teaching walkthrough of
  the real Fourier coordinates: it reproduces the Supplement's `4 x 4`
  example by hand and in code, verifies that the executed `rfft2` route equals
  `Q_h^T X Q_w`, checks phase retention, Parseval, and inversion, repeats the
  checks on a synthetic 70S particle (requires `./data/fetch_data.sh 70s`), and
  ends with a small fixed-rank 2SDR fit. Its outputs are committed, so it can
  be read without the data.
- `notebooks/02_dense_vs_rfft_pack_benchmark.ipynb` times the construction of
  the real Fourier matrix from spatial images and from an existing `rfft2`
  half spectrum against dense `Q_h^T X Q_w` products, on the CPU and, when
  available, on a GPU. Its timings are specific to the recorded hardware and
  are not claims of the paper.

Run the notebooks from the repository root after
`python -m pip install -e '.[experiments,notebook]'`.

## Repository map

```text
src/fourier_2sdr/     real Fourier coordinates (fourier.py), fixed-rank 2SDR (model.py),
                      Tu/HHI and corrected-holdout selection (selection.py),
                      whitening and the corrected holdout score (whitening.py),
                      MRC input (io.py), experiment dispatcher (experiments.py)
experiments/          one folder per numerical study: runner, protocol.json, README
results/              frozen outputs of the formal runs with SHA-256 manifests
scripts/              make_paper_assets.py, build_result_manifests.py
data/                 fetch_data.sh; downloaded stacks (data/70s/) are git-ignored
notebooks/            teaching notebook and packing benchmark
tests/                unit tests, result-manifest and asset-regeneration checks
```

## Citation

If you use this code, please cite the paper:

> *Fourier-Domain Two-Stage Dimension Reduction for Image Applications*
> (authors and venue to be added).
