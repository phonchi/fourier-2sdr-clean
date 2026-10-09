# Synthetic 70S no-split SURE/GIC application

This application uses the three coupled synthetic 70S stacks downloaded by
`./data/fetch_data.sh 70s`.

The scientific question is deliberately narrow: when the original 2SDR
same-sample rank-selection algorithm is applied to all 5,000 particles, which
Stage-I envelope and Stage-II ranks are selected, and what are the resulting
same-sample reconstruction errors relative to HHI-GIC PCA and the full-core
MPCA endpoint?

No data split is used. Stage I follows equations (2.1)--(2.3) of the original
2SDR paper: one predeclared upper MPCA envelope is fitted, its nested row and
column bases define all candidate models, and the Tu SURE score selects
`(p,q)`. Stage II uses HHI `2/n` GIC. The PCA comparison applies the same HHI
criterion directly to all vectorized coordinates. MPCA retains the full
`p*q` core selected by Stage I.

Rank selection uses the complete 130-by-130 real orthogonal Fourier
coordinate system. No circular mask, radial band, or PSD weighting is applied
before selection, because those operations would invalidate the iid-noise
assumption behind the original Tu SURE formula. The runner computes an
orthonormal `rfft2`, removes Hermitian redundancy with real sine--cosine
blocks, and fits the complete square `130 x 130` separable `Q` coordinate
matrix. Reconstructions are inverted to pixels before MSE is evaluated. A
complete separable orthogonal Fourier transform changes neither the exact Tu
score nor the reconstruction loss; the explicit route is retained here so
that the application demonstrates the practical Fourier data path.

The noise variance is estimated without clean targets from the coupled
SNR=.04 and SNR=.06 noisy observations. All three SNR rank selections finish
before `clean_observed.mrcs` is opened. Evaluation then uses the same 5,000
particles and the post-CTF, pre-noise clean observations. Results are a
fixed-instance same-sample application, not held-out evidence or a
generalization claim. HHI theory applies directly to vector PCA; its use on
the adaptively selected MPCA core is a pre-specified practical plug-in.

Fetch the verified release inputs, then run from this repository root:

```bash
./data/fetch_data.sh 70s
fourier-2sdr-experiment seventy-s-same-sample \
  --mode smoke --output-root runs/70s-smoke
fourier-2sdr-experiment seventy-s-same-sample \
  --mode formal --output-root runs/70s-formal
```

Quick mode reads 256 real SNR=.05 particles and is only a runtime/integration
smoke. Formal mode uses all 5,000 particles at all three registered SNRs.
Each attempt retains its command, source/input hashes, complete `run.log`,
selection and candidate tables, selected bases, per-view results, and a
montage containing all 50 view slots.
