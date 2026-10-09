# Fourier 2SDR advantage-region experiment

This isolated experiment tests one predeclared setting in which separable
dimension reduction should help: complete \(32\times32\) real orthogonal
Fourier coordinates, white \(N(0,1)\) noise, a balanced \(4\times4\)
separable signal envelope, and four diagonal core directions.

No split is used. This is deliberate and is restricted to the exact branch:
the observation is a full orthogonal real transform with iid Gaussian noise,
so the original same-sample MPCA SURE generalized degrees of freedom apply
to Stage I. HHI \(2/n\) GIC applies directly to the vector-PCA comparator. On
the adaptively selected and fitted MPCA core it is a prespecified practical
plug-in algorithm; this experiment does not claim a direct transfer of the
HHI theorem to that adaptive core. The nine Stage-I candidates are the
Cartesian grid \(p,q\in\{2,4,8\}\), represented as nested truncations of one
fitted upper envelope. MPCA uses a predeclared 250-iteration cap and
\(10^{-5}\) projector tolerance; convergence belongs to that shared fit and is
reported without excluding replicates.

The primary generator has `eta=0`. The `eta=0.02` cells put two percent of
total clean variance in a generic ambient direction orthogonal to the true
\(4\times4\) envelope. They are secondary generic-misspecification
sensitivity results, not a nearby separable-rank boundary. Signal component
variances are normalized so their population sum divided by 1024 equals the
registered per-pixel SNR.

Methods are:

- SURE-selected MPCA followed by HHI GIC (`two_sdr_sure_hhi`);
- HHI GIC on all 1024 vectorized coordinates (`vector_pca_hhi`);
- the full SURE-selected MPCA core (`mpca_sure_full_core`);
- a rank-oracle diagnostic with fitted \(p=q=r=4\)
  (`fixed_true_rank_diagnostic`).

Every risk is the exact conditional risk for predicting an independent clean
draw after the estimator has been fitted. It includes omitted signal,
transmitted test noise, and the fitted-mean term. Therefore no Monte Carlo
test set is required.

Discovery and confirmation use disjoint seed namespaces. Discovery is
non-confirmatory regardless of its outcome. Formal confirmation has 100 new
paired replicates per cell. It succeeds only if, for every `eta=0` cell, the
upper paired 95% Student-\(t\) confidence bound is below zero for both
`2SDR - vector PCA` and `2SDR - full-core MPCA`. This rule is stored in
`protocol.json` and must not be changed after inspecting results.

Run from this repository root:

```bash
fourier-2sdr-experiment same-sample-selection \
  --mode smoke --output-root runs/same-sample-smoke
fourier-2sdr-experiment same-sample-selection \
  --mode formal --output-root runs/same-sample-formal
```

Resume an interrupted attempt by passing both its original mode and
`--resume-attempt /absolute/path/to/attempt-*`. Completed per-replicate JSON
files are immutable checkpoints; aggregate CSV/JSON files are rebuilt from
them. Resume is refused if the protocol, seed namespace, runner, or selector
source hashes have changed.

Each attempt records `run.log`, `status.json`, the protocol hash and seed
namespace, source hashes, per-replicate risks/ranks/censor flags, paired
summaries, and `decision.json`. Only a complete formal run can produce a
confirmatory pass/fail decision. The decision concerns only algorithm-level
conditional prediction risk. Censoring, ambiguity, or nonconvergence is
reported separately and blocks any rank-recovery claim, but does not redefine
the risk comparison after results are seen.
