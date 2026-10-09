# Frozen results

Each folder holds the compact outputs of the formal run behind one part of
the paper: protocol, summary tables, per-replicate or per-view rows, selected
figures, and a `result_manifest.json` with the byte count and SHA-256 of
every file and the name of the source run. Raw checkpoints, logs, fitted
bases, and input data are not included.

| Folder | Paper | Source run |
|---|---|---|
| `controlled-projection-family` | Section 6.2 (Figure 2, Table S1) | `controlled_projection_family_snr004_006_seed20260721_r100` |
| `same-sample-selection` | Section 6.3 (Table 3) | `formal_nested_v1/attempt-20260804T024153Z` |
| `seventy-s-same-sample` | Section 6.4 (Table 4, Figure 3) | `formal_blockalgebra_v2/attempt-20260803T081656Z` |
| `whitened-calibration` | Section 6.5 calibration (Table S2) | `formal_rho03`, `formal_rho06`, `formal_rho08`, one subfolder per `rho` |
| `colored-70s` | Section 6.5 colored-noise 70S (Table 5, Table S3) | one subfolder per formal run; `experiments/colored_70s/README.md` maps each to its command |
| `paper-assets` | all tables and figures | reference copies written by `scripts/make_paper_assets.py` |

`python scripts/make_paper_assets.py --check` regenerates every table and
figure from these folders and compares it byte for byte with
`paper-assets/`. `tests/test_results.py` verifies every file against its
manifest. After an intentional update, run
`python scripts/build_result_manifests.py` from the repository root to
rebuild the manifests; a changed checksum without a corresponding source run
is a provenance error.
