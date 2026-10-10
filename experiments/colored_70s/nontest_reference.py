#!/usr/bin/env python3
"""Reference for the complete-estimator comparison of Section 6.5.

For every reported realization, the whitened same-sample Tu--HHI procedure of
Section 6.4 (20 x 20 upper envelope, Tu working criterion with unit noise
variance after known-spectrum whitening, HHI GIC) is run on the 4,000
non-test particles (the fitting and scoring parts of the view-stratified
60/20/20 split).  The resulting 2SDR estimator is evaluated at its selected
(p, q, r) on the same test particles as the estimator chosen by the corrected
holdout score, and the relative reduction in test MSE is recorded.

Two uses:

  # summarize the frozen output (condition means quoted in the paper)
  python experiments/colored_70s/nontest_reference.py --summarize \
      results/colored-70s/reference_tuhhi_nontest/reference_tuhhi_nontest.csv

  # recompute it (about 15 minutes on CPU; needs data/70s)
  python experiments/colored_70s/nontest_reference.py --output-dir runs/reference [--smoke]

The recomputation rebuilds each noisy realization from its frozen seed and
checks that the seed, the noise variance, and the mean-only test MSE agree
with the frozen Part B outputs in results/colored-70s/.
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(path: Path) -> dict[str, float]:
    """Condition means of the reduction and their ranges for white and colored noise."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_condition = collections.defaultdict(list)
    for row in rows:
        by_condition[(row["rho"], row["white_reference_snr"])].append(float(row["reduction"]))
    means = {key: sum(v) / len(v) for key, v in by_condition.items()}
    white = [v for key, v in means.items() if key[0] == "0.0"]
    colored = [v for key, v in means.items() if key[0] != "0.0"]
    for key in sorted(means, key=lambda k: (float(k[0]), float(k[1]))):
        print(f"rho={key[0]} s0={key[1]}: mean reduction {100 * means[key]:.1f}% "
              f"(n={len(by_condition[key])})")
    summary = {"white_min": min(white), "white_max": max(white),
               "colored_min": min(colored), "colored_max": max(colored),
               "realizations": len(rows),
               "tu_at_upper_bound": sum(r["tu_stage1_at_upper"] == "True" for r in rows)}
    print(f"white condition means {100 * summary['white_min']:.1f}% to "
          f"{100 * summary['white_max']:.1f}%; colored {100 * summary['colored_min']:.1f}% to "
          f"{100 * summary['colored_max']:.1f}%; Tu at the 20 x 20 upper bound in "
          f"{summary['tu_at_upper_bound']} of {summary['realizations']} realizations")
    return summary


def recompute(protocol_path: Path, results_dir: Path, output_dir: Path, smoke: bool) -> Path:
    cw = _module("colored_runner", HERE / "run_experiment.py")  # installs the small-core HHI rule
    mt = _module("colored_make_tables", HERE / "make_tables.py")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cfg = dict(protocol["selection"])

    groups = collections.defaultdict(list)
    for row in mt.load(results_dir, "part_b_candidates.csv"):
        groups[(row["rho"], row["white_reference_snr"], int(row["realization"]))].append(row)
    keys = sorted(groups, key=lambda k: (float(k[0]), float(k[1]), k[2]))
    if smoke:
        keys = [("0.6", "0.04", 0), ("0.8", "0.04", 0)]

    datasets = cw.white70s.discover_datasets(protocol)
    by_snr = {float(d.snr): d for d in datasets}
    stacks = {snr: cw.white70s._mrc(by_snr[snr].paths["particles.mrcs"]) for snr in (0.04, 0.06)}
    count = protocol["dataset"]["particle_count"]
    base = cw.white70s.estimate_base_noise_variance(
        stacks[0.04], stacks[0.06], low_snr=0.04, high_snr=0.06, count=count)
    del stacks
    view_ids = np.asarray([int(r["gt_view_id"]) for r in by_snr[0.05].records])[:count]
    clean_spatial = np.asarray(
        cw.white70s._mrc(by_snr[0.05].paths["clean_observed.mrcs"])[:count], dtype=np.float32)
    clean_q = cw.white70s.images_to_q_coordinates(clean_spatial)
    clean_t = torch.as_tensor(clean_q)
    shape = clean_q.shape[1:]
    centered_energy = np.square(clean_q - clean_q.mean(axis=0), dtype=np.float64).mean(axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "reference_tuhhi_nontest.csv"
    out.unlink(missing_ok=True)
    for rho_s, snr_s, k in keys:
        started = time.perf_counter()
        rho, snr = float(rho_s), float(snr_s)
        cands = groups[(rho_s, snr_s, k)]
        frozen = cands[0]
        scale = cw.separable_noise_scale(shape, rho)
        kappa = float((centered_energy / scale**2).sum() / centered_energy.sum())
        sigma2 = base / snr * kappa
        seed = (int(protocol["seeds"]["noise_base"]) + 1000 * int(round(10 * rho))
                + 100 * int(round(100 * snr)) + k)
        assert seed == int(frozen["seed"]), (rho, snr, k, seed, frozen["seed"])
        assert math.isclose(sigma2, float(frozen["sigma2"]), rel_tol=1e-9), (sigma2, frozen["sigma2"])

        z = clean_q + cw.colored_noise(count, scale, sigma2, seed)
        split = cw.view_split(view_ids, seed, protocol["part_b_split"]["fractions"])
        z_t = torch.as_tensor(z)
        test = torch.as_tensor(split["test"])
        nontest = torch.as_tensor(np.sort(np.concatenate([split["fit"], split["score"]])))

        mean_fit = z_t[torch.as_tensor(split["fit"])].reshape(len(split["fit"]), -1).mean(dim=0)
        mean_only = cw.risk(mean_fit.expand(len(test), -1), clean_t[test])
        frozen_mean = float(next(r for r in cands if r["candidate"] == "mean_only")["test_true_risk"])
        assert math.isclose(mean_only, frozen_mean, rel_tol=1e-6), (mean_only, frozen_mean)

        s = torch.as_tensor(math.sqrt(sigma2) * scale.reshape(-1), dtype=torch.float32)
        fitted = cw.fit_pipeline((z_t[nontest] / s.reshape(shape)).numpy(), 1.0, cfg)
        estimate = cw.apply_basis(z_t[test] / s.reshape(shape), fitted["mean_t"],
                                  fitted["basis"]["2sdr"]) * s
        tu_test = cw.risk(estimate, clean_t[test])

        pick = min(cands, key=lambda r: float(r["split_sure"]))
        ch_test = float(pick["test_true_risk"])
        row = {"rho": rho_s, "white_reference_snr": snr_s, "realization": k, "seed": seed,
               "tu_p": fitted["selected_p"], "tu_q": fitted["selected_q"], "tu_r": fitted["selected_r"],
               "tu_stage1_at_upper": fitted["stage1"]["selected_at_p_upper_bound"]
               or fitted["stage1"]["selected_at_q_upper_bound"],
               "tu_r_at_bound": dict(fitted["core_gic"]).get("selected_at_upper_bound"),
               "tu_test_mse": tu_test, "ch_candidate": pick["candidate"], "ch_test_mse": ch_test,
               "reduction": 1 - ch_test / tu_test, "mean_only_test_mse": mean_only}
        cw._csv_append(out, [row])
        print(f"rho={rho_s} s0={snr_s} k={k} Tu({row['tu_p']},{row['tu_q']},{row['tu_r']}) "
              f"reduction {100 * row['reduction']:.1f}% ({time.perf_counter() - started:.0f}s)")
        del z, z_t, fitted
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Section 6.5 non-test Tu--HHI reference.")
    parser.add_argument("--summarize", type=Path, metavar="CSV",
                        help="summarize a reference_tuhhi_nontest.csv and exit")
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="two realizations only")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.summarize is not None:
        summarize(args.summarize)
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --summarize is given")
    torch.set_num_threads(args.threads)
    out = recompute(args.protocol, args.results_dir, args.output_dir, args.smoke)
    if not args.smoke:
        summarize(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
