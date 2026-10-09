#!/usr/bin/env python3
"""Reduction in test MSE of the holdout-selected estimator relative to the
same-sample envelope (Section 6.5 and Supplementary Section S7.D).

Definition:
  * selected estimator: the Part B candidate minimizing the corrected
    holdout score (split_sure);
  * baseline: the Part B 2SDR candidate, in the same coordinates (raw or
    whitened) as the selected estimator, whose (p,q) is the grid point
    nearest to the whitened same-sample (Part A) envelope of the same
    realization, each axis separately, ties to the smaller grid value;
    it is fitted on the same 60% fitting part with r from HHI GIC;
  * reduction 1 - test MSE(selected) / test MSE(baseline) per realization,
    averaged within each (rho, s0) condition; the reported ranges are the
    min and max of these condition means (white: rho = 0; colored: rho > 0).
Reads frozen CSVs only, through make_tables.load().
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("colored_70s_make_tables", HERE / "make_tables.py")
mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mt)

GRID = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]


def nearest(value: int) -> int:
    return min(GRID, key=lambda g: (abs(g - value), g))


def reductions(results_dir: Path):
    candidates = mt.load(results_dir, "part_b_candidates.csv")
    selections = mt.load(results_dir, "part_a_selections.csv")
    same_sample = {(r["rho"], r["white_reference_snr"], r["realization"]): (int(r["p"]), int(r["q"]))
                   for r in selections if r["variant"] == "white"}
    groups = collections.defaultdict(list)
    for r in candidates:
        groups[(r["rho"], r["white_reference_snr"], r["realization"])].append(r)
    cut = collections.defaultdict(list)
    refs = collections.defaultdict(collections.Counter)
    for key, cands in groups.items():
        pick = min(cands, key=lambda r: float(r["split_sure"]))
        p, q = (nearest(v) for v in same_sample[key])
        family = pick["family"].split("_")[0]
        ref = next(r for r in cands if r["candidate"] == f"{family}_2sdr_p{p}_q{q}")
        cut[key[:2]].append(1 - float(pick["test_true_risk"]) / float(ref["test_true_risk"]))
        refs[key[:2]][(family, p, q)] += 1
    return cut, refs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="root of the frozen results (contains colored-70s/)")
    args = parser.parse_args()
    cut, refs = reductions(args.results_dir)
    means = {}
    for k in sorted(cut):
        means[k] = st.mean(cut[k])
        print(k, dict(refs[k]),
              f"mean {100 * means[k]:.1f}%  min {100 * min(cut[k]):.1f}%  n={len(cut[k])}")
    white = [v for k, v in means.items() if k[0] == "0.0"]
    colored = [v for k, v in means.items() if k[0] != "0.0"]
    print(f"white condition means: {100 * min(white):.1f}% to {100 * max(white):.1f}%")
    print(f"colored condition means: {100 * min(colored):.1f}% to {100 * max(colored):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
