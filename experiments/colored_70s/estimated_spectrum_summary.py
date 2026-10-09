#!/usr/bin/env python3
"""Part B with an estimated noise spectrum versus the known spectrum.

Run ``formal_rho06_partb_estpsd`` repeats Part B at rho = 0.6 for the first
five realizations per s0, with the separable spectrum estimated from 500
noise-only images replacing the known one both for whitening and in the
score's noise covariance.  Seeds equal those of the known-spectrum runs
(``formal_rho06`` for s0 = .04, .05 and ``formal_rho06_s006`` for s0 = .06),
so every realization is paired.  The script reports

* the number of paired realizations in which both selections (minimum
  corrected holdout score) share the same family and envelope ``(p,q)``, and
  the selected ``r`` in those realizations;
* per s0, the rise in mean test MSE of the estimated-spectrum selection,
  ``mean(test MSE, estimated) / mean(test MSE, known) - 1``;
* for each estimated-spectrum selection, its test MSE relative to the lowest
  test MSE among the candidates of the same run, ``test / min test - 1``.

Reads frozen CSVs only.
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics as st
from pathlib import Path

RHO = "0.6"
REALIZATIONS = 5
KNOWN_RUNS = ("formal_rho06", "formal_rho06_s006")
ESTIMATED_RUN = "formal_rho06_partb_estpsd"


def _groups(results_dir: Path, runs) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups = collections.defaultdict(list)
    for run in runs:
        path = Path(results_dir) / "colored-70s" / run / "part_b_candidates.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["rho"] == RHO and int(row["realization"]) < REALIZATIONS:
                    groups[(row["white_reference_snr"], row["realization"])].append(row)
    return groups


def _pick(candidates):
    return min(candidates, key=lambda r: float(r["split_sure"]))


def summarize(results_dir: Path) -> dict[str, object]:
    known = _groups(results_dir, KNOWN_RUNS)
    estimated = _groups(results_dir, (ESTIMATED_RUN,))
    if set(known) != set(estimated):
        raise RuntimeError("known- and estimated-spectrum realizations are not paired")
    same, same_r = 0, collections.Counter()
    tests = collections.defaultdict(lambda: ([], []))
    within = []
    for key in sorted(estimated):
        pick_est, pick_known = _pick(estimated[key]), _pick(known[key])
        if (pick_est["family"], pick_est["p"], pick_est["q"]) == (
                pick_known["family"], pick_known["p"], pick_known["q"]):
            same += 1
            same_r[(int(pick_est["r"]), int(pick_known["r"]))] += 1
        tests[key[0]][0].append(float(pick_est["test_true_risk"]))
        tests[key[0]][1].append(float(pick_known["test_true_risk"]))
        lowest = min(float(r["test_true_risk"]) for r in estimated[key])
        within.append(float(pick_est["test_true_risk"]) / lowest - 1)
    rise = {s0: st.mean(est) / st.mean(kn) - 1 for s0, (est, kn) in sorted(tests.items())}
    return {"paired": len(estimated), "same_envelope": same,
            "same_envelope_r": dict(same_r), "mean_test_rise": rise,
            "max_within_lowest": max(within)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="root of the frozen results (contains colored-70s/)")
    args = parser.parse_args()
    out = summarize(args.results_dir)
    print(f"same family and envelope (p,q): {out['same_envelope']} of {out['paired']} "
          f"paired realizations; (r estimated, r known) counts {out['same_envelope_r']}")
    for s0, value in out["mean_test_rise"].items():
        print(f"s0 = {s0}: mean test MSE rise {100 * value:.2f}%")
    print(f"largest excess of a selection over the lowest test MSE among its "
          f"candidates: {100 * out['max_within_lowest']:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
