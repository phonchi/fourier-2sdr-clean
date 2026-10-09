#!/usr/bin/env python3
"""Summarize colored-noise 70S runs: Part A (no split) and Part B (split).

Usage: python summarize.py OUT_DIR [--max-realizations N] RUN [RUN ...]
Each RUN is an attempt directory written by run_experiment.py (CSV files in
RUN/outputs/) or a frozen result directory such as
results/colored-70s/formal_rho06 (CSV files directly in RUN/).  The runs are
pooled (realizations keep their seeds), realizations with index < N are kept
when N is given, and summary_part_a.csv, summary_part_b.csv and summary.md are
written into OUT_DIR.  Uncertainty is over noise realizations (paired t
intervals).
"""

from __future__ import annotations

import collections
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

PART_A = ("white_2sdr", "white_pca", "white_mpca", "raw_2sdr", "raw_pca",
          "raw_mpca", "estpsd_2sdr", "mean_only")
PART_B_FAMILIES = ("white_2sdr", "raw_2sdr", "white_mpca", "raw_mpca",
                   "white_pca", "raw_pca", "white_pcasplit", "raw_pcasplit",
                   "mean_only")


def ci(values) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    if len(x) < 2:
        return mean, mean, mean
    half = float(student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return mean, mean - half, mean + half


def fmt(triple, scale=1e4) -> str:
    m, lo, hi = (v * scale for v in triple)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("attempts", type=Path, nargs="+")
    parser.add_argument("--max-realizations", type=int)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    limit = args.max_realizations

    def load(name):
        rows = []
        for attempt in args.attempts:
            path = attempt / "outputs" / name
            if not path.exists():
                path = attempt / name
            if path.exists():
                rows += [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
                         if limit is None or int(r["realization"]) < limit]
        return rows

    md = [f"# Summary of {', '.join(f'`{a}`' for a in args.attempts)}",
          f"Realizations per condition: first {limit}" if limit else "All realizations", "", "Values are per-pixel MSE x 1e4; "
          "brackets are 95% paired t intervals over noise realizations.", ""]

    rows_a = load("part_a_pixel_mse.csv")
    sel_a = load("part_a_selections.csv")
    summary_a = []
    md += ["## Part A: no split, all particles", ""]
    for (rho, s0) in sorted({(r["rho"], r["white_reference_snr"]) for r in rows_a}):
        rs = [r for r in rows_a if r["rho"] == rho and r["white_reference_snr"] == s0]
        md.append(f"### rho {rho}, white-reference SNR {s0} ({len(rs)} realizations)")
        md.append("")
        md.append("| method | MSE | white 2SDR minus method |")
        md.append("|---|---|---|")
        for method in PART_A:
            values = [float(r[method]) for r in rs]
            diff = [float(r["white_2sdr"]) - float(r[method]) for r in rs]
            summary_a.append({"rho": rho, "s0": s0, "method": method,
                              "n": len(rs), "mse": ci(values)[0],
                              "mse_lo": ci(values)[1], "mse_hi": ci(values)[2],
                              "diff_white2sdr_minus": ci(diff)[0],
                              "diff_lo": ci(diff)[1], "diff_hi": ci(diff)[2]})
            md.append(f"| {method} | {fmt(ci(values))} | "
                      f"{'—' if method == 'white_2sdr' else fmt(ci(diff))} |")
        ss = [r for r in sel_a if r["rho"] == rho and r["white_reference_snr"] == s0]
        for variant in ("white", "estpsd", "raw"):
            v = [r for r in ss if r["variant"] == variant]
            triples = collections.Counter((r["p"], r["q"], r["r"]) for r in v)
            bound = sum(r["stage1_at_upper"] == "True" or r["r_at_bound"] == "True"
                        for r in v)
            pca_bound = sum(r["pca_at_bound"] == "True" for r in v)
            md.append(f"- {variant}: (p,q,r) {dict(triples.most_common(4))}; "
                      f"2SDR at a bound {bound}/{len(v)}; PCA at bound {pca_bound}/{len(v)}")
        md.append("")

    summary_b = []
    rows_b = load("part_b_candidates.csv")
    if rows_b:
        md += ["## Part B: corrected holdout score over complete estimators", ""]
        groups = collections.defaultdict(list)
        for r in rows_b:
            groups[(r["rho"], r["white_reference_snr"], r["realization"])].append(r)
        for (rho, s0) in sorted({(k[0], k[1]) for k in groups}):
            keys = [k for k in groups if k[0] == rho and k[1] == s0]
            picks, regret, pick_test = collections.Counter(), [], []
            family_test = collections.defaultdict(list)
            rel_bias = []
            for k in keys:
                cands = groups[k]
                pick = min(cands, key=lambda r: float(r["split_sure"]))
                oracle = min(cands, key=lambda r: float(r["test_true_risk"]))
                picks[(pick["family"], pick["p"], pick["q"], pick["r"])] += 1
                regret.append(float(pick["test_true_risk"]) - float(oracle["test_true_risk"]))
                pick_test.append(float(pick["test_true_risk"]))
                for fam in PART_B_FAMILIES:
                    fc = [r for r in cands if r["family"] == fam]
                    if fc:
                        best = min(fc, key=lambda r: float(r["split_sure"]))
                        family_test[fam].append(float(best["test_true_risk"]))
                rel_bias += [abs(float(r["split_sure"]) - float(r["score_true_risk"]))
                             / float(r["score_true_risk"]) for r in cands]
            md.append(f"### rho {rho}, white-reference SNR {s0} ({len(keys)} realizations)")
            md.append("")
            md.append(f"- overall pick (family,p,q,r): {dict(picks.most_common(5))}")
            md.append(f"- pick test MSE {fmt(ci(pick_test))}; regret to test oracle "
                      f"{fmt(ci(regret))}")
            md.append(f"- max relative |score - score-split risk| over all candidates "
                      f"{max(rel_bias):.3f}")
            md.append("")
            md.append("| family (score-selected within family) | test MSE | "
                      "pick minus family |")
            md.append("|---|---|---|")
            for fam, values in family_test.items():
                diff = [a - b for a, b in zip(pick_test, values)]
                md.append(f"| {fam} | {fmt(ci(values))} | {fmt(ci(diff))} |")
                summary_b.append({"rho": rho, "s0": s0, "family": fam, "n": len(values),
                                  "test_mse": ci(values)[0],
                                  "pick_minus_family": ci(diff)[0],
                                  "diff_lo": ci(diff)[1], "diff_hi": ci(diff)[2]})
            md.append("")

    for name, rows in (("summary_part_a.csv", summary_a), ("summary_part_b.csv", summary_b)):
        if rows:
            with open(out / name, "w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
