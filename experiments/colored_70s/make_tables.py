#!/usr/bin/env python3
"""Write the Section 6.5 and supplement tables from frozen run outputs.

Reads ``RESULTS_DIR/whitened-calibration/rho{03,06,08}/candidate_scores.csv``
and ``RESULTS_DIR/colored-70s/<run>/{part_a_pixel_mse,part_a_selections,
part_b_candidates}.csv`` and writes into ``OUTPUT_DIR``:

  paper1_colored_calibration.tex   calibration of the corrected holdout score
  paper1_colored_70s.tex           complete-estimator selection (Part B)
  paper1_colored_70s_partA.tex     same-sample whitened pipeline (Part A)

Reported design: rho = 0.6 first 10 realizations per s0; rho = 0, 0.3 and
0.8 five each.
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics as st
from pathlib import Path

from scipy.stats import t as student_t

RUNS = {
    "0.0": ["anchor_rho00"],
    "0.6": ["formal_rho06", "formal_rho06_s006"],
    "0.3": ["sensitivity_rho03"],
    "0.8": ["sensitivity_rho08"],
}
LIMIT = {"0.0": 5, "0.6": 10, "0.3": 5, "0.8": 5}


def ci(values):
    x = list(values)
    m = st.mean(x)
    if len(x) < 2:
        return m, m, m
    h = student_t.ppf(0.975, len(x) - 1) * st.stdev(x) / math.sqrt(len(x))
    return m, m - h, m + h


def load(results_dir: Path, name: str) -> list[dict[str, str]]:
    """Rows of ``name`` from the reported runs, filtered by rho and realization."""
    rows = []
    for rho, runs in RUNS.items():
        for run in runs:
            path = Path(results_dir) / "colored-70s" / run / name
            with path.open(newline="", encoding="utf-8") as handle:
                rows += [r for r in csv.DictReader(handle)
                         if r["rho"] == rho and int(r["realization"]) < LIMIT[rho]]
    return rows


def calibration_table(results_dir: Path) -> str:
    lines = [r"\begin{table}[tbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Accuracy of the corrected holdout score over 100 replicates per "
             r"$\rho$.  The 73 candidates comprise 36 whitened MPCA fits, 36 raw MPCA fits, "
             r"and the mean-only estimator.  The columns report the largest absolute mean score "
             r"bias, the number of candidates whose 95\% confidence intervals for mean bias "
             r"contain zero, the frequency of selecting the analytic-MSE oracle, the mean excess "
             r"MSE of the estimator selected by the uncorrected holdout residual, and the mean "
             r"MSEs of the whitened $(4,4)$ and best raw candidates.  MSEs are per coordinate.}",
             r"\label{tab:colored-calibration}",
             r"\begin{tabular}{rcccccc}", r"\toprule",
             r" & largest & intervals & oracle & uncorrected & whitened $(4,4)$ & best raw\\",
             r"$\rho$ & $|$bias$|$ & with 0 & selected & excess MSE & MSE & MSE\\",
             r"\midrule"]
    calib = Path(results_dir) / "whitened-calibration"
    for tag in ("03", "06", "08"):
        with (calib / f"rho{tag}" / "candidate_scores.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        by = collections.defaultdict(list)
        for r in rows:
            by[r["candidate"]].append(float(r["corrected_split"]) - float(r["oracle_risk"]))
        bias = max(abs(st.mean(v)) for v in by.values())
        # The two (32,32) candidates are the identity map with zero bias; their
        # floating-point residue (~1e-17) counts as covering zero.
        cover = sum(abs(st.mean(v)) < 1e-12 or ci(v)[1] <= 0 <= ci(v)[2]
                    for v in by.values())
        reps = collections.defaultdict(list)
        for r in rows:
            reps[r["replicate"]].append(r)
        hit, excess = 0, []
        for cands in reps.values():
            oracle = min(cands, key=lambda r: float(r["oracle_risk"]))
            pick = min(cands, key=lambda r: float(r["corrected_split"]))
            hit += pick is oracle
            un = min(cands, key=lambda r: float(r["uncorrected_holdout"]))
            excess.append(float(un["oracle_risk"]) - float(oracle["oracle_risk"]))
        risk = collections.defaultdict(list)
        for r in rows:
            risk[r["candidate"]].append(float(r["oracle_risk"]))
        raw_best = min(st.mean(v) for k, v in risk.items() if k.startswith("raw_"))
        lines.append(f"{int(tag) / 10:.1f} & ${bias * 1e4:.1f}\\times10^{{-4}}$ & "
                     f"{cover}/{len(by)} & {hit}/{len(reps)} & {st.mean(excess):.2f} & "
                     f"{st.mean(risk['white_p4_q4']):.3f} & {raw_best:.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def same_estimator(a, b) -> bool:
    """Whether two candidates differ only in raw versus whitened coordinates."""
    strip = lambda name: name.removeprefix("raw_").removeprefix("white_")
    return strip(a["candidate"]) == strip(b["candidate"])


def part_b_table(results_dir: Path) -> str:
    rows = load(results_dir, "part_b_candidates.csv")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["rho"], r["white_reference_snr"], r["realization"])].append(r)
    lines = [r"\begin{table}[tbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
             r"\caption{Complete-estimator selection by the corrected holdout score on "
             r"the 70S particles under white and colored noise, over 10 noise realizations "
             r"at $\rho=0.6$ and 5 otherwise.  The corrected holdout score selected 2SDR in "
             r"all 75 realizations.  The coordinate choices indicated in the group headings "
             r"were consistent across all realizations within each noise condition.  The "
             r"second and third columns give the most "
             r"frequent selected $(p,q,r)$ and its frequency, with equally frequent triples "
             r"listed together; the fourth gives the number of realizations in which the "
             r"corrected holdout score selects the candidate with the lowest empirical test "
             r"MSE among all considered candidates.  The last three "
             r"columns give mean test MSE $\times10^{4}$ for the selected 2SDR estimator and "
             r"for the PCA and full-core MPCA candidates with the best score.  Under white "
             r"noise whitening is a uniform rescaling, so raw and whitened fits coincide "
             r"and are counted as one candidate.}",
             r"\label{tab:colored-70s}",
             r"\begin{tabular}{rllrrrr}", r"\toprule",
             r" & & & selected test-MSE & \multicolumn{3}{c}{test MSE $\times10^{4}$}\\",
             r"\cmidrule(lr){5-7}",
             r"$s_0$ & selected $(p,q,r)$ & frequency & minimizer (count) & 2SDR & PCA & MPCA\\",
             r"\midrule"]
    for rho in ("0.0", "0.3", "0.6", "0.8"):
        block = []
        families = collections.Counter()
        for s0 in ("0.04", "0.05", "0.06"):
            keys = [k for k in groups if k[0] == rho and k[1] == s0]
            fam = collections.Counter()
            triples = collections.Counter()
            hits = 0
            pick_test, pca_test, mpca_test = [], [], []
            for k in keys:
                cands = groups[k]
                pick = min(cands, key=lambda r: float(r["split_sure"]))
                oracle = min(cands, key=lambda r: float(r["test_true_risk"]))
                # At rho = 0 whitening is a uniform rescaling, so raw and
                # whitened fits coincide and count as one candidate; report
                # them as 2SDR.
                if rho == "0.0":
                    hits += same_estimator(pick, oracle)
                else:
                    hits += pick is oracle
                fam["2sdr" if rho == "0.0" else pick["family"]] += 1
                triples[(int(pick["p"]), int(pick["q"]), int(pick["r"]))] += 1
                pick_test.append(float(pick["test_true_risk"]))
                pcas = [r for r in cands if "pca" in r["family"] and "mpca" not in r["family"]]
                pca_test.append(float(min(pcas, key=lambda r: float(r["split_sure"]))["test_true_risk"]))
                mp = [r for r in cands if "mpca" in r["family"]]
                mpca_test.append(float(min(mp, key=lambda r: float(r["split_sure"]))["test_true_risk"]))
            (family, count), = fam.most_common(1)
            label = family.replace("_2sdr", "").replace("white", "whitened").replace("2sdr", "")
            mode_n = max(triples.values())
            modes = [m for m, n in triples.items() if n == mode_n]
            shown = "/".join(f"$({m[0]},{m[1]},{m[2]})$" for m in modes)
            # Every realization selects the same estimator family within a rho
            # block, so the family is reported once in the block heading.
            assert count == len(keys), (rho, s0, fam)
            families[label] += 1
            block.append(f".{s0[-2:]} & {shown} & ({mode_n}/{len(keys)}) & {hits}/{len(keys)} & "
                         f"{st.mean(pick_test) * 1e4:.3f} & {st.mean(pca_test) * 1e4:.3f} & "
                         f"{st.mean(mpca_test) * 1e4:.3f}\\\\")
        (label,), = [tuple(families)]
        heading = ("White noise ($\\rho=0$)" if rho == "0.0"
                   else f"Colored noise ($\\rho={rho}$, {label} selected)")
        if rho != "0.0":
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\itshape {heading}}}\\")
        lines += block
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def part_a_table(results_dir: Path) -> str:
    rows = load(results_dir, "part_a_pixel_mse.csv")
    sels = load(results_dir, "part_a_selections.csv")
    methods = [("white_2sdr", "2SDR"), ("white_pca", "PCA"), ("white_mpca", "MPCA"),
               ("raw_2sdr", "2SDR"), ("raw_pca", "PCA"), ("estpsd_2sdr", "2SDR")]
    lines = [r"\begin{table}[tbp]", r"\centering", r"\small",
             r"\caption{Same-sample reconstruction of synthetic 70S particles under white and "
             r"colored noise.  Panel (a) gives mean per-pixel MSE, multiplied by $10^4$, over the "
             r"noise realizations; the mean-only estimator has MSE between 3.42 and 3.45 in the "
             r"same units across all conditions.  Panel (b) reports the most frequent selected "
             r"$(p,q,r)$ and its frequency; a dagger indicates that no unique modal triple is "
             r"reported, in which case the ranges of the three dimensions are shown.  The "
             r"$\rho=0$ results use the five additional white-noise realizations of the "
             r"complete-estimator study (Table~5 of the main text).}",
             r"\label{tab:supp-colored-partA}",
             r"\textit{(a) Reconstruction MSE $\times10^{4}$}\\[2pt]",
             r"\begin{tabular}{rrrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{whitened} & \multicolumn{2}{c}{raw} & estimated-spectrum\\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
             r"$\rho$ & $s_0$ & " + " & ".join(n for _, n in methods) + r"\\", r"\midrule"]
    panel_b = [r"\bigskip", "",
               r"\textit{(b) Selected dimensions}\\[2pt]",
               r"\begin{tabular}{rrll}", r"\toprule",
               r"$\rho$ & $s_0$ & whitened $(p,q,r)$ & raw $(p,q,r)$\\", r"\midrule"]
    for rho in ("0.0", "0.3", "0.6", "0.8"):
        for s0 in ("0.04", "0.05", "0.06"):
            rs = [r for r in rows if r["rho"] == rho and r["white_reference_snr"] == s0]
            ss = [r for r in sels if r["rho"] == rho and r["white_reference_snr"] == s0]
            vals = " & ".join(f"{st.mean(float(r[m]) for r in rs) * 1e4:.3f}" for m, _ in methods)
            def mode(v):
                # The unique most frequent triple when it occurs more than once;
                # otherwise the range of each selected dimension.
                rows_v = [r for r in ss if r["variant"] == v]
                counts = collections.Counter((r["p"], r["q"], r["r"]) for r in rows_v).most_common()
                if counts[0][1] > 1 and (len(counts) == 1 or counts[1][1] < counts[0][1]):
                    return "$(" + ",".join(counts[0][0]) + ")$, " + f"{counts[0][1]}/{len(rows_v)}"
                def span(key):
                    lo, hi = min(int(r[key]) for r in rows_v), max(int(r[key]) for r in rows_v)
                    return str(lo) if lo == hi else f"{lo}\\text{{--}}{hi}"
                return "$(" + ",".join(span(k) for k in ("p", "q", "r")) + ")^\\dagger$"
            w, a = mode("white"), mode("raw")
            lines.append(f"{rho} & .{s0[-2:]} & {vals}\\\\")
            panel_b.append(f"{rho} & .{s0[-2:]} & {w} & {a}\\\\")
        if rho != "0.8":
            lines.append(r"\addlinespace")
            panel_b.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"] + panel_b
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_tables(results_dir: Path, output_dir: Path) -> dict[str, str]:
    """Write the three tables into ``output_dir`` and return their text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"paper1_colored_calibration.tex": calibration_table(results_dir),
              "paper1_colored_70s.tex": part_b_table(results_dir),
              "paper1_colored_70s_partA.tex": part_a_table(results_dir)}
    for name, text in tables.items():
        (output_dir / name).write_text(text, encoding="utf-8")
    return tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="root of the frozen results (contains whitened-calibration/ "
                             "and colored-70s/)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name, text in write_tables(args.results_dir, args.output_dir).items():
        print(f"== {name}\n{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
