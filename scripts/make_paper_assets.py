#!/usr/bin/env python3
"""Regenerate the paper's tables and figures from the frozen outputs in results/.

Nothing is refitted.  Each asset is written to --output-dir (default
build/paper_assets) under the file name used by the manuscript:

    table3_same_sample_risk.tex         Table 3   (Section 6.3)
    paper1_70s_table.tex                Table 4   (Section 6.4)
    paper_70s_montage.png               Figure 3  (Section 6.4)
    supplement_projection_family_risk.pdf   Figure 2  (Section 6.2; redrawn from summary.csv)
    supplement_projection_family_table.tex  Table S1  (Supplement S4)
    paper1_colored_calibration.tex      Table S2  (Supplement S7.A)
    paper1_colored_70s.tex              Table 5   (Section 6.5)
    paper1_colored_70s_partA.tex        Table S3  (Supplement S7.C)

Tables 5, S2 and S3 are written by experiments/colored_70s/make_tables.py,
which this script calls.  With --check, every regenerated file is compared
byte for byte with the reference copy in results/paper-assets/.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def same_sample_table() -> str:
    """Table 3: conditional per-coordinate risk under same-sample selection."""
    rows = read_csv(RESULTS / "same-sample-selection/per_replicate.csv")
    methods = ("risk_two_sdr_sure_hhi", "risk_vector_pca_hhi", "risk_mpca_sure_full_core")
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{Conditional per-coordinate reconstruction risk under same-sample",
        r"selection.  Entries are means (standard deviations), multiplied by",
        r"$10^3$, over 100 training-sample replicates.}",
        r"\label{tab:algorithm-risk}",
        r"\begin{tabular}{rrlrrr}",
        r"\toprule",
        r"$n$ & SNR & Core spectrum & 2SDR & PCA & MPCA\\",
        r"\midrule",
    ]
    for n in ("625", "1000"):
        for snr in ("0.04", "0.06"):
            for spectrum in ("equal", "geometric"):
                cell = [r for r in rows if r["n"] == n and r["snr"] == snr
                        and r["spectrum"] == spectrum and float(r["eta"]) == 0.0]
                assert len(cell) == 100, (n, snr, spectrum, len(cell))
                values = []
                for method in methods:
                    risks = [1e3 * float(r[method]) for r in cell]
                    values.append(rf"${mean(risks):.3f}\;({stdev(risks):.3f})$")
                lines.append(f"{n:<4} & .{snr[-2:]} & {spectrum:<9} & " + " & ".join(values) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def projection_family_table() -> str:
    """Table S1: paired 2SDR minus PCA differences at the prespecified sizes."""
    scenarios = (
        ("interior_row_thin", "$(2,8,4)$"),
        ("interior_balanced", "$(4,4,4)$"),
        ("interior_col_thin", "$(8,2,4)$"),
    )
    rows = {
        (r["scenario"], r["snr"], r["sample_size"]): r
        for r in read_csv(RESULTS / "controlled-projection-family/paired_summary.csv")
        if r["comparison"] == "twosdr_minus_pca" and r["metric"] == "population_projection_risk"
    }
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"Geometry & SNR & $n=250$ & $n=1000$ \\", r"\midrule"]
    for index, (scenario, label) in enumerate(scenarios):
        if index:
            lines.append(r"\addlinespace")
        for snr in ("0.04", "0.05", "0.06"):
            cells = []
            for size in ("250", "1000"):
                r = rows[(scenario, snr, size)]
                cells.append(
                    f"${100 * float(r['mean']):.2f}$ "
                    f"$({100 * float(r['ci95_low']):.2f},\\,{100 * float(r['ci95_high']):.2f})$"
                )
            lines.append(f"{label} & {snr} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def seventy_s_table() -> str:
    """Table 4: same-sample selection on the synthetic 70S particles."""
    folder = RESULTS / "seventy-s-same-sample"
    selections = {row["snr"]: row for row in read_csv(folder / "selection.csv")}
    per_view = read_csv(folder / "per_view_metrics.csv")
    methods = ("two_sdr_tu_sure_hhi", "vector_pca_hhi", "mpca_tu_sure_full_core")
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{Same-sample selection and reconstruction in the",
        r"synthetic 70S application.  MSE entries are means (standard deviations)",
        r"across the 50 viewing directions, multiplied by $10^4$; each view",
        r"contains 100 particles.}",
        r"\label{tab:70s}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"SNR & Selected $(p,q,r)$ & PCA $r$ & 2SDR MSE & PCA MSE & MPCA MSE\\",
        r"\midrule",
    ]
    for snr in ("0.04", "0.05", "0.06"):
        selection = selections[snr]
        cells = []
        for method in methods:
            values = [10000 * float(row["mse"]) for row in per_view
                      if row["snr"] == snr and row["method"] == method]
            cells.append(rf"${mean(values):.4f}\;({stdev(values):.4f})$")
        lines.append(
            rf"${float(snr):.2f}$ & "
            rf"$({selection['p']},{selection['q']},{selection['r']})$ & "
            rf"${selection['pca_r']}$ & " + " & ".join(cells) + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def projection_family_figure(path: Path) -> None:
    """Figure 2: redraw the normalized population risk from the frozen summary."""
    sys.path.insert(0, str(ROOT / "experiments" / "controlled_projection_family"))
    import run_experiment as controlled  # noqa: E402  (needs its sibling module core.py)

    rows = []
    for row in read_csv(RESULTS / "controlled-projection-family/summary.csv"):
        row = dict(row)
        for key in ("mean", "ci95_low", "ci95_high"):
            row[key] = float(row[key])
        rows.append(row)
    sizes = tuple(sorted({int(r["sample_size"]) for r in rows}))
    controlled.plot_population_risk(rows, sizes, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the paper's tables and figures from results/.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/paper_assets")
    parser.add_argument("--check", action="store_true",
                        help="compare every asset with results/paper-assets/")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    (out / "table3_same_sample_risk.tex").write_text(same_sample_table(), encoding="utf-8")
    (out / "supplement_projection_family_table.tex").write_text(projection_family_table(), encoding="utf-8")
    (out / "paper1_70s_table.tex").write_text(seventy_s_table(), encoding="utf-8")
    shutil.copyfile(RESULTS / "seventy-s-same-sample/paper_montage.png", out / "paper_70s_montage.png")
    projection_family_figure(out / "supplement_projection_family_risk.pdf")
    subprocess.run([sys.executable, str(ROOT / "experiments/colored_70s/make_tables.py"),
                    "--results-dir", str(RESULTS), "--output-dir", str(out)], check=True)

    if not args.check:
        print(f"wrote paper assets to {out}")
        return 0
    reference = RESULTS / "paper-assets"
    failures = [p.name for p in sorted(reference.iterdir())
                if p.name != "result_manifest.json"
                and (not (out / p.name).exists()
                     or not filecmp.cmp(p, out / p.name, shallow=False))]
    for name in failures:
        print(f"DIFFERS: {name}")
    print("all paper assets match" if not failures else f"{len(failures)} asset(s) differ")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
