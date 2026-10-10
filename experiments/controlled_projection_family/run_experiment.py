#!/usr/bin/env python3
"""Run the controlled PCA--2SDR--MPCA projection-family experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

import core


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PROTOCOL_PATH = HERE / "protocol.json"
SAMPLE_SIZES = (250, 1000, 5000)
SNR_VALUES = (0.04, 0.05, 0.06)
INTERIOR_SCENARIOS = (
    "interior_row_thin", "interior_balanced", "interior_col_thin",
)
METHODS = ("pca", "twosdr", "mpca")
METRICS = (
    "projector_frobenius_error",
    "clean_signal_capture",
    "retained_noise_energy",
    "retained_noise_fraction",
    "population_projection_risk",
    "heldout_projection_risk",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(
            value, indent=2, sort_keys=True,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        ) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(arguments: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger(f"controlled_projection_family.{output}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(output / "run.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def source_hashes() -> dict[str, str]:
    return {
        path.name: sha256(path)
        for path in (PROTOCOL_PATH, HERE / "README.md", HERE / "core.py", Path(__file__))
    }


def summarize_values(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    count = len(array)
    mean = float(array.mean())
    if count > 1:
        sd = float(array.std(ddof=1))
        se = sd / np.sqrt(count)
        critical = float(student_t.ppf(0.975, count - 1))
        low, high = mean - critical * se, mean + critical * se
    else:
        sd = se = 0.0
        low = high = mean
    return {
        "count": count, "mean": mean, "sample_sd": sd,
        "standard_error": se, "ci95_low": low, "ci95_high": high,
    }


def summarize_method_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario"], row["role"], row["p"], row["q"], row["r"],
            row["snr"], row["sample_size"], row["method"],
        )
        groups[key].append(row)
    result = []
    for key, members in sorted(groups.items()):
        scenario, role, p, q, r, snr, sample_size, method = key
        for metric in METRICS:
            result.append({
                "scenario": scenario, "role": role, "p": p, "q": q, "r": r,
                "snr": snr, "sample_size": sample_size, "method": method,
                "metric": metric,
                **summarize_values([float(row[metric]) for row in members]),
            })
    return result


def paired_rows(method_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in method_rows:
        key = (row["scenario"], row["snr"], row["sample_size"], row["replicate"])
        grouped[key][row["method"]] = row
    result = []
    for key, methods in sorted(grouped.items()):
        if set(methods) != set(METHODS):
            raise RuntimeError(f"method universe mismatch for {key}: {sorted(methods)}")
        scenario, snr, sample_size, replicate = key
        for comparator in ("pca", "mpca"):
            row = {
                "scenario": scenario, "snr": snr, "sample_size": sample_size,
                "replicate": replicate, "comparison": f"twosdr_minus_{comparator}",
            }
            for metric in METRICS:
                row[f"{metric}_delta"] = (
                    float(methods["twosdr"][metric]) - float(methods[comparator][metric])
                )
            result.append(row)
    return result


def summarize_paired(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["snr"], row["sample_size"], row["comparison"])].append(row)
    result = []
    for key, members in sorted(groups.items()):
        scenario, snr, sample_size, comparison = key
        for metric in METRICS:
            result.append({
                "scenario": scenario, "snr": snr, "sample_size": sample_size,
                "comparison": comparison, "metric": metric,
                **summarize_values([float(row[f"{metric}_delta"]) for row in members]),
            })
    return result


def boundary_summary(rows: list[dict]) -> tuple[list[dict], dict]:
    result = []
    definitions = (
        ("pca_boundary", "pca", "twosdr"),
        ("mpca_boundary", "twosdr", "mpca"),
    )
    for scenario, first, second in definitions:
        selected = [row for row in rows if row["scenario"] == scenario]
        result.append({
            "scenario": scenario,
            "comparison": f"{first}_versus_{second}",
            "replicates": len(selected),
            "maximum_projector_difference": max(float(row["projector_difference"]) for row in selected),
            "maximum_population_risk_difference": max(float(row["population_risk_difference"]) for row in selected),
            "projector_tolerance": 1e-10,
            "risk_tolerance": 1e-12,
            "criterion_met": all(
                float(row["projector_difference"]) <= 1e-10
                and float(row["population_risk_difference"]) <= 1e-12
                for row in selected
            ),
        })
    decision = {
        "all_boundary_identities_met": all(row["criterion_met"] for row in result),
        "comparisons": result,
    }
    return result, decision


def decision_summary(
    paired_summary: list[dict], boundary_decision: dict, *,
    sample_sizes: tuple[int, ...], snr_values: tuple[float, ...],
    formal_full_design: bool,
) -> dict:
    primary = [
        row for row in paired_summary
        if row["scenario"] in INTERIOR_SCENARIOS
        and row["sample_size"] in (250, 1000)
        and row["comparison"] == "twosdr_minus_pca"
        and row["metric"] == "population_projection_risk"
    ]
    mpca = [
        row for row in paired_summary
        if row["scenario"] in INTERIOR_SCENARIOS
        and row["comparison"] == "twosdr_minus_mpca"
        and row["metric"] == "population_projection_risk"
    ]
    primary_sample_sizes = tuple(n for n in sample_sizes if n in (250, 1000))
    expected_primary = (
        len(INTERIOR_SCENARIOS) * len(snr_values) * len(primary_sample_sizes)
    )
    expected_mpca = len(INTERIOR_SCENARIOS) * len(snr_values) * len(sample_sizes)
    if len(primary) != expected_primary or len(mpca) != expected_mpca:
        raise RuntimeError(
            f"decision universe mismatch: primary={len(primary)} mpca={len(mpca)}"
        )
    return {
        "claim_boundary": (
            "2SDR is evaluated as a structured subfamily of unrestricted PCA; "
            "a favorable interior result is finite-sample estimation evidence, "
            "not a larger representation-family claim."
        ),
        "boundary_identities_met": boundary_decision["all_boundary_identities_met"],
        "formal_full_design_evaluated": formal_full_design,
        "structured_efficiency_comparisons": len(primary),
        "structured_efficiency_strict_comparisons": sum(row["ci95_high"] < 0.0 for row in primary),
        "structured_efficiency_criterion_met": (
            all(row["ci95_high"] < 0.0 for row in primary)
            if formal_full_design else None
        ),
        "core_reduction_comparisons": len(mpca),
        "core_reduction_strict_comparisons": sum(row["ci95_high"] < 0.0 for row in mpca),
        "core_reduction_criterion_met": (
            all(row["ci95_high"] < 0.0 for row in mpca)
            if formal_full_design else None
        ),
        "criterion_interpretation": (
            "Each strict comparison requires the upper endpoint of the paired "
            "Student-t 95% Monte Carlo interval for the 2SDR-minus-comparator "
            "population-risk difference to be below zero."
        ),
    }


def summary_lookup(summary_rows: list[dict]) -> dict[tuple, dict]:
    return {
        (row["scenario"], float(row["snr"]), int(row["sample_size"]), row["method"], row["metric"]): row
        for row in summary_rows
    }


SCENARIO_LABELS = {
    "interior_row_thin": "smaller row rank (2,8,4)",
    "interior_balanced": "equal row and column ranks (4,4,4)",
    "interior_col_thin": "smaller column rank (8,2,4)",
}
METHOD_COLORS = {"pca": "#4C78A8", "twosdr": "#E45756", "mpca": "#59A14F"}
METHOD_LABELS = {"pca": "PCA", "twosdr": "2SDR", "mpca": "MPCA"}
# Fixed PDF metadata so that a figure redrawn from the same numbers is byte-identical.
PDF_METADATA = {"CreationDate": None, "Creator": None, "Producer": None}


def plot_population_risk(
    summary_rows: list[dict], sample_sizes: tuple[int, ...], risk_pdf: Path,
    risk_png: Path | None = None,
) -> None:
    """Figure 2 of the paper: normalized population MSE with 95% intervals."""
    lookup = summary_lookup(summary_rows)
    labels, colors, method_labels = SCENARIO_LABELS, METHOD_COLORS, METHOD_LABELS
    n_snr = len(SNR_VALUES)
    fig, axes = plt.subplots(
        n_snr, 3, figsize=(11.2, 2.75 * n_snr + 0.8), sharex=True,
        squeeze=False,
    )
    for row_index, snr in enumerate(SNR_VALUES):
        for col_index, scenario in enumerate(INTERIOR_SCENARIOS):
            ax = axes[row_index, col_index]
            for method in METHODS:
                entries = [
                    lookup[(scenario, snr, n, method, "population_projection_risk")]
                    for n in sample_sizes
                ]
                means = np.array([entry["mean"] for entry in entries])
                low = np.array([entry["ci95_low"] for entry in entries])
                high = np.array([entry["ci95_high"] for entry in entries])
                ax.errorbar(
                    sample_sizes, means, yerr=(means - low, high - means), marker="o",
                    linewidth=1.6, capsize=2.5, color=colors[method], label=method_labels[method],
                )
            ax.set_xscale("log")
            ax.set_xlim(200, 6500)
            ax.set_xticks(list(sample_sizes))
            ax.set_xticklabels([str(n) for n in sample_sizes])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.grid(alpha=0.25)
            ax.set_title(f"{labels[scenario]}\nSNR {snr:.2f}")
            if col_index == 0:
                ax.set_ylabel("Normalized population MSE")
            if row_index == n_snr - 1:
                ax.set_xlabel("Fitting sample size")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    if risk_png is not None:
        fig.savefig(risk_png, dpi=220, bbox_inches="tight")
    fig.savefig(risk_pdf, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)


def make_figures(
    output: Path, summary_rows: list[dict], sample_sizes: tuple[int, ...],
) -> list[Path]:
    figure_dir = output / "figures"
    figure_dir.mkdir()
    lookup = summary_lookup(summary_rows)
    labels, colors, method_labels = SCENARIO_LABELS, METHOD_COLORS, METHOD_LABELS

    risk_png = figure_dir / "population_projection_risk.png"
    risk_pdf = figure_dir / "population_projection_risk.pdf"
    plot_population_risk(summary_rows, sample_sizes, risk_pdf, risk_png)

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.3), sharex=True)
    line_styles = ("--", "-", ":", "-.")
    styles = {snr: line_styles[index] for index, snr in enumerate(SNR_VALUES)}
    for col_index, scenario in enumerate(INTERIOR_SCENARIOS):
        for row_index, metric in enumerate(("projector_frobenius_error", "clean_signal_capture")):
            ax = axes[row_index, col_index]
            for method in METHODS:
                for snr in SNR_VALUES:
                    entries = [lookup[(scenario, snr, n, method, metric)] for n in sample_sizes]
                    ax.plot(
                        sample_sizes, [entry["mean"] for entry in entries], marker="o",
                        linewidth=1.4, linestyle=styles[snr], color=colors[method],
                        label=f"{method_labels[method]}, SNR {snr:.2f}",
                    )
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(labels[scenario] if row_index == 0 else "")
            if col_index == 0:
                ax.set_ylabel("Projector error" if row_index == 0 else "Clean-signal capture")
            if row_index == 1:
                ax.set_xlabel("Fitting sample size")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3, frameon=False,
        fontsize=8, bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    mechanism_png = figure_dir / "projector_and_signal_capture.png"
    mechanism_pdf = figure_dir / "projector_and_signal_capture.pdf"
    fig.savefig(mechanism_png, dpi=220, bbox_inches="tight")
    fig.savefig(mechanism_pdf, bbox_inches="tight")
    plt.close(fig)
    return [risk_png, risk_pdf, mechanism_png, mechanism_pdf]


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def make_tex_table(
    output: Path, summary_rows: list[dict], paired_summary: list[dict],
    sample_sizes: tuple[int, ...],
) -> Path:
    table_dir = output / "tables"
    table_dir.mkdir()
    lookup = summary_lookup(summary_rows)
    paired_lookup = {
        (row["scenario"], float(row["snr"]), int(row["sample_size"]), row["comparison"], row["metric"]): row
        for row in paired_summary
    }
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Envelope & SNR & $n=250$ & $n=1000$ & $n=5000$ \\",
        r" & & \multicolumn{3}{c}{2SDR--PCA population-risk $\Delta$ (95\% CI)} \\",
        r"\midrule",
    ]
    short = {
        "interior_row_thin": "$(2,8,4)$",
        "interior_balanced": "$(4,4,4)$",
        "interior_col_thin": "$(8,2,4)$",
    }
    for scenario in INTERIOR_SCENARIOS:
        for snr in SNR_VALUES:
            cells = []
            for sample_size in sample_sizes:
                delta = paired_lookup[(
                    scenario, snr, sample_size, "twosdr_minus_pca", "population_projection_risk"
                )]
                cells.append(
                    f"{delta['mean']:+.4f} "
                    f"({delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f})"
                )
            lines.append(
                f"{short[scenario]} & {snr:.2f} & " + " & ".join(cells) + " " + r"\\"
            )
        lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path = table_dir / "controlled_projection_family.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(arguments) -> Path:
    replicates = 2 if arguments.smoke else arguments.replicates
    sample_sizes = (250,) if arguments.smoke else SAMPLE_SIZES
    output = arguments.output_dir
    if output is None:
        suffix = (
            "snr004_006_smoke" if arguments.smoke
            else f"snr004_006_seed{arguments.seed}_r{replicates}"
        )
        output = REPO_ROOT / "runs" / f"controlled_projection_family_{suffix}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logger = make_logger(output)
    started = utc_now()
    shutil.copy2(PROTOCOL_PATH, output / "protocol.json")
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    status = git_value(["status", "--porcelain"])
    provenance = {
        "created_at_utc": started,
        "exact_command": command,
        "git_revision": git_value(["rev-parse", "HEAD"]),
        "git_dirty_at_start": bool(status) if status is not None else None,
        "source_sha256": source_hashes(),
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    write_json(output / "provenance.json", provenance)
    configuration = {
        "analysis_id": "controlled_projection_family_v2",
        "smoke": arguments.smoke,
        "sample_sizes": list(sample_sizes),
        "snr_values": list(SNR_VALUES),
        "replicates_per_combination": replicates,
        "heldout_sample_size": core.HELDOUT_N,
        "model_seed": arguments.model_seed,
        "master_seed": arguments.seed,
        "scenario_specs": [list(spec) for spec in core.SCENARIO_SPECS],
        "mpca_max_iter": core.MAX_ITER,
        "mpca_projector_tolerance": core.TOL,
    }
    write_json(output / "config.json", configuration)
    logger.info("config=%s", json.dumps(configuration, sort_keys=True))
    try:
        scenarios = core.build_scenarios(arguments.model_seed)
        method_rows: list[dict] = []
        raw_boundaries: list[dict] = []
        seeds: list[dict] = []
        total = len(scenarios) * len(SNR_VALUES) * len(sample_sizes) * replicates
        completed = 0
        for scenario_index, scenario in enumerate(scenarios.values()):
            for snr in SNR_VALUES:
                for sample_size in sample_sizes:
                    for replicate in range(replicates):
                        seed = core.replicate_seed(
                            arguments.seed, scenario_index, snr, sample_size, replicate
                        )
                        rng = np.random.default_rng(seed)
                        scatter = core.training_scatter(scenario, snr, sample_size, rng)
                        signal, noise = core.heldout_draw(scenario, snr, rng)
                        fits = core.fit_methods(scatter, scenario)
                        metrics = {
                            method: core.method_metrics(fit, scenario, snr, signal, noise)
                            for method, fit in fits.items()
                        }
                        common = {
                            "scenario": scenario.scenario, "role": scenario.role,
                            "p": scenario.p, "q": scenario.q, "r": scenario.r,
                            "snr": snr, "sample_size": sample_size,
                            "replicate": replicate, "replicate_seed": seed,
                        }
                        for method in METHODS:
                            method_rows.append({**common, **metrics[method]})
                        if scenario.scenario == "pca_boundary":
                            first, second = "pca", "twosdr"
                        elif scenario.scenario == "mpca_boundary":
                            first, second = "twosdr", "mpca"
                        else:
                            first = second = None
                        if first is not None:
                            projector_delta, risk_delta = core.boundary_residual(
                                fits[first], fits[second],
                                metrics[first]["population_projection_risk"],
                                metrics[second]["population_projection_risk"],
                            )
                            raw_boundaries.append({
                                **common, "comparison": f"{first}_versus_{second}",
                                "projector_difference": projector_delta,
                                "population_risk_difference": risk_delta,
                            })
                        seeds.append({**common})
                        completed += 1
                        logger.info(
                            "replicate=%d/%d scenario=%s snr=%.2f n=%d index=%d "
                            "iterations=%d converged=%s",
                            completed, total, scenario.scenario, snr, sample_size, replicate,
                            fits["twosdr"].iterations, fits["twosdr"].converged,
                        )
        write_csv(output / "raw_metrics.csv", method_rows)
        pair_rows = paired_rows(method_rows)
        write_csv(output / "paired_differences.csv", pair_rows)
        method_summary = summarize_method_rows(method_rows)
        pair_summary = summarize_paired(pair_rows)
        write_csv(output / "summary.csv", method_summary)
        write_csv(output / "paired_summary.csv", pair_summary)
        write_csv(output / "boundary_checks_raw.csv", raw_boundaries)
        boundary_rows, boundary_decision = boundary_summary(raw_boundaries)
        write_csv(output / "boundary_checks.csv", boundary_rows)
        write_json(output / "boundary_checks.json", boundary_decision)
        decision = decision_summary(
            pair_summary, boundary_decision, sample_sizes=sample_sizes,
            snr_values=SNR_VALUES, formal_full_design=not arguments.smoke,
        )
        twosdr_rows = [row for row in method_rows if row["method"] == "twosdr"]
        decision["twosdr_fit_count"] = len(twosdr_rows)
        decision["twosdr_nonconverged_count"] = sum(
            not bool(row["converged"]) for row in twosdr_rows
        )
        decision["all_twosdr_fits_converged"] = (
            decision["twosdr_nonconverged_count"] == 0
        )
        write_json(output / "decision.json", decision)
        write_json(output / "seed_manifest.json", {
            "master_seed": arguments.seed,
            "model_seed": arguments.model_seed,
            "derivation": (
                "SeedSequence([master_seed, scenario_index, "
                "round(snr * 1000000), sample_size, replicate])"
            ),
            "replicates": seeds,
        })
        figures = make_figures(output, method_summary, sample_sizes)
        tex_table = make_tex_table(output, method_summary, pair_summary, sample_sizes)
        summary = {
            "analysis_id": "controlled_projection_family_v2",
            "raw_method_rows": len(method_rows),
            "paired_rows": len(pair_rows),
            "method_summary_rows": len(method_summary),
            "paired_summary_rows": len(pair_summary),
            "boundary_checks": boundary_decision,
            "decision": decision,
            "claim_boundary": (
                "The 2SDR family is a structured subfamily whose full-mode boundary "
                "recovers unrestricted PCA; favorable interior results concern "
                "finite-sample structured estimation."
            ),
        }
        write_json(output / "summary.json", summary)
        logger.info(
            "completed output=%s decision=%s", output,
            json.dumps(
                decision, sort_keys=True,
                default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
            ),
        )
        for handler in logger.handlers:
            handler.flush()
        artifact_paths = [
            output / name for name in (
                "protocol.json", "config.json", "provenance.json", "seed_manifest.json",
                "run.log", "raw_metrics.csv", "paired_differences.csv", "summary.csv",
                "paired_summary.csv", "boundary_checks_raw.csv", "boundary_checks.csv",
                "boundary_checks.json", "decision.json", "summary.json",
            )
        ] + figures + [tex_table]
        marker = {
            "status": "success",
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "output_dir": str(output),
            "raw_method_rows": len(method_rows),
            "paired_rows": len(pair_rows),
            "decision": decision,
            "artifacts_sha256": {
                str(path.relative_to(output)): sha256(path) for path in artifact_paths
            },
        }
        write_json(output / "SUCCESS.json", marker)
        return output
    except Exception:
        logger.exception("experiment failed; SUCCESS.json was not written")
        raise


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=core.MASTER_SEED)
    parser.add_argument("--model-seed", type=int, default=core.MODEL_SEED)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.replicates <= 0:
        parser.error("--replicates must be positive")
    return arguments


def main() -> None:
    print(run(parse_arguments()))


if __name__ == "__main__":
    main()
