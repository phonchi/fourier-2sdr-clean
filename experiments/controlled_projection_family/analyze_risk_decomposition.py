#!/usr/bin/env python3
"""Decompose controlled projection risk into missed signal and retained noise.

The analysis consumes only a completed run's ``raw_metrics.csv``.  It is
therefore reusable for the superseded 0.05/0.20 run and the current v2 design.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


REQUIRED_FIELDS = {
    "scenario", "role", "p", "q", "r", "snr", "sample_size",
    "replicate", "method", "clean_signal_capture", "retained_noise_energy",
    "population_projection_risk",
}
RAW_FIELDS = (
    "scenario", "role", "p", "q", "r", "snr", "sample_size", "replicate",
    "method", "missed_signal_risk", "retained_noise_risk",
    "reconstructed_population_risk", "reported_population_projection_risk",
    "risk_identity_residual",
)
SUMMARY_FIELDS = (
    "scenario", "role", "p", "q", "r", "snr", "sample_size", "method",
    "component", "count", "mean", "sample_sd", "standard_error", "ci95_low",
    "ci95_high",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_raw(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"raw_metrics.csv is missing fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("raw_metrics.csv is empty")
    return rows


def decompose_rows(rows: Iterable[Mapping[str, str]], *, ambient_dim: int) -> list[dict[str, Any]]:
    if ambient_dim <= 0:
        raise ValueError("ambient_dim must be positive")
    result: list[dict[str, Any]] = []
    for source in rows:
        snr = float(source["snr"])
        if not math.isfinite(snr) or snr <= 0:
            raise ValueError(f"invalid SNR: {source['snr']!r}")
        capture = float(source["clean_signal_capture"])
        retained = float(source["retained_noise_energy"])
        reported = float(source["population_projection_risk"])
        missed_risk = 1.0 - capture
        retained_risk = retained / (ambient_dim * snr)
        reconstructed = missed_risk + retained_risk
        result.append({
            "scenario": source["scenario"], "role": source["role"],
            "p": int(source["p"]), "q": int(source["q"]), "r": int(source["r"]),
            "snr": snr, "sample_size": int(source["sample_size"]),
            "replicate": int(source["replicate"]), "method": source["method"],
            "missed_signal_risk": missed_risk,
            "retained_noise_risk": retained_risk,
            "reconstructed_population_risk": reconstructed,
            "reported_population_projection_risk": reported,
            "risk_identity_residual": reconstructed - reported,
        })
    return result


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    count = len(array)
    mean = float(array.mean())
    if count > 1:
        sd = float(array.std(ddof=1))
        se = sd / math.sqrt(count)
        critical = float(student_t.ppf(0.975, count - 1))
        low, high = mean - critical * se, mean + critical * se
    else:
        sd = se = 0.0
        low = high = mean
    return {
        "count": count, "mean": mean, "sample_sd": sd, "standard_error": se,
        "ci95_low": low, "ci95_high": high,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario"], row["role"], row["p"], row["q"], row["r"],
            row["snr"], row["sample_size"], row["method"],
        )
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        common = dict(zip(
            ("scenario", "role", "p", "q", "r", "snr", "sample_size", "method"), key
        ))
        for component, field in (
            ("missed_signal", "missed_signal_risk"),
            ("retained_noise", "retained_noise_risk"),
            ("total", "reconstructed_population_risk"),
        ):
            output.append({
                **common, "component": component,
                **_summary([float(row[field]) for row in members]),
            })
    return output


def _write_figure(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    interior = sorted({
        str(row["scenario"]) for row in rows
        if str(row["scenario"]).startswith("interior_")
    })
    scenarios = interior or sorted({str(row["scenario"]) for row in rows})
    snrs = sorted({float(row["snr"]) for row in rows})
    sample_sizes = sorted({int(row["sample_size"]) for row in rows})
    methods = [name for name in ("pca", "twosdr", "mpca") if any(row["method"] == name for row in rows)]
    colors = {"missed_signal": "#4c78a8", "retained_noise": "#f58518"}
    labels = {"pca": "PCA", "twosdr": "2SDR", "mpca": "MPCA"}
    lookup = {
        (row["scenario"], float(row["snr"]), int(row["sample_size"]), row["method"], row["component"]): row
        for row in rows
    }
    figure, axes = plt.subplots(
        len(snrs), len(scenarios), figsize=(3.8 * len(scenarios), 3.1 * len(snrs)),
        squeeze=False, sharey="row",
    )
    width = 0.74
    group_centers = 4.0 * np.arange(len(sample_sizes), dtype=np.float64)
    offsets = np.linspace(-0.9, 0.9, len(methods)) if len(methods) > 1 else np.zeros(1)
    for row_index, snr in enumerate(snrs):
        for col_index, scenario in enumerate(scenarios):
            axis = axes[row_index, col_index]
            for method_index, method in enumerate(methods):
                position = group_centers + offsets[method_index]
                missed = np.array([
                    lookup[(scenario, snr, n, method, "missed_signal")]["mean"]
                    for n in sample_sizes
                ])
                noise = np.array([
                    lookup[(scenario, snr, n, method, "retained_noise")]["mean"]
                    for n in sample_sizes
                ])
                axis.bar(position, missed, width, color=colors["missed_signal"], alpha=0.9)
                axis.bar(position, noise, width, bottom=missed, color=colors["retained_noise"], alpha=0.9)
            tick_positions, tick_labels = [], []
            for center, sample_size in zip(group_centers, sample_sizes):
                for method_index, method in enumerate(methods):
                    tick_positions.append(center + offsets[method_index])
                    tick_labels.append(labels.get(method, method))
                axis.text(
                    center, -0.19, f"n={sample_size}",
                    transform=axis.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7,
                )
            axis.set_xticks(
                tick_positions, tick_labels, fontsize=6.5, rotation=28, ha="right"
            )
            axis.grid(axis="y", alpha=0.25)
            axis.set_title(f"{scenario.replace('interior_', '').replace('_', ' ')}\nSNR {snr:.2f}")
            if col_index == 0:
                axis.set_ylabel("Normalized population risk")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["missed_signal"]),
        plt.Rectangle((0, 0), 1, 1, color=colors["retained_noise"]),
    ]
    figure.suptitle("Population-risk mechanism decomposition", y=0.998)
    figure.legend(
        handles, ["Missed signal", "Retained noise"], loc="upper center",
        ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.971),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.918))
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"population_risk_decomposition.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ambient-dim", type=int, default=192)
    parser.add_argument("--identity-tolerance", type=float, default=5e-12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    raw_path = run_dir / "raw_metrics.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    output = (args.output_dir or run_dir / "risk_decomposition").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = decompose_rows(_read_raw(raw_path), ambient_dim=args.ambient_dim)
    maximum_residual = max(abs(float(row["risk_identity_residual"])) for row in rows)
    if maximum_residual > args.identity_tolerance:
        raise RuntimeError(
            f"population-risk identity residual {maximum_residual:.3e} exceeds "
            f"tolerance {args.identity_tolerance:.3e}"
        )
    summary_rows = summarize(rows)
    _write_csv(output / "risk_decomposition_raw.csv", rows, RAW_FIELDS)
    _write_csv(output / "risk_decomposition_summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_figure(summary_rows, output)
    provenance = {
        "schema_version": 1, "created_at_utc": _utc_now(),
        "analysis": "controlled_projection_family_risk_decomposition",
        "source_run_dir": str(run_dir), "source_raw_metrics": str(raw_path),
        "source_raw_metrics_sha256": _sha256(raw_path),
        "analysis_source": str(Path(__file__).resolve()),
        "analysis_source_sha256": _sha256(Path(__file__).resolve()),
        "ambient_dimension": args.ambient_dim,
        "identity_tolerance": args.identity_tolerance,
        "python": platform.python_version(), "numpy": np.__version__,
    }
    _write_json(output / "provenance.json", provenance)
    success = {
        "status": "success", "completed_at_utc": _utc_now(),
        "raw_row_count": len(rows), "summary_row_count": len(summary_rows),
        "maximum_absolute_risk_identity_residual": maximum_residual,
        "source_snr_values": sorted({float(row["snr"]) for row in rows}),
        "artifacts": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    _write_json(output / "SUCCESS.json", success)
    print(json.dumps(success, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
