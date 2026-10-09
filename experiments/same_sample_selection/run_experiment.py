#!/usr/bin/env python3
"""Exact-branch no-split experiment for the predeclared 2SDR advantage region."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import shlex
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1] / "src" / "fourier_2sdr"
from fourier_2sdr.selection import HHI_GIC, _eigen_gap_sum, select_pca_rank_hhi_gic
from fourier_2sdr.model import _core, _fit_spatial_subspaces


METHODS = (
    "two_sdr_sure_hhi",
    "vector_pca_hhi",
    "mpca_sure_full_core",
    "fixed_true_rank_diagnostic",
)
MPCA_MAX_ITER = 250
MPCA_TOL = 1e-5
STAGE1_CONSTRUCTION = (
    "nested_upper_envelope_truncation_of_one_over_specified_fit"
)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(namespace: int, cell_id: str, replicate: int) -> int:
    material = f"{namespace}:{cell_id}:{replicate}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (
        2**32 - 1
    )


@dataclass(frozen=True)
class GeneratedCell:
    observed: torch.Tensor
    signal_covariance: torch.Tensor
    true_left: torch.Tensor
    true_right: torch.Tensor
    true_signal_basis: torch.Tensor
    component_variances: tuple[float, ...]
    achieved_snr: float


def _ambient_tensor_basis(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Columns are C-order vec(left[:,a] right[:,b]^T)."""
    return torch.einsum("pa,qb->pqab", left, right).reshape(
        left.shape[0] * right.shape[0], left.shape[1] * right.shape[1]
    )


def generate_cell(
    *,
    shape: tuple[int, int],
    n: int,
    snr: float,
    spectrum: Iterable[float],
    eta: float,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> GeneratedCell:
    """Generate an exact real-Fourier matrix-normal signal plus white noise."""
    if shape != (32, 32):
        raise ValueError("the registered study is fixed at shape (32, 32)")
    if n < 3 or snr <= 0 or not 0 <= eta < 1:
        raise ValueError("invalid n, snr, or eta")
    ratios = torch.as_tensor(tuple(spectrum), dtype=dtype)
    if ratios.shape != (4,) or (ratios <= 0).any():
        raise ValueError("spectrum must contain four positive ratios")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    left, _ = torch.linalg.qr(
        torch.randn(shape[0], 4, generator=generator, dtype=dtype),
        mode="reduced",
    )
    right, _ = torch.linalg.qr(
        torch.randn(shape[1], 4, generator=generator, dtype=dtype),
        mode="reduced",
    )
    envelope = _ambient_tensor_basis(left, right)
    diagonal_indices = torch.tensor([0, 5, 10, 15])
    structured_basis = envelope.index_select(1, diagonal_indices)

    total_variance = float(shape[0] * shape[1]) * snr
    structured_variances = (
        ratios / ratios.sum() * ((1.0 - eta) * total_variance)
    )
    basis = structured_basis
    variances = structured_variances
    if eta:
        candidate = torch.randn(
            shape[0] * shape[1], generator=generator, dtype=dtype
        )
        candidate = candidate - envelope @ (envelope.mT @ candidate)
        candidate = candidate / torch.linalg.vector_norm(candidate)
        basis = torch.column_stack((basis, candidate))
        variances = torch.cat(
            (variances, variances.new_tensor([eta * total_variance]))
        )
    scores = torch.randn(n, len(variances), generator=generator, dtype=dtype)
    scores = scores * variances.sqrt()
    clean_flat = scores @ basis.mT
    noise = torch.randn(
        n, shape[0] * shape[1], generator=generator, dtype=dtype
    )
    observed = (clean_flat + noise).reshape(n, *shape)
    signal_covariance = (basis * variances.unsqueeze(0)) @ basis.mT
    achieved_snr = float(signal_covariance.trace() / (shape[0] * shape[1]))
    return GeneratedCell(
        observed=observed,
        signal_covariance=signal_covariance,
        true_left=left,
        true_right=right,
        true_signal_basis=basis,
        component_variances=tuple(float(value) for value in variances),
        achieved_snr=achieved_snr,
    )


def conditional_clean_prediction_risk(
    basis: torch.Tensor,
    fitted_mean: torch.Tensor,
    signal_covariance: torch.Tensor,
    *,
    sigma2: float = 1.0,
) -> float:
    """Exact risk for a fresh clean target, conditional on the fitted projector."""
    dimension = signal_covariance.shape[0]
    if basis.ndim != 2 or basis.shape[0] != dimension:
        raise ValueError("basis has the wrong ambient dimension")
    if fitted_mean.shape != (dimension,):
        raise ValueError("fitted_mean has the wrong ambient dimension")
    if basis.shape[1]:
        gram = basis.mT @ basis
        identity = torch.eye(len(gram), dtype=basis.dtype)
        if not torch.allclose(gram, identity, atol=2e-8, rtol=0):
            raise ValueError("basis must have orthonormal columns")
        captured_signal = torch.sum(basis * (signal_covariance @ basis))
        projected_mean = basis.mT @ fitted_mean
        omitted_mean = fitted_mean.square().sum() - projected_mean.square().sum()
    else:
        captured_signal = signal_covariance.new_zeros(())
        omitted_mean = fitted_mean.square().sum()
    risk = (
        signal_covariance.trace()
        - captured_signal
        + sigma2 * basis.shape[1]
        + omitted_mean
    ) / dimension
    return float(risk)


def _leading_core_basis(core: torch.Tensor, rank: int) -> torch.Tensor:
    if rank == 0:
        return core.new_empty((core.shape[1], 0))
    covariance = core.mT @ core / len(core)
    _, vectors = torch.linalg.eigh(covariance)
    return vectors[:, -rank:].flip(1).contiguous()


def _select_stage1(
    observed: torch.Tensor, candidate_values: tuple[int, ...]
) -> tuple[tuple[int, int], dict[str, dict[str, float]]]:
    """Select the Stage-I pair by the nested upper-envelope SURE criterion.

    This follows equations (2.1)--(2.3) of the original 2SDR paper: one
    over-specified MPCA fit at the largest candidate pair supplies the bases,
    and every candidate scores the leading truncation of those same bases.  The
    70S runner uses the same construction, so both studies that choose a
    Stage-I pair now minimize the criterion the published rank-pair theorem
    covers.  The candidate grid is a product of ``candidate_values``, so its
    largest entry serves as the upper envelope and every candidate is a nested
    truncation of it.
    """
    upper_p = upper_q = max(candidate_values)
    n, height, width = observed.shape
    mean_matrix = observed.mean(dim=0)
    centered = observed - mean_matrix
    left, right, upper_iterations, upper_converged = _fit_spatial_subspaces(
        (centered,), upper_p, upper_q, MPCA_MAX_ITER, MPCA_TOL
    )
    core = _core(centered, left, right).reshape(n, upper_p, upper_q)

    # Nested retained energy: the (p,q) entry is the energy kept by the leading
    # p-by-q block of the upper-envelope core.
    coordinate_energy = core.to(torch.float64).square().sum(dim=0) / n
    retained = coordinate_energy.cumsum(0).cumsum(1)
    total_energy = float(centered.to(torch.float64).square().sum() / n)

    xb = centered @ right
    covariance_left = torch.einsum("npb,nrb->pr", xb, xb).to(torch.float64) / n
    ax = torch.einsum("pa,npq->naq", left, centered)
    covariance_right = torch.einsum("naq,nar->qr", ax, ax).to(torch.float64) / n
    eigen_left = torch.linalg.eigvalsh(covariance_left).flip(0)
    eigen_right = torch.linalg.eigvalsh(covariance_right).flip(0)

    sigma2 = 1.0
    records: dict[str, dict[str, float]] = {}
    scores: dict[tuple[int, int], float] = {}
    for p in candidate_values:
        gap_left = float(_eigen_gap_sum(eigen_left, p))
        for q in candidate_values:
            gap_right = float(_eigen_gap_sum(eigen_right, q))
            residual = total_energy - float(retained[p - 1, q - 1])
            df = height * width + (n - 1) * p * q + gap_left + gap_right
            score = (
                residual
                + (2.0 * sigma2 / n) * df
                - height * width * sigma2
            )
            scores[(p, q)] = score
            records[f"{p}x{q}"] = {
                "score": score,
                "residual": residual,
                "df": df,
                # One fit serves every candidate, so the convergence record
                # belongs to the upper envelope.
                "iterations": upper_iterations,
                "converged": upper_converged,
            }

    # The registered scientific tie rule, kept explicit and auditable.
    minimum = min(scores.values())
    tolerance = (
        128
        * torch.finfo(torch.float64).eps
        * max(1.0, max(abs(value) for value in scores.values()))
    )
    tied = [pair for pair, value in scores.items() if value - minimum <= tolerance]
    selected = min(tied, key=lambda pair: (pair[0] * pair[1], pair[0], pair[1]))
    envelope = {
        "mean": mean_matrix,
        "centered": centered,
        "left": left,
        "right": right,
        "iterations": upper_iterations,
        "converged": upper_converged,
    }
    return selected, records, envelope


def _truncated_mpca_basis(
    envelope: dict[str, object], p: int, q: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take the leading p-by-q block of the upper-envelope bases.

    Chung et al. define the candidate bases this way, as submatrices of the
    single over-specified fit, so every candidate shares one alternating path.
    """
    left = envelope["left"][:, :p].contiguous()
    right = envelope["right"][:, :q].contiguous()
    centered = envelope["centered"]
    return (
        _ambient_tensor_basis(left, right),
        envelope["mean"].reshape(-1),
        _core(centered, left, right).reshape(len(centered), -1),
    )


def fit_and_score(
    generated: GeneratedCell,
    *,
    candidate_values: tuple[int, ...] = (2, 4, 8),
) -> dict[str, object]:
    observed = generated.observed
    n, height, width = observed.shape
    ambient = height * width
    selected_pair, sure_scores, envelope = _select_stage1(
        observed, candidate_values
    )
    p, q = selected_pair
    mpca_basis, mean, core = _truncated_mpca_basis(envelope, p, q)
    converged = bool(envelope["converged"])
    iterations = int(envelope["iterations"])
    core_gic = select_pca_rank_hhi_gic(
        core, min_rank=0, variant=HHI_GIC
    )
    core_rank = core_gic.selected_rank
    core_direction = _leading_core_basis(core, core_rank)
    two_sdr_basis = mpca_basis @ core_direction

    flat = observed.reshape(n, ambient)
    centered_flat = flat - flat.mean(dim=0)
    vector_gic = select_pca_rank_hhi_gic(
        centered_flat, min_rank=0, variant=HHI_GIC
    )
    vector_rank = vector_gic.selected_rank
    if vector_rank:
        # The smaller sample Gram gives the selected right singular subspace
        # without forming a second 1024-by-1024 covariance eigendecomposition.
        gram = centered_flat @ centered_flat.mT
        _, sample_vectors = torch.linalg.eigh(gram)
        sample_vectors = sample_vectors[:, -vector_rank:].flip(1)
        vector_basis = centered_flat.mT @ sample_vectors
        vector_basis = torch.linalg.qr(vector_basis, mode="reduced").Q
    else:
        vector_basis = flat.new_empty((ambient, 0))

    # The oracle diagnostic takes the same truncation at the true pair, so it
    # differs from the selected fit only in where its ranks come from.
    true_mpca_basis, true_mean, true_core = _truncated_mpca_basis(
        envelope, 4, 4
    )
    true_converged = converged
    true_iterations = iterations
    true_core_direction = _leading_core_basis(true_core, 4)
    fixed_true_rank_basis = true_mpca_basis @ true_core_direction

    bases = {
        "two_sdr_sure_hhi": two_sdr_basis,
        "vector_pca_hhi": vector_basis,
        "mpca_sure_full_core": mpca_basis,
        "fixed_true_rank_diagnostic": fixed_true_rank_basis,
    }
    means = {
        "two_sdr_sure_hhi": mean,
        "vector_pca_hhi": flat.mean(dim=0),
        "mpca_sure_full_core": mean,
        "fixed_true_rank_diagnostic": true_mean,
    }
    risks = {
        method: conditional_clean_prediction_risk(
            basis, means[method], generated.signal_covariance
        )
        for method, basis in bases.items()
    }
    return {
        "n": n,
        "selected_p": p,
        "selected_q": q,
        "selected_r": core_rank,
        "vector_pca_selected_r": vector_rank,
        "stage1_p_censored": p == max(candidate_values),
        "stage1_q_censored": q == max(candidate_values),
        "core_gic_rank_censored": core_gic.selected_at_upper_bound,
        "vector_gic_rank_censored": vector_gic.selected_at_upper_bound,
        "core_gic_rank_ambiguous": core_gic.rank_ambiguous,
        "vector_gic_rank_ambiguous": vector_gic.rank_ambiguous,
        "mpca_converged": converged,
        "mpca_iterations": iterations,
        "fixed_true_rank_mpca_converged": true_converged,
        "fixed_true_rank_mpca_iterations": true_iterations,
        "all_sure_candidates_converged": all(
            bool(record["converged"]) for record in sure_scores.values()
        ),
        "risks": risks,
        "sure_scores": sure_scores,
        "core_gic_scores": {
            str(rank): {
                "score": score.score,
                "log_determinant": score.log_determinant,
                "bias": score.bias,
                "penalty_weight": score.penalty_weight,
            }
            for rank, score in core_gic.scores.items()
        },
        "vector_gic_scores": {
            str(rank): {
                "score": score.score,
                "log_determinant": score.log_determinant,
                "bias": score.bias,
                "penalty_weight": score.penalty_weight,
            }
            for rank, score in vector_gic.scores.items()
        },
    }


def _cells(protocol: dict[str, object], mode: str) -> list[dict[str, object]]:
    mode_spec = protocol["modes"][mode]
    if "cells" in mode_spec:
        return list(mode_spec["cells"])
    filters = mode_spec["cell_filter"]
    return [
        {"n": n, "snr": snr, "spectrum": spectrum, "eta": eta}
        for n in filters["sample_sizes"]
        for snr in filters["per_pixel_snr"]
        for spectrum in filters["spectra"]
        for eta in filters["eta"]
    ]


def _cell_id(cell: dict[str, object]) -> str:
    return (
        f"n{cell['n']}_snr{float(cell['snr']):.3f}_"
        f"{cell['spectrum']}_eta{float(cell['eta']):.3f}"
    ).replace(".", "p")


def _paired_summary(
    rows: list[dict[str, object]],
    *,
    t_critical: float,
) -> list[dict[str, object]]:
    summaries = []
    comparisons = (
        ("vector_pca_hhi", "two_sdr_minus_pca"),
        ("mpca_sure_full_core", "two_sdr_minus_mpca"),
    )
    for cell_id in sorted({str(row["cell_id"]) for row in rows}):
        cell_rows = [row for row in rows if row["cell_id"] == cell_id]
        for comparator, label in comparisons:
            differences = np.asarray([
                float(row["risk_two_sdr_sure_hhi"])
                - float(row[f"risk_{comparator}"])
                for row in cell_rows
            ])
            mean = float(differences.mean())
            if len(differences) > 1:
                standard_error = float(differences.std(ddof=1) / math.sqrt(len(differences)))
                critical = (
                    t_critical
                    if len(differences) == 100
                    else float(student_t.ppf(0.975, len(differences) - 1))
                )
                lower = mean - critical * standard_error
                upper = mean + critical * standard_error
            else:
                standard_error = lower = upper = float("nan")
            summaries.append({
                "cell_id": cell_id,
                "n": int(cell_rows[0]["n"]),
                "snr": float(cell_rows[0]["snr"]),
                "spectrum": cell_rows[0]["spectrum"],
                "eta": float(cell_rows[0]["eta"]),
                "comparison": label,
                "replicates": len(differences),
                "mean_difference": mean,
                "standard_error": standard_error,
                "ci_lower": lower,
                "ci_upper": upper,
            })
    return summaries


def confirmation_decision(
    summaries: list[dict[str, object]],
    *,
    mode: str,
    expected_replicates: int,
    expected_primary_cell_ids: set[str],
) -> dict[str, object]:
    if mode != "formal":
        return {
            "status": "not_evaluated",
            "reason": "smoke and discovery outputs are non-confirmatory",
        }
    primary = [row for row in summaries if float(row["eta"]) == 0.0]
    expected_keys = {
        (cell_id, comparison)
        for cell_id in expected_primary_cell_ids
        for comparison in ("two_sdr_minus_pca", "two_sdr_minus_mpca")
    }
    observed_keys = {
        (str(row["cell_id"]), str(row["comparison"])) for row in primary
    }
    complete = observed_keys == expected_keys and len(primary) == len(expected_keys)
    complete = complete and all(
        int(row["replicates"]) == expected_replicates for row in primary
    )
    if not complete:
        return {
            "status": "incomplete",
            "reason": "not every primary comparison has 100 paired replicates",
        }
    passed = all(float(row["ci_upper"]) < 0.0 for row in primary)
    return {
        "status": "pass" if passed else "fail",
        "rule": (
            "every eta=0 primary cell has upper paired 95% CI below zero "
            "for 2SDR-PCA and 2SDR-MPCA"
        ),
        "primary_comparisons": len(primary),
    }


def _expected_checkpoints(
    cells: list[dict[str, object]], replicates: int, namespace: int
) -> dict[str, dict[str, object]]:
    expected: dict[str, dict[str, object]] = {}
    for cell in cells:
        cell_id = _cell_id(cell)
        for replicate in range(replicates):
            filename = f"{cell_id}__rep{replicate:03d}.json"
            if filename in expected:
                raise ValueError(f"duplicate registered checkpoint: {filename}")
            expected[filename] = {
                "cell_id": cell_id,
                "replicate": replicate,
                "seed": _seed(namespace, cell_id, replicate),
                "n": int(cell["n"]),
                "snr": float(cell["snr"]),
                "spectrum": str(cell["spectrum"]),
                "eta": float(cell["eta"]),
            }
    return expected


def _validate_checkpoint(
    path: Path, expected: dict[str, object]
) -> dict[str, object]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint {path}: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"checkpoint is not a JSON object: {path}")
    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise ValueError(
                f"checkpoint metadata mismatch for {path.name}: "
                f"{key}={record.get(key)!r}, expected {expected_value!r}"
            )
    missing = {"risks", "selected_p", "selected_q", "selected_r"} - set(record)
    if missing:
        raise ValueError(
            f"checkpoint {path.name} is missing required fields: {sorted(missing)}"
        )
    return record


def _flatten_replicates(
    attempt: Path, expected: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    paths = {
        path.name: path
        for path in (attempt / "replicates").glob("*.json")
    }
    if set(paths) != set(expected):
        missing = sorted(set(expected) - set(paths))
        extras = sorted(set(paths) - set(expected))
        raise ValueError(
            f"checkpoint set mismatch; missing={missing}, extras={extras}"
        )
    rows = []
    for filename in sorted(expected):
        record = _validate_checkpoint(paths[filename], expected[filename])
        row = {
            key: value for key, value in record.items()
            if key not in {"risks", "sure_scores", "core_gic_scores", "vector_gic_scores"}
        }
        row.update({f"risk_{key}": value for key, value in record["risks"].items()})
        rows.append(row)
    return rows


def _selector_diagnostics(rows: list[dict[str, object]]) -> dict[str, object]:
    fields = {
        "stage1_boundary": lambda row: (
            bool(row["stage1_p_censored"]) or bool(row["stage1_q_censored"])
        ),
        "core_gic_upper_bound": lambda row: bool(row["core_gic_rank_censored"]),
        "vector_gic_upper_bound": lambda row: bool(row["vector_gic_rank_censored"]),
        "core_gic_ambiguous": lambda row: bool(row["core_gic_rank_ambiguous"]),
        "vector_gic_ambiguous": lambda row: bool(row["vector_gic_rank_ambiguous"]),
        "selected_mpca_nonconverged": lambda row: not bool(row["mpca_converged"]),
        "any_sure_candidate_nonconverged": lambda row: not bool(
            row["all_sure_candidates_converged"]
        ),
    }
    output: dict[str, object] = {}
    rank_claim_blocked = False
    for scope, scoped_rows in (
        ("all", rows),
        ("primary_eta_0", [row for row in rows if float(row["eta"]) == 0.0]),
    ):
        counts = {
            label: sum(predicate(row) for row in scoped_rows)
            for label, predicate in fields.items()
        }
        output[scope] = {
            "replicates": len(scoped_rows),
            "counts": counts,
            "rates": {
                label: count / len(scoped_rows) if scoped_rows else float("nan")
                for label, count in counts.items()
            },
        }
        if scope == "primary_eta_0":
            rank_claim_blocked = any(counts.values())
    output["formal_decision_scope"] = (
        "algorithm-level paired conditional clean-prediction risk"
    )
    output["rank_recovery_claim_allowed"] = not rank_claim_blocked
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(attempt: Path, protocol: dict[str, object], mode: str) -> None:
    if protocol["selection"]["stage1_construction"] != STAGE1_CONSTRUCTION:
        raise ValueError(
            "protocol must register the nested upper-envelope Stage-I construction"
        )
    shape = tuple(protocol["generator"]["shape"])
    spectra = protocol["generator"]["spectra"]
    mode_spec = protocol["modes"][mode]
    namespace = int(mode_spec["seed_namespace"])
    replicates = int(mode_spec["replicates"])
    cells = _cells(protocol, mode)
    expected = _expected_checkpoints(cells, replicates, namespace)
    existing = {
        path.name: path for path in (attempt / "replicates").glob("*.json")
    }
    extras = sorted(set(existing) - set(expected))
    if extras:
        raise ValueError(f"unregistered checkpoint files present: {extras}")
    for filename, metadata in expected.items():
        destination = attempt / "replicates" / filename
        if destination.is_file():
            _validate_checkpoint(destination, metadata)
            continue
        run_seed = int(metadata["seed"])
        logging.info(
            "%s replicate %d/%d seed=%d",
            metadata["cell_id"],
            int(metadata["replicate"]) + 1,
            replicates,
            run_seed,
        )
        generated = generate_cell(
            shape=shape,
            n=int(metadata["n"]),
            snr=float(metadata["snr"]),
            spectrum=spectra[str(metadata["spectrum"])],
            eta=float(metadata["eta"]),
            seed=run_seed,
        )
        result = fit_and_score(generated)
        result.update({
            **metadata,
            "achieved_population_snr": generated.achieved_snr,
            "component_variances": generated.component_variances,
        })
        _write_json(destination, result)

    rows = _flatten_replicates(attempt, expected)
    _write_csv(attempt / "outputs" / "per_replicate.csv", rows)
    summaries = _paired_summary(
        rows,
        t_critical=float(protocol["confirmation"]["t_critical_df_99"]),
    )
    _write_csv(attempt / "outputs" / "paired_summary.csv", summaries)
    _write_json(attempt / "outputs" / "paired_summary.json", summaries)
    decision = confirmation_decision(
        summaries,
        mode=mode,
        expected_replicates=int(
            protocol["confirmation"]["formal_replicates_per_cell"]
        ),
        expected_primary_cell_ids={
            _cell_id(cell) for cell in cells if float(cell["eta"]) == 0.0
        },
    )
    diagnostics = _selector_diagnostics(rows)
    decision["scope"] = diagnostics["formal_decision_scope"]
    decision["rank_recovery_claim_allowed"] = diagnostics[
        "rank_recovery_claim_allowed"
    ]
    _write_json(attempt / "outputs" / "selector_diagnostics.json", diagnostics)
    _write_json(attempt / "decision.json", decision)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--resume-attempt", type=Path)
    args = parser.parse_args()
    if sum((args.smoke, args.quick, args.formal)) != 1:
        raise SystemExit("choose exactly one of --smoke, --quick, or --formal")
    mode = "smoke" if args.smoke else "discovery" if args.quick else "formal"
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = _sha256(args.protocol)

    if args.resume_attempt:
        attempt = args.resume_attempt.resolve()
        if not attempt.is_dir():
            raise FileNotFoundError(
                f"resume attempt does not exist: {attempt}"
            )
    else:
        attempt = args.output_root.resolve() / (
            "attempt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        attempt.mkdir(parents=True)
    (attempt / "outputs").mkdir(exist_ok=True)
    (attempt / "replicates").mkdir(exist_ok=True)
    sources = [
        Path(__file__),
        args.protocol,
        PACKAGE_ROOT / "selection.py",
        PACKAGE_ROOT / "model.py",
    ]
    source_hashes = {
        str(path.resolve()): _sha256(path)
        for path in sources
    }
    runtime_fingerprint = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": sys.modules["scipy"].__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_default_dtype": str(torch.get_default_dtype()),
    }
    configuration = {
        "analysis_id": protocol["analysis_id"],
        "mode": mode,
        "protocol_sha256": protocol_hash,
        "seed_namespace": protocol["modes"][mode]["seed_namespace"],
        "source_sha256": source_hashes,
        "runtime": runtime_fingerprint,
    }
    configuration_path = attempt / "configuration.json"
    if configuration_path.is_file():
        if json.loads(configuration_path.read_text(encoding="utf-8")) != configuration:
            raise RuntimeError("resume protocol, mode, or seed namespace differs")
    else:
        _write_json(configuration_path, configuration)
    command = [sys.executable, *sys.argv]
    initial_command = attempt / "command.json"
    if not initial_command.is_file():
        _write_json(
            initial_command,
            {"argv": command, "shell": shlex.join(command)},
        )
    source_provenance = attempt / "source_provenance.json"
    if not source_provenance.is_file():
        _write_json(source_provenance, {
        "schema_version": 1,
        "runtime": runtime_fingerprint,
        "sources": source_hashes,
        })
    (attempt / "invocations").mkdir(exist_ok=True)
    invocation_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-pid{os.getpid()}"
    )
    _write_json(attempt / "invocations" / f"{invocation_id}.json", {
        "argv": command,
        "shell": shlex.join(command),
        "runtime": runtime_fingerprint,
        "source_sha256": source_hashes,
    })
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(attempt / "run.log"),
            logging.StreamHandler(),
        ],
    )
    _write_json(attempt / "status.json", {"status": "running", "mode": mode})
    try:
        run(attempt, protocol, mode)
    except Exception as error:
        logging.error("%s\n%s", error, traceback.format_exc())
        _write_json(
            attempt / "status.json",
            {"status": "failed", "mode": mode, "error": repr(error)},
        )
        raise
    _write_json(attempt / "status.json", {"status": "success", "mode": mode})
    logging.info("completed %s attempt at %s", mode, attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
