#!/usr/bin/env python3
"""Same-sample Tu-SURE/HHI-GIC application to the synthetic 70S stacks."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import shlex
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO_ROOT = EXPERIMENTS.parent
WORKSPACE = REPO_ROOT
PACKAGE_ROOT = REPO_ROOT / "src" / "fourier_2sdr"
from fourier_2sdr.fourier import (
    _from_valid_rfft2,
    real_fourier_inverse as separable_real_fourier_inverse,
)
from fourier_2sdr.selection import (
    HHI_GIC,
    _eigen_gap_sum,
    _gic_components,
    hhi_rank_admissible_upper_bound,
    hhi_rank_upper_bound,
    select_pca_rank_hhi_gic,
)
from fourier_2sdr.model import _core, _fit_spatial_subspaces
from fourier_2sdr import io as mrc_io


COORDINATE_EXECUTION = (
    "orthonormal_rfft2_then_conjugacy_aware_block_algebra_to_complete_130_by_130_Q_coordinates"
)

METHODS = (
    "two_sdr_tu_sure_hhi",
    "vector_pca_hhi",
    "mpca_tu_sure_full_core",
)


def images_to_q_coordinates(images: np.ndarray) -> np.ndarray:
    """Convert spatial images through rfft2 to the square real Fourier matrix."""
    array = np.asarray(images)
    if not array.flags.writeable:
        array = array.copy()
    observed = torch.as_tensor(array, dtype=torch.float32)
    if observed.ndim != 3:
        raise ValueError("images must have shape (n,height,width)")
    half_spectrum = torch.fft.rfft2(observed, norm="ortho")
    coordinates = _from_valid_rfft2(
        half_spectrum, spatial_shape=tuple(observed.shape[-2:])
    )
    return coordinates.numpy()


def q_coordinates_to_images(coordinates: np.ndarray) -> np.ndarray:
    """Invert the square real Fourier matrix to spatial images."""
    values = torch.as_tensor(np.asarray(coordinates), dtype=torch.float32)
    if values.ndim != 3:
        raise ValueError("coordinates must have shape (n,height,width)")
    return separable_real_fourier_inverse(values).numpy()


def _json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(dict.fromkeys(
        key for row in rows for key in row
    ))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _serializable(dataclasses.asdict(value))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mrc(path: Path) -> np.memmap:
    header, offset = mrc_io.mrc_header(path)
    return np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=offset,
        shape=tuple(header["shape"]),
        order="C",
    )


def discover_datasets(protocol: dict[str, object]) -> list[SimpleNamespace]:
    spec = protocol["dataset"]
    root = WORKSPACE / str(spec["dataset_root"])
    output = []
    signatures = []
    for snr in spec["snrs"]:
        directory = root / f"white_on_black_snr{int(round(100 * snr)):03d}_seed0"
        paths = {
            name: directory / name
            for name in spec["required_inputs"]
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"synthetic 70S source is incomplete: {missing}")
        manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
        parameters = manifest["parameters"]
        if (
            int(parameters["n_particles"]) != int(spec["particle_count"])
            or int(parameters["n_views"]) != int(spec["view_count"])
            or not math.isclose(float(parameters["snr"]), float(snr))
            or parameters["contrast_convention"] != spec["contrast_convention"]
        ):
            raise RuntimeError(f"70S manifest contract mismatch: {directory}")
        records = _records(paths["ground_truth.jsonl"])
        if len(records) != int(spec["particle_count"]):
            raise RuntimeError(f"70S metadata count mismatch: {directory}")
        if any(
            abs(float(row.get("in_plane_rotation_rad", 0.0))) > 1e-12
            or any(abs(float(value)) > 1e-12 for value in row["shift_pixels"])
            for row in records
        ):
            raise RuntimeError(f"70S dataset violates the zero-pose contract: {directory}")
        signature = [
            (
                int(row["particle_index"]),
                int(row["gt_view_id"]),
                int(row["ctf_index"]),
            )
            for row in records
        ]
        signatures.append(signature)
        output.append(
            SimpleNamespace(
                snr=float(snr),
                directory=directory,
                paths=paths,
                manifest=manifest,
                records=records,
            )
        )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("particle/view/CTF metadata are not coupled across SNR")
    reference_shape = tuple(_mrc(output[0].paths["particles.mrcs"]).shape)
    expected_shape = (
        int(spec["particle_count"]),
        int(spec["shape"][0]),
        int(spec["shape"][1]),
    )
    if reference_shape != expected_shape:
        raise RuntimeError(
            f"70S particle shape mismatch: expected {expected_shape}, got {reference_shape}"
        )
    return output


def estimate_base_noise_variance(
    low_stack: np.ndarray,
    high_stack: np.ndarray,
    *,
    low_snr: float,
    high_snr: float,
    count: int,
    chunk_size: int = 64,
) -> float:
    """Estimate the coupled SNR=1 base-noise variance without clean targets."""
    denominator = 1.0 / math.sqrt(low_snr) - 1.0 / math.sqrt(high_snr)
    total = 0.0
    values = 0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        difference = (
            np.asarray(low_stack[start:stop], dtype=np.float64)
            - np.asarray(high_stack[start:stop], dtype=np.float64)
        ) / denominator
        total += float(np.square(difference).sum(dtype=np.float64))
        values += int(difference.size)
    if values == 0 or total <= 0:
        raise RuntimeError("coupled-observation noise estimate is nonpositive")
    return total / values


def _ambient_tensor_basis(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.einsum("pa,qb->pqab", left, right).reshape(
        left.shape[0] * right.shape[0], left.shape[1] * right.shape[1]
    )


def _fit_upper_envelope(
    images: torch.Tensor,
    *,
    upper_p: int,
    upper_q: int,
    max_iter: int,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
    mean = images.mean(dim=0)
    centered = images - mean
    left, right, iterations, converged = _fit_spatial_subspaces(
        (centered,), upper_p, upper_q, max_iter, tolerance
    )
    return mean, centered, left, right, iterations, converged


def select_nested_tu_sure(
    images: torch.Tensor,
    *,
    sigma2: float,
    upper_p: int,
    upper_q: int,
    max_iter: int,
    tolerance: float,
) -> dict[str, object]:
    """Select nested Stage-I ranks using original 2SDR equations (2.1)--(2.3)."""
    if images.ndim != 3 or images.is_complex():
        raise ValueError("images must be a real tensor of shape (n,p,q)")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    n, height, width = images.shape
    if not 0 < upper_p < height or not 0 < upper_q < width:
        raise ValueError("upper envelope ranks must be inside the image dimensions")
    mean, centered, left, right, iterations, converged = _fit_upper_envelope(
        images,
        upper_p=upper_p,
        upper_q=upper_q,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    core = _core(centered, left, right).reshape(n, upper_p, upper_q)
    coordinate_energy = core.to(torch.float64).square().sum(dim=0) / n
    retained = coordinate_energy.cumsum(0).cumsum(1)
    total_energy = float(
        centered.to(torch.float64).square().sum() / n
    )

    xb = centered @ right
    covariance_left = (
        torch.einsum("npb,nrb->pr", xb, xb).to(torch.float64) / n
    )
    ax = torch.einsum("pa,npq->naq", left, centered)
    covariance_right = (
        torch.einsum("naq,nar->qr", ax, ax).to(torch.float64) / n
    )
    eigen_left = torch.linalg.eigvalsh(covariance_left).flip(0)
    eigen_right = torch.linalg.eigvalsh(covariance_right).flip(0)
    gap_left = {
        p: float(_eigen_gap_sum(eigen_left, p))
        for p in range(1, upper_p + 1)
    }
    gap_right = {
        q: float(_eigen_gap_sum(eigen_right, q))
        for q in range(1, upper_q + 1)
    }
    rows = []
    for p in range(1, upper_p + 1):
        for q in range(1, upper_q + 1):
            residual = total_energy - float(retained[p - 1, q - 1])
            df = (
                height * width
                + (n - 1) * p * q
                + gap_left[p]
                + gap_right[q]
            )
            score = residual + (2.0 * sigma2 / n) * df - height * width * sigma2
            rows.append(
                {
                    "p": p,
                    "q": q,
                    "score": score,
                    "residual": residual,
                    "df": df,
                    "finite": math.isfinite(score),
                }
            )
    finite = [row for row in rows if bool(row["finite"])]
    if not finite:
        raise RuntimeError("all nested Tu SURE candidates are nonfinite")
    minimum = min(float(row["score"]) for row in finite)
    scale = max(1.0, *(abs(float(row["score"])) for row in finite))
    tie_tolerance = 128 * torch.finfo(torch.float64).eps * scale
    tied = [
        row
        for row in finite
        if float(row["score"]) - minimum <= tie_tolerance
    ]
    selected = min(
        tied,
        key=lambda row: (
            int(row["p"]) * int(row["q"]),
            int(row["p"]),
            int(row["q"]),
        ),
    )
    return {
        "selected_p": int(selected["p"]),
        "selected_q": int(selected["q"]),
        "selected_score": float(selected["score"]),
        "candidate_rows": rows,
        "mean": mean,
        "centered": centered,
        "upper_left": left,
        "upper_right": right,
        "upper_core": core,
        "upper_iterations": int(iterations),
        "upper_converged": bool(converged),
        "selected_at_p_upper_bound": int(selected["p"]) == upper_p,
        "selected_at_q_upper_bound": int(selected["q"]) == upper_q,
        "tie_tolerance": tie_tolerance,
        "tied_pairs": [
            [int(row["p"]), int(row["q"])] for row in tied
        ],
    }


def select_hhi_from_eigenvalues(
    eigenvalues: torch.Tensor,
    *,
    sample_count: int,
    dimension: int,
    max_rank: int,
) -> dict[str, object]:
    """Apply the shared HHI 2/n score to an already computed eigenspectrum."""
    values = eigenvalues.to(dtype=torch.float64, device="cpu")
    if values.shape != (dimension,):
        raise ValueError("eigenvalues must contain the complete ambient spectrum")
    registered_upper = hhi_rank_upper_bound(sample_count, dimension)
    admissible_upper = hhi_rank_admissible_upper_bound(
        sample_count, dimension
    )
    if not 0 <= max_rank <= admissible_upper:
        raise ValueError(
            f"max_rank {max_rank} exceeds HHI admissible upper bound "
            f"{admissible_upper}"
        )
    rows = []
    for rank in range(0, max_rank + 1):
        logdet, base_bias = _gic_components(values, rank)
        bias = base_bias + dimension
        score = logdet + (2.0 / sample_count) * bias
        rows.append(
            {
                "rank": rank,
                "score": float(score),
                "log_determinant": float(logdet),
                "bias": float(bias),
                "penalty_weight": 2.0,
                "finite": bool(torch.isfinite(score)),
            }
        )
    for index, row in enumerate(rows):
        row["delta_score"] = (
            None
            if index == 0
            else float(row["score"]) - float(rows[index - 1]["score"])
        )
    finite = [row for row in rows if bool(row["finite"])]
    if not finite:
        raise RuntimeError("all HHI GIC candidates are nonfinite")
    best = min(finite, key=lambda row: float(row["score"]))
    scale = max(1.0, *(abs(float(row["score"])) for row in finite))
    tolerance = 128 * torch.finfo(torch.float64).eps * scale
    tied = [
        row
        for row in finite
        if float(row["score"]) - float(best["score"]) <= tolerance
    ]
    selected = min(int(row["rank"]) for row in tied)
    return {
        "selected_rank": selected,
        "rows": rows,
        "theoretical_upper_bound": registered_upper,
        "registered_upper_bound": registered_upper,
        "admissible_upper_bound": admissible_upper,
        "search_upper_bound": max_rank,
        "selected_at_upper_bound": selected == max_rank,
        "rank_ambiguous": len(tied) > 1,
        "tied_ranks": [int(row["rank"]) for row in tied],
    }


def _leading_core_basis(core: torch.Tensor, rank: int) -> torch.Tensor:
    if rank == 0:
        return core.new_empty((core.shape[1], 0))
    covariance = core.mT @ core / len(core)
    _, vectors = torch.linalg.eigh(covariance.to(torch.float64))
    return vectors[:, -rank:].flip(1).to(core.dtype).contiguous()


def fit_condition(
    images: np.ndarray,
    *,
    sigma2: float,
    upper: tuple[int, int],
    pca_max_rank: int,
    max_iter: int,
    tolerance: float,
) -> dict[str, object]:
    """Fit all ranks and return the state needed for delayed target evaluation."""
    observed = torch.as_tensor(np.asarray(images), dtype=torch.float32)
    stage1 = select_nested_tu_sure(
        observed,
        sigma2=sigma2,
        upper_p=upper[0],
        upper_q=upper[1],
        max_iter=max_iter,
        tolerance=tolerance,
    )
    p = int(stage1["selected_p"])
    q = int(stage1["selected_q"])
    left = stage1["upper_left"][:, :p].contiguous()
    right = stage1["upper_right"][:, :q].contiguous()
    centered = stage1["centered"]
    core = _core(centered, left, right).reshape(len(observed), p * q)
    core_gic = select_pca_rank_hhi_gic(
        core.to(torch.float64),
        min_rank=0,
        variant=HHI_GIC,
    )
    core_rank = int(core_gic.selected_rank)
    core_basis = _leading_core_basis(core, core_rank)

    flat = centered.reshape(len(observed), -1).contiguous()
    gram = (flat @ flat.mT).to(torch.float64) / len(flat)
    gram_eigenvalues, gram_eigenvectors = torch.linalg.eigh(gram)
    order = torch.arange(
        len(gram_eigenvalues) - 1, -1, -1, device=gram_eigenvalues.device
    )
    nonzero = gram_eigenvalues.index_select(0, order).clamp_min(0)
    full_eigenvalues = torch.cat(
        (
            nonzero,
            torch.zeros(
                flat.shape[1] - len(flat),
                dtype=nonzero.dtype,
                device=nonzero.device,
            ),
        )
    )
    pca_gic = select_hhi_from_eigenvalues(
        full_eigenvalues,
        sample_count=len(flat),
        dimension=flat.shape[1],
        max_rank=pca_max_rank,
    )
    pca_rank = int(pca_gic["selected_rank"])
    pca_sample_basis = (
        gram_eigenvectors.index_select(1, order[:pca_rank]).to(torch.float32)
        if pca_rank
        else torch.empty((len(flat), 0), dtype=torch.float32)
    )
    return {
        "mean": stage1["mean"].detach().cpu().numpy(),
        "left": left.detach().cpu().numpy(),
        "right": right.detach().cpu().numpy(),
        "core_basis": core_basis.detach().cpu().numpy(),
        "pca_sample_basis": pca_sample_basis.detach().cpu().numpy(),
        "selected_p": p,
        "selected_q": q,
        "selected_r": core_rank,
        "pca_selected_r": pca_rank,
        "stage1": {
            key: value
            for key, value in stage1.items()
            if key not in {
                "mean",
                "centered",
                "upper_left",
                "upper_right",
                "upper_core",
            }
        },
        "core_gic": _serializable(core_gic),
        "pca_gic": pca_gic,
    }


def reconstruct_methods(
    images: np.ndarray,
    fitted: dict[str, object],
) -> dict[str, np.ndarray]:
    observed = torch.as_tensor(np.asarray(images), dtype=torch.float32)
    mean = torch.as_tensor(fitted["mean"], dtype=torch.float32)
    centered = observed - mean
    left = torch.as_tensor(fitted["left"], dtype=torch.float32)
    right = torch.as_tensor(fitted["right"], dtype=torch.float32)
    core = _core(centered, left, right).reshape(len(observed), -1)
    core_basis = torch.as_tensor(fitted["core_basis"], dtype=torch.float32)
    if core_basis.shape[1]:
        reduced_core = (core @ core_basis) @ core_basis.mT
    else:
        reduced_core = torch.zeros_like(core)
    two_sdr = (
        torch.einsum(
            "pa,nab,qb->npq",
            left,
            reduced_core.reshape(len(observed), left.shape[1], right.shape[1]),
            right,
        )
        + mean
    )
    mpca = (
        torch.einsum(
            "pa,nab,qb->npq",
            left,
            core.reshape(len(observed), left.shape[1], right.shape[1]),
            right,
        )
        + mean
    )
    sample_basis = torch.as_tensor(
        fitted["pca_sample_basis"], dtype=torch.float32
    )
    flat = centered.reshape(len(observed), -1)
    pca = (
        (sample_basis @ (sample_basis.mT @ flat)).reshape_as(observed)
        + mean
        if sample_basis.shape[1]
        else mean.expand_as(observed)
    )
    return {
        "two_sdr_tu_sure_hhi": two_sdr.numpy(),
        "vector_pca_hhi": pca.numpy(),
        "mpca_tu_sure_full_core": mpca.numpy(),
    }


def _selection_rows(
    snr: float,
    fitted: dict[str, object],
    *,
    sigma2: float,
    runtime: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    stage1 = fitted["stage1"]
    core_gic = fitted["core_gic"]
    pca_gic = fitted["pca_gic"]
    common = {
        "snr": snr,
        "n_particles": len(fitted["pca_sample_basis"]),
        "p": fitted["selected_p"],
        "q": fitted["selected_q"],
        "r": fitted["selected_r"],
        "pca_r": fitted["pca_selected_r"],
        "sigma2_plugin": sigma2,
        "runtime_seconds": runtime,
        "stage1_p_right_censored": stage1["selected_at_p_upper_bound"],
        "stage1_q_right_censored": stage1["selected_at_q_upper_bound"],
        "core_gic_right_censored": core_gic["selected_at_upper_bound"],
        "pca_gic_right_censored": pca_gic["selected_at_upper_bound"],
        "core_gic_registered_upper_bound": core_gic[
            "rank_upper_bound"
        ],
        "core_gic_admissible_upper_bound": core_gic[
            "admissible_upper_bound"
        ],
        "core_gic_search_upper_bound": core_gic[
            "search_upper_bound"
        ],
        "pca_gic_registered_upper_bound": pca_gic[
            "registered_upper_bound"
        ],
        "pca_gic_admissible_upper_bound": pca_gic[
            "admissible_upper_bound"
        ],
        "pca_gic_search_upper_bound": pca_gic[
            "search_upper_bound"
        ],
        "core_gic_rank_ambiguous": core_gic["rank_ambiguous"],
        "pca_gic_rank_ambiguous": pca_gic["rank_ambiguous"],
        "upper_envelope_converged": stage1["upper_converged"],
        "upper_envelope_iterations": stage1["upper_iterations"],
    }
    sure_rows = [
        {**common, **row}
        for row in stage1["candidate_rows"]
    ]
    gic_rows = [
        {
            **common,
            "method": "two_sdr_core",
            "candidate_rank": int(rank),
            **{
                key: value
                for key, value in score.items()
                if key != "rank"
            },
        }
        for rank, score in core_gic["scores"].items()
    ] + [
        {
            **common,
            "method": "vector_pca",
            "candidate_rank": int(row["rank"]),
            **{
                key: value
                for key, value in row.items()
                if key not in {"rank", "finite"}
            },
        }
        for row in pca_gic["rows"]
    ]
    return common, sure_rows, gic_rows


def _metric_rows(
    snr: float,
    reconstructions: dict[str, np.ndarray],
    target: np.ndarray,
    view_ids: np.ndarray,
    fitted: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranks = {
        "two_sdr_tu_sure_hhi": int(fitted["selected_r"]),
        "vector_pca_hhi": int(fitted["pca_selected_r"]),
        "mpca_tu_sure_full_core": (
            int(fitted["selected_p"]) * int(fitted["selected_q"])
        ),
    }
    aggregate = []
    per_view = []
    for method, estimate in reconstructions.items():
        squared = np.square(
            np.asarray(estimate, dtype=np.float64)
            - np.asarray(target, dtype=np.float64)
        )
        per_particle = squared.mean(axis=(1, 2))
        aggregate.append(
            {
                "snr": snr,
                "method": method,
                "rank": ranks[method],
                "p": (
                    fitted["selected_p"]
                    if method != "vector_pca_hhi"
                    else ""
                ),
                "q": (
                    fitted["selected_q"]
                    if method != "vector_pca_hhi"
                    else ""
                ),
                "n_particles": len(target),
                "mse": float(per_particle.mean()),
                "estimand": "same_sample_full_band_clean_observed",
            }
        )
        for view in sorted(np.unique(view_ids)):
            chosen = view_ids == view
            per_view.append(
                {
                    "snr": snr,
                    "view_id": int(view),
                    "method": method,
                    "n_particles": int(chosen.sum()),
                    "rank": ranks[method],
                    "mse": float(per_particle[chosen].mean()),
                }
            )
    return aggregate, per_view


def _difference_rows(
    per_view: list[dict[str, object]],
) -> list[dict[str, object]]:
    lookup = {
        (float(row["snr"]), int(row["view_id"]), str(row["method"])): float(
            row["mse"]
        )
        for row in per_view
    }
    return [
        {
            "snr": snr,
            "view_id": view,
            "two_sdr_minus_vector_pca_mse": (
                lookup[(snr, view, "two_sdr_tu_sure_hhi")]
                - lookup[(snr, view, "vector_pca_hhi")]
            ),
            "two_sdr_minus_full_core_mpca_mse": (
                lookup[(snr, view, "two_sdr_tu_sure_hhi")]
                - lookup[(snr, view, "mpca_tu_sure_full_core")]
            ),
        }
        for snr, view in sorted(
            {(key[0], key[1]) for key in lookup}
        )
    ]


def _montage(
    output_dir: Path,
    panels: dict[float, dict[str, np.ndarray]],
    view_ids: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = (
        ("clean", "Clean"),
        ("noisy", "Noisy"),
        ("vector_pca_hhi", "PCA"),
        ("mpca_tu_sure_full_core", "MPCA"),
        ("two_sdr_tu_sure_hhi", "2SDR"),
    )
    snr = 0.05 if 0.05 in panels else sorted(panels)[0]
    values = panels[snr]
    views = sorted(np.unique(view_ids))
    columns = 10
    groups = int(math.ceil(len(views) / columns))
    figure, axes = plt.subplots(
        groups * len(methods),
        columns,
        figsize=(16, 2.0 * groups * len(methods)),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for view_index, view in enumerate(views):
        group, column = divmod(view_index, columns)
        particle = int(np.flatnonzero(view_ids == view)[0])
        clean = values["clean"][particle]
        lower, upper = np.quantile(clean, (0.01, 0.99))
        for method_index, (method, label) in enumerate(methods):
            row = group * len(methods) + method_index
            axis = axes[row, column]
            axis.imshow(values[method][particle], cmap="gray", vmin=lower, vmax=upper)
            if method_index == 0:
                axis.set_title(f"view {int(view)}", fontsize=7)
            if column == 0:
                axis.set_ylabel(label, fontsize=7)
            axis.axis("off")
    figure.suptitle(f"All 50 view slots at SNR={snr:.2f}", fontsize=11)
    figure.tight_layout()
    figure.savefig(output_dir / "all_view_montage.png", dpi=150)
    plt.close(figure)

    selected_views = (0, 17, 34)
    selected_snrs = tuple(sorted(panels, reverse=True))
    missing = [
        view for view in selected_views if view not in set(np.unique(view_ids))
    ]
    if missing:
        raise ValueError(f"prespecified paper views are absent: {missing}")
    figure, axes = plt.subplots(
        len(methods),
        len(selected_snrs) * len(selected_views),
        figsize=(15.5, 7.2),
        squeeze=False,
    )
    for snr_index, current_snr in enumerate(selected_snrs):
        current = panels[current_snr]
        for view_index, view in enumerate(selected_views):
            column = snr_index * len(selected_views) + view_index
            particle = int(np.flatnonzero(view_ids == view)[0])
            clean = current["clean"][particle]
            lower, upper = np.quantile(clean, (0.01, 0.99))
            for row, (method, label) in enumerate(methods):
                axis = axes[row, column]
                axis.imshow(
                    current[method][particle],
                    cmap="gray",
                    vmin=lower,
                    vmax=upper,
                )
                if row == 0:
                    axis.set_title(
                        f"SNR={current_snr:.2f}\nview {view}",
                        fontsize=9,
                    )
                if column == 0:
                    axis.set_ylabel(label, fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
    for boundary in range(len(selected_views), axes.shape[1], len(selected_views)):
        axes[0, boundary].spines["left"].set_visible(True)
        axes[0, boundary].spines["left"].set_linewidth(1.5)
    figure.tight_layout()
    figure.savefig(
        output_dir / "paper_montage.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def _source_provenance(
    path: Path,
    protocol_path: Path,
    datasets: Iterable[SimpleNamespace],
    command: list[str],
) -> None:
    sources = (
        Path(__file__),
        protocol_path,
        HERE / "README.md",
        PACKAGE_ROOT / "fourier.py",
        PACKAGE_ROOT / "selection.py",
        PACKAGE_ROOT / "model.py",
        PACKAGE_ROOT / "io.py",
    )
    inputs = target_free_input_provenance(datasets)
    _json(
        path,
        {
            "command": {"argv": command, "shell": shlex.join(command)},
            "sources": {
                str(source): _sha256(source)
                for source in sources
            },
            "inputs": inputs,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "torch_threads": torch.get_num_threads(),
            },
        },
    )


def target_free_input_provenance(
    datasets: Iterable[SimpleNamespace],
) -> dict[str, object]:
    """Hash selection inputs while leaving clean-target bytes unopened."""
    output = {}
    for dataset in datasets:
        records = {
            name: {
                "path": str(dataset.paths[name]),
                "sha256": _sha256(dataset.paths[name]),
                "hash_timing": "before_selection",
            }
            for name in ("particles.mrcs", "manifest.json", "ground_truth.jsonl")
        }
        target = dataset.paths["clean_observed.mrcs"]
        records["clean_observed.mrcs"] = {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": None,
            "hash_timing": "after_all_rank_selections",
        }
        output[str(dataset.snr)] = records
    return output


def finalize_target_hashes(
    provenance_path: Path,
    datasets: Iterable[SimpleNamespace],
) -> None:
    """Add target hashes only after the rank-selection boundary."""
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    for dataset in datasets:
        record = payload["inputs"][str(dataset.snr)]["clean_observed.mrcs"]
        record["sha256"] = _sha256(dataset.paths["clean_observed.mrcs"])
        record["hash_completed_after_all_rank_selections"] = True
    _json(provenance_path, payload)


def run(
    attempt: Path,
    protocol: dict[str, object],
    protocol_path: Path,
    *,
    quick: bool,
) -> None:
    datasets = discover_datasets(protocol)
    command = [sys.executable, *sys.argv]
    _source_provenance(
        attempt / "provenance.json", protocol_path, datasets, command
    )
    mode = "quick" if quick else "formal"
    mode_spec = protocol["runtime_modes"][mode]
    selected_snrs = tuple(float(value) for value in mode_spec["snrs"])
    count = int(mode_spec["particles"])
    upper = tuple(int(value) for value in mode_spec["upper_envelope"])
    pca_max_rank = int(mode_spec["vector_pca_gic_max_rank"])
    max_iter = int(protocol["selection"]["mpca_max_iter"])
    tolerance = float(protocol["selection"]["mpca_tolerance"])
    by_snr = {float(dataset.snr): dataset for dataset in datasets}
    noisy_stacks = {
        snr: _mrc(dataset.paths["particles.mrcs"])
        for snr, dataset in by_snr.items()
    }
    base_variance = estimate_base_noise_variance(
        noisy_stacks[0.04],
        noisy_stacks[0.06],
        low_snr=0.04,
        high_snr=0.06,
        count=count,
    )
    logging.info(
        "noise variance: base=%g count=%d source=coupled noisy observations",
        base_variance,
        count,
    )
    _json(
        attempt / "noise_variance.json",
        {
            "base_variance": base_variance,
            "particle_count": count,
            "clean_target_used": False,
            "condition_variance": {
                str(snr): base_variance / snr for snr in selected_snrs
            },
        },
    )

    fitted_by_snr: dict[float, dict[str, object]] = {}
    selection_rows = []
    sure_rows = []
    gic_rows = []
    bases_payload = {}
    target_access_events = []
    for snr in selected_snrs:
        started = time.perf_counter()
        logging.info(
            "selection start SNR=%.2f n=%d upper=%dx%d PCA-max=%d",
            snr,
            count,
            upper[0],
            upper[1],
            pca_max_rank,
        )
        observed_spatial = np.asarray(
            noisy_stacks[snr][:count], dtype=np.float32
        )
        observed = images_to_q_coordinates(observed_spatial)
        fitted = fit_condition(
            observed,
            sigma2=base_variance / snr,
            upper=upper,
            pca_max_rank=pca_max_rank,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        runtime = time.perf_counter() - started
        fitted_by_snr[snr] = fitted
        row, condition_sure, condition_gic = _selection_rows(
            snr, fitted, sigma2=base_variance / snr, runtime=runtime
        )
        selection_rows.append(row)
        sure_rows.extend(condition_sure)
        gic_rows.extend(condition_gic)
        condition = f"snr{int(round(100 * snr)):03d}"
        for name in (
            "mean",
            "left",
            "right",
            "core_basis",
            "pca_sample_basis",
        ):
            bases_payload[f"{condition}_{name}"] = fitted[name]
        target_access_events.append(
            {
                "event": "selection_complete",
                "snr": snr,
                "p": fitted["selected_p"],
                "q": fitted["selected_q"],
                "r": fitted["selected_r"],
                "pca_r": fitted["pca_selected_r"],
                "seconds": runtime,
            }
        )
        logging.info(
            "selection complete SNR=%.2f ranks=(%d,%d,%d) PCA=%d seconds=%.1f",
            snr,
            fitted["selected_p"],
            fitted["selected_q"],
            fitted["selected_r"],
            fitted["pca_selected_r"],
            runtime,
        )
        del observed, observed_spatial

    if len(fitted_by_snr) != len(selected_snrs):
        raise RuntimeError("not every rank selection completed before target access")
    target_access_events.append(
        {
            "event": "all_rank_selections_complete",
            "snrs": list(selected_snrs),
        }
    )
    finalize_target_hashes(attempt / "provenance.json", datasets)
    target_access_events.append(
        {
            "event": "clean_target_hashes_complete",
            "snrs": [float(dataset.snr) for dataset in datasets],
        }
    )
    _csv(attempt / "outputs" / "selection.csv", selection_rows)
    _csv(attempt / "outputs" / "sure_candidates.csv", sure_rows)
    _csv(attempt / "outputs" / "gic_candidates.csv", gic_rows)
    np.savez_compressed(
        attempt / "outputs" / "selected_bases.npz", **bases_payload
    )

    metric_rows = []
    per_view_rows = []
    montage_panels = {}
    representative_snr = 0.05 if 0.05 in selected_snrs else selected_snrs[0]
    for snr in selected_snrs:
        target_access_events.append(
            {"event": "clean_target_open", "snr": snr}
        )
        logging.info("target access SNR=%.2f after all selections", snr)
        target = np.asarray(
            _mrc(by_snr[snr].paths["clean_observed.mrcs"])[:count],
            dtype=np.float32,
        )
        observed = np.asarray(noisy_stacks[snr][:count], dtype=np.float32)
        observed_q = images_to_q_coordinates(observed)
        reconstructions_q = reconstruct_methods(
            observed_q, fitted_by_snr[snr]
        )
        reconstructions = {
            method: q_coordinates_to_images(values)
            for method, values in reconstructions_q.items()
        }
        view_ids = np.asarray(
            [
                int(row["gt_view_id"])
                for row in by_snr[snr].records[:count]
            ],
            dtype=np.int64,
        )
        aggregate, per_view = _metric_rows(
            snr,
            reconstructions,
            target,
            view_ids,
            fitted_by_snr[snr],
        )
        metric_rows.extend(aggregate)
        per_view_rows.extend(per_view)
        montage_panels[snr] = {
            "clean": target,
            "noisy": observed,
            **reconstructions,
        }
    _csv(attempt / "outputs" / "metrics.csv", metric_rows)
    _csv(attempt / "outputs" / "per_view_metrics.csv", per_view_rows)
    differences = _difference_rows(per_view_rows)
    _csv(attempt / "outputs" / "per_view_differences.csv", differences)
    representative_view_ids = np.asarray(
        [
            int(row["gt_view_id"])
            for row in by_snr[representative_snr].records[:count]
        ],
        dtype=np.int64,
    )
    _montage(attempt / "outputs", montage_panels, representative_view_ids)
    _json(
        attempt / "target_access_audit.json",
        {
            "events": target_access_events,
            "target_used_for_selection": False,
            "first_target_open_after_all_rank_selections": True,
        },
    )
    metric_lookup = {
        (float(row["snr"]), str(row["method"])): float(row["mse"])
        for row in metric_rows
    }
    summary_conditions = []
    for row in selection_rows:
        snr = float(row["snr"])
        summary_conditions.append(
            {
                **row,
                "mse_2sdr": metric_lookup[(snr, "two_sdr_tu_sure_hhi")],
                "mse_pca": metric_lookup[(snr, "vector_pca_hhi")],
                "mse_mpca": metric_lookup[(snr, "mpca_tu_sure_full_core")],
                "mse_2sdr_minus_pca": (
                    metric_lookup[(snr, "two_sdr_tu_sure_hhi")]
                    - metric_lookup[(snr, "vector_pca_hhi")]
                ),
                "mse_2sdr_minus_mpca": (
                    metric_lookup[(snr, "two_sdr_tu_sure_hhi")]
                    - metric_lookup[(snr, "mpca_tu_sure_full_core")]
                ),
            }
        )
    _json(
        attempt / "outputs" / "summary.json",
        {
            "mode": mode,
            "scientific_scope": (
                "fixed-instance same-sample application; no held-out or "
                "generalization claim"
            ),
            "coordinate_execution": COORDINATE_EXECUTION,
            "conditions": summary_conditions,
            "negative_difference_means_lower_2sdr_mse": True,
            "right_censoring_is_diagnostic_not_a_rank_recovery_claim": True,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.quick == args.formal:
        raise SystemExit("choose exactly one of --quick or --formal")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["scientific_scope"]["coordinate_execution"] != (
        COORDINATE_EXECUTION
    ):
        raise SystemExit(
            "70S protocol must register the rfft2-then-block-algebra route"
        )
    attempt = args.output_root.resolve() / (
        "attempt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "outputs").mkdir()
    command = [sys.executable, *sys.argv]
    _json(
        attempt / "command.json",
        {"argv": command, "shell": shlex.join(command)},
    )
    _json(attempt / "protocol_snapshot.json", protocol)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(attempt / "run.log"),
            logging.StreamHandler(),
        ],
    )
    _json(attempt / "status.json", {"status": "running"})
    started = time.perf_counter()
    try:
        run(attempt, protocol, protocol_path, quick=args.quick)
    except Exception as error:
        logging.error("%s\n%s", error, traceback.format_exc())
        _json(
            attempt / "status.json",
            {
                "status": "failed",
                "error": repr(error),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise
    _json(
        attempt / "status.json",
        {
            "status": "success",
            "mode": "formal" if args.formal else "quick",
            "wall_seconds": time.perf_counter() - started,
        },
    )
    logging.info(
        "completed %s attempt at %s in %.1f seconds",
        "formal" if args.formal else "quick",
        attempt,
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
