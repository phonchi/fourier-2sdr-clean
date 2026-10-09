#!/usr/bin/env python3
"""Calibration of the corrected holdout score with Fourier whitening.

Data are generated in whitened Fourier coordinates, ``z_w = signal + white
noise``, with four signal directions of variance 16 in a 4x4 Kronecker
envelope, and observed as ``z = S z_w`` where ``S^2`` is the separable periodic
AR(1) spectrum.  For ideal nested projectors the whitened risk is
``pq + 16[4 - min(p,q,4)]``, so ``(4,4)`` is the unique risk minimizer and
lies inside the grid.

Candidates are MPCA fits for every ``(p,q)`` on the grid, fitted either to the
whitened data (estimator ``S P S^{-1}``) or to the raw data (estimator
``P``), plus the mean-only estimator.  Every score is compared with the
analytic conditional risk on the original scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fourier_2sdr.model import _fit_spatial_subspaces  # noqa: E402
from fourier_2sdr.whitening import (  # noqa: E402
    affine_noise_trace,
    periodic_ar1_spectrum,
    whitened_projection,
)

SMOKE = {"replicates": 2, "grid": [1, 2, 4], "n_fit": 100, "n_holdout": 100}


def kron(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.einsum("pa,qb->pqab", left, right).reshape(
        left.shape[0] * right.shape[0], left.shape[1] * right.shape[1])


def fit_basis(centered: torch.Tensor, p: int, q: int, shape) -> torch.Tensor:
    """Orthonormal Kronecker MPCA basis (D x pq) fitted to centered rows."""
    left, right, _, _ = _fit_spatial_subspaces(
        (centered.reshape(len(centered), *shape).float(),), p, q, 250, 1e-6)
    return torch.linalg.qr(kron(left.double(), right.double())).Q


def tu_score(centered_w: torch.Tensor, basis: torch.Tensor, p: int, q: int,
             shape) -> float:
    """Per-coordinate Tu working score on whitened data (unit noise)."""
    n = len(centered_w)
    residual = float((centered_w - (centered_w @ basis) @ basis.mT)
                     .square().sum(dim=1).mean())
    height, width = shape
    df = height * width + (n - 1) * p * q  # eigengap terms are lower order
    return (residual + 2.0 / n * df - height * width) / (height * width)


def run(protocol: dict, rho: float, replicates: int, attempt: Path) -> None:
    shape = tuple(protocol["shape"])
    dim = shape[0] * shape[1]
    grid = protocol["grid"]
    n_fit, n_hold = protocol["n_fit"], protocol["n_holdout"]
    e = torch.as_tensor(periodic_ar1_spectrum(shape[0], rho), dtype=torch.float64)
    f = torch.as_tensor(periodic_ar1_spectrum(shape[1], rho), dtype=torch.float64)
    omega = (e[:, None] * f[None, :]).reshape(-1)  # original-scale noise variance
    s = omega.sqrt()
    geometry = torch.Generator().manual_seed(protocol["geometry_seed"])
    left = torch.linalg.qr(torch.randn(shape[0], 4, generator=geometry, dtype=torch.float64)).Q
    right = torch.linalg.qr(torch.randn(shape[1], 4, generator=geometry, dtype=torch.float64)).Q
    signal = kron(left, right)[:, [0, 5, 10, 15]]
    variance = torch.full((4,), float(protocol["signal_variance"]), dtype=torch.float64)
    sigma_w = (signal * variance) @ signal.mT  # whitened signal covariance
    sigma_x = s[:, None] * sigma_w * s[None, :]  # original-scale signal covariance
    identity = torch.eye(dim, dtype=torch.float64)

    def oracle(mean: torch.Tensor, basis: torch.Tensor, whitened: bool) -> float:
        """Analytic per-coordinate risk of f given the fitted mean and basis."""
        if basis.shape[1] == 0:
            h = torch.zeros_like(identity)
        else:
            h = basis @ basis.mT
            if whitened:
                h = s[:, None] * h / s[None, :]
        bias = (identity - h) @ mean
        diff = h - identity
        noise = (h * omega[None, :] * h).sum() if basis.shape[1] else 0.0
        total = float(bias.square().sum() + (diff @ sigma_x * diff).sum() + noise)
        return total / dim

    rows = []
    for rep in range(replicates):
        started = time.perf_counter()
        gen = torch.Generator().manual_seed(protocol["seed"] + 1000 * int(round(10 * rho)) + rep)

        def draw(n: int) -> torch.Tensor:
            scores = torch.randn(n, 4, generator=gen, dtype=torch.float64) * variance.sqrt()
            white = scores @ signal.mT + torch.randn(n, dim, generator=gen, dtype=torch.float64)
            return white * s  # observed on the original scale

        fit, hold = draw(n_fit), draw(n_hold)
        mean = fit.mean(dim=0)
        candidates = [("mean_only", "none", 0, 0, mean.new_zeros((dim, 0)), False)]
        for whitened in (True, False):
            data = (fit - mean) / s if whitened else fit - mean
            for p in grid:
                for q in grid:
                    basis = fit_basis(data, p, q, shape)
                    tag = "white" if whitened else "raw"
                    candidates.append((f"{tag}_p{p}_q{q}", tag, p, q, basis, whitened))
        trace_sigma = float(omega.sum())
        for label, family, p, q, basis, whitened in candidates:
            scaling = s if whitened else None
            trace_h = affine_noise_trace(basis, omega)
            holdout_residual = float(
                (hold - whitened_projection(hold, mean, basis, scaling))
                .square().sum(dim=1).mean())
            fit_residual = float(
                (fit - whitened_projection(fit, mean, basis, scaling))
                .square().sum(dim=1).mean())
            rows.append({
                "rho": rho, "replicate": rep, "candidate": label, "family": family,
                "p": p, "q": q, "dimension": int(basis.shape[1]),
                "oracle_risk": oracle(mean, basis, whitened),
                "corrected_split": (holdout_residual + 2 * trace_h - trace_sigma) / dim,
                "uncorrected_holdout": holdout_residual / dim,
                "fit_sample_plugin": (fit_residual + 2 * trace_h - trace_sigma) / dim,
                "tu_whitened": (tu_score((fit - mean) / s, basis, p, q, shape)
                                if whitened else float("nan")),
            })
        logging.info("rho=%.1f rep=%d %.1fs", rho, rep, time.perf_counter() - started)
        if rep % 10 == 9 or rep == replicates - 1:
            with (attempt / "outputs" / "candidate_scores.csv").open("w", newline="") as h:
                writer = csv.DictWriter(h, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="an attempt-<UTC> directory is created inside it")
    parser.add_argument("--rho", type=float, default=None,
                        help="noise correlation (default: the protocol's rho_primary)")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny configuration for checking the installation")
    parser.add_argument("--threads", type=int, default=8,
                        help="torch CPU threads (the reported runs used 8)")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.smoke:
        protocol.update({k: v for k, v in SMOKE.items() if k != "replicates"})
    rho = args.rho if args.rho is not None else protocol["rho_primary"]
    replicates = args.replicates or (SMOKE["replicates"] if args.smoke
                                     else protocol["replicates"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = args.output_root.resolve() / f"attempt-{stamp}"
    (attempt / "outputs").mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(attempt / "run.log"),
                                  logging.StreamHandler()])
    (attempt / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (attempt / "command.json").write_text(json.dumps(
        {"argv": [sys.executable, *sys.argv], "rho": rho, "replicates": replicates,
         "smoke": args.smoke, "threads": args.threads}, indent=2))
    torch.set_num_threads(args.threads)
    run(protocol, rho, replicates, attempt)
    (attempt / "status.json").write_text(json.dumps({"status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
