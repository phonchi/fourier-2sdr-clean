#!/usr/bin/env python3
"""Colored-noise 70S study: Fourier whitening and corrected holdout selection.

The clean synthetic 70S stack receives separable periodic AR(1) noise.  That
noise is diagonal in the separable real Fourier coordinates, so known-spectrum
whitening is the row/column scaling ``Z -> Z / S`` (see
``fourier_2sdr.whitening``); the whitened stack then runs through exactly the
same-sample pipeline of ``experiments/seventy_s_same_sample`` (nested Tu SURE
Stage I, HHI GIC Stage II).  The transform is orthonormal, so pixel MSE equals
squared error in these coordinates and every loss is computed there.

Part A (no split, all 5,000 particles) compares whitened and unwhitened 2SDR,
PCA and full-core MPCA, an estimated-spectrum variant of whitened 2SDR, and the
mean-only baseline.  Part B splits the particles within each view into
fit/score/test parts (60/20/20) and uses the corrected holdout score to choose
among complete fitted estimators on the original pixel scale.  Clean targets
are used for evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fourier_2sdr.model import _core, _fit_spatial_subspaces  # noqa: E402
from fourier_2sdr.whitening import (  # noqa: E402
    affine_noise_trace,
    separable_noise_scale,
)

_spec = importlib.util.spec_from_file_location(
    "seventy_s_same_sample_runner",
    EXPERIMENTS / "seventy_s_same_sample" / "run_experiment.py",
)
white70s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(white70s)

_hhi = white70s.select_pca_rank_hhi_gic


class _Rank(dict):
    __getattr__ = dict.__getitem__


def _hhi_small_core(core, **kwargs):
    """A 1x1 core admits only r <= m - 1 = 0, i.e. the mean-only estimator."""
    if core.shape[1] < 2:
        return _Rank({"selected_rank": 0, "selected_at_upper_bound": True,
                      "note": "core dimension 1; r <= m-1 forces r = 0"})
    return _hhi(core, **kwargs)


# The same-sample runner calls select_pca_rank_hhi_gic from its module
# namespace; the small-core rule is installed there so that fit_condition
# applies it too.
white70s.select_pca_rank_hhi_gic = _hhi_small_core

SMOKE = {
    "rhos": [0.6],
    "snrs": [0.04],
    "realizations": 1,
    "particles": 1000,
    "selection": {"upper": [8, 8], "pca_max_rank": 20,
                  "split_grid": [4, 8], "split_pca_ranks": [5, 10, 20]},
}


def _csv_append(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def colored_noise(count: int, scale: np.ndarray, sigma2: float,
                  seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    white = generator.standard_normal((count, *scale.shape), dtype=np.float32)
    return white * np.float32(math.sqrt(sigma2)) * scale.astype(np.float32)


def estimate_separable_psd(noise_only: np.ndarray) -> np.ndarray:
    """Rank-one (separable) estimate of the per-coordinate noise variance."""
    psd = np.square(noise_only, dtype=np.float64).mean(axis=0)
    rows, cols = psd.mean(axis=1), psd.mean(axis=0)
    return np.outer(rows, cols) / psd.mean()


def pca_basis(centered: torch.Tensor, rank: int) -> torch.Tensor:
    """Out-of-sample PCA basis (dimension x rank) from a Gram eigensystem."""
    flat = centered.reshape(len(centered), -1)
    if rank == 0:
        return flat.new_zeros((flat.shape[1], 0))
    gram = (flat @ flat.mT).to(torch.float64)
    _, vectors = torch.linalg.eigh(gram)
    vectors = vectors[:, -rank:].flip(1)
    basis = flat.mT.to(torch.float64) @ vectors
    basis = basis / torch.linalg.vector_norm(basis, dim=0)
    return basis.to(flat.dtype)


def fit_pipeline(z: np.ndarray, sigma2: float, cfg: dict) -> dict[str, object]:
    """Same-sample selection on z, returning out-of-sample linear operators."""
    fitted = white70s.fit_condition(
        z, sigma2=sigma2, upper=tuple(cfg["upper"]),
        pca_max_rank=cfg["pca_max_rank"], max_iter=cfg["mpca_max_iter"],
        tolerance=cfg["mpca_tolerance"],
    )
    observed = torch.as_tensor(z, dtype=torch.float32)
    mean = torch.as_tensor(fitted["mean"])
    left = torch.as_tensor(fitted["left"])
    right = torch.as_tensor(fitted["right"])
    kron = white70s._ambient_tensor_basis(left, right)
    core_basis = torch.as_tensor(fitted["core_basis"])
    fitted["basis"] = {
        "2sdr": kron @ core_basis,
        "mpca": kron,
        "pca": pca_basis(observed - mean, int(fitted["pca_selected_r"])),
    }
    fitted["mean_t"] = mean.reshape(-1)
    return fitted


def apply_basis(z: torch.Tensor, mean: torch.Tensor,
                basis: torch.Tensor) -> torch.Tensor:
    """mean + V V^T (z - mean) in the coordinates in which V was fitted."""
    flat = z.reshape(len(z), -1) - mean
    if basis.shape[1] == 0:
        return mean.expand(len(z), -1).clone()
    return mean + (flat @ basis) @ basis.mT


def risk(estimate: torch.Tensor, clean: torch.Tensor,
         weight: torch.Tensor | None = None) -> float:
    diff = estimate - clean.reshape(len(clean), -1)
    if weight is not None:
        diff = diff * weight
    return float(diff.square().mean(dtype=torch.float64))


def split_sure(z: torch.Tensor, estimate: torch.Tensor, basis: torch.Tensor,
               omega: torch.Tensor) -> float:
    """Per-pixel corrected holdout score on the original scale.

    For f(z) = mu + S V V^T S^{-1} (z - mu) and noise covariance diag(omega),
    tr(diag(omega) H) = sum_k v_k^T diag(omega) v_k for every diagonal S.
    """
    residual = float((z.reshape(len(z), -1) - estimate).square()
                     .sum(dim=1, dtype=torch.float64).mean())
    trace_h = affine_noise_trace(basis, omega)
    trace_omega = float(omega.sum())
    return (residual + 2.0 * trace_h - trace_omega) / omega.numel()


def part_a(z: np.ndarray, clean_t: torch.Tensor, scale: np.ndarray,
           scale_est: np.ndarray, sigma2: float, cfg: dict) -> tuple[list, dict]:
    """No-split comparison on all particles; returns rows and selections."""
    # Whitening scales include sigma, so whitened noise has unit variance.
    s = torch.as_tensor(math.sqrt(sigma2) * scale.reshape(-1), dtype=torch.float32)
    s_est = torch.as_tensor(scale_est.reshape(-1), dtype=torch.float32)
    z_t = torch.as_tensor(z)
    rows: dict[str, float] = {}
    whitened_rows: dict[str, float] = {}
    selections: dict[str, object] = {}
    for label, scaling, tu_sigma2 in (
        ("white", s, 1.0), ("raw", None, sigma2), ("estpsd", s_est, 1.0)
    ):
        if scaling is None:
            zz = z
        else:
            zz = (z_t / scaling.reshape(z.shape[1:])).numpy()
        fitted = fit_pipeline(zz, tu_sigma2, cfg)
        selections[label] = {
            "p": fitted["selected_p"], "q": fitted["selected_q"],
            "r": fitted["selected_r"], "pca_r": fitted["pca_selected_r"],
            "r_at_bound": dict(fitted["core_gic"]).get("selected_at_upper_bound"),
            "pca_at_bound": fitted["pca_gic"]["selected_at_upper_bound"],
            "stage1_at_upper": fitted["stage1"]["selected_at_p_upper_bound"]
            or fitted["stage1"]["selected_at_q_upper_bound"],
        }
        methods = ("2sdr",) if label == "estpsd" else ("2sdr", "pca", "mpca")
        zz_t = torch.as_tensor(zz)
        for method in methods:
            est = apply_basis(zz_t, fitted["mean_t"], fitted["basis"][method])
            if scaling is not None:
                est = est * scaling
            rows[f"{label}_{method}"] = risk(est, clean_t)
            whitened_rows[f"{label}_{method}"] = risk(est, clean_t, 1.0 / s)
        del zz, zz_t
    mean_est = z_t.reshape(len(z_t), -1).mean(dim=0).expand(len(z_t), -1)
    rows["mean_only"] = risk(mean_est, clean_t)
    whitened_rows["mean_only"] = risk(mean_est, clean_t, 1.0 / s)
    return [rows, whitened_rows], selections


def part_b(z: np.ndarray, clean_t: torch.Tensor, scale: np.ndarray,
           sigma2: float, split: dict[str, np.ndarray], cfg: dict,
           s_full: np.ndarray | None = None) -> list:
    """Split selection of complete two-stage estimators on the pixel scale.

    On the fit split, one upper MPCA envelope is fitted in raw and in whitened
    coordinates.  Every nested (p,q) on the grid becomes a complete 2SDR
    estimator once HHI GIC on the fit-split core fixes r.  Vector PCA (HHI GIC
    rank and a grid of ranks), the full-core MPCA candidates and mean-only join
    the candidate set.  The corrected holdout score on the score split ranks
    all of them on the original pixel scale; the test split measures the
    realized risk.  With ``s_full`` the estimated spectrum replaces the known
    one both for whitening and in the score's noise covariance.
    """
    if s_full is None:
        s = torch.as_tensor(math.sqrt(sigma2) * scale.reshape(-1), dtype=torch.float32)
    else:
        s = torch.as_tensor(s_full.reshape(-1), dtype=torch.float32)
    omega = s.to(torch.float64) ** 2
    z_t = torch.as_tensor(z)
    shape = z.shape[1:]
    fit, score, test = (torch.as_tensor(split[k]) for k in ("fit", "score", "test"))
    grid = cfg["split_grid"]
    upper = max(grid)
    rows = []

    def evaluate(name, family, p, q, r, mean, basis, scaling):
        row = {"candidate": name, "family": family, "p": p, "q": q, "r": r,
               "dimension": int(basis.shape[1])}
        for part, index in (("score", score), ("test", test)):
            zz = z_t[index]
            inner = zz if scaling is None else zz / scaling.reshape(shape)
            est = apply_basis(inner, mean, basis)
            if scaling is not None:
                est = est * scaling
            row[f"{part}_true_risk"] = risk(est, clean_t[index])
            if part == "score":
                row["split_sure"] = split_sure(zz, est, basis, omega)
        rows.append(row)

    for label, scaling in (("white", s), ("raw", None)):
        zf = z_t[fit] if scaling is None else z_t[fit] / scaling.reshape(shape)
        mean = zf.mean(dim=0)
        centered = zf - mean
        left, right, _, _ = _fit_spatial_subspaces(
            (centered,), upper, upper, cfg["mpca_max_iter"], cfg["mpca_tolerance"])
        flat_mean = mean.reshape(-1)
        for p in grid:
            for q in grid:
                l, r_ = left[:, :p].contiguous(), right[:, :q].contiguous()
                kron = white70s._ambient_tensor_basis(l, r_)
                core = _core(centered, l, r_).reshape(len(centered), p * q)
                gic = white70s.select_pca_rank_hhi_gic(
                    core.to(torch.float64), min_rank=0, variant=white70s.HHI_GIC)
                rank = int(gic.selected_rank)
                basis = kron @ white70s._leading_core_basis(core, rank)
                evaluate(f"{label}_2sdr_p{p}_q{q}", f"{label}_2sdr", p, q, rank,
                         flat_mean, basis, scaling)
                evaluate(f"{label}_mpca_p{p}_q{q}", f"{label}_mpca", p, q, p * q,
                         flat_mean, kron, scaling)
        flat = centered.reshape(len(centered), -1)
        gram = (flat @ flat.mT).to(torch.float64) / len(flat)
        values = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0)
        full = torch.cat((values, values.new_zeros(flat.shape[1] - len(values))))
        pca_rank = int(white70s.select_hhi_from_eigenvalues(
            full, sample_count=len(flat), dimension=flat.shape[1],
            max_rank=cfg["pca_max_rank"])["selected_rank"])
        evaluate(f"{label}_pca", f"{label}_pca", 0, 0, pca_rank, flat_mean,
                 pca_basis(centered, pca_rank), scaling)
        # Vector PCA with r itself chosen by the corrected holdout score.
        top = pca_basis(centered, cfg["pca_max_rank"])
        for rank in cfg["split_pca_ranks"]:
            evaluate(f"{label}_pcasplit_r{rank}", f"{label}_pcasplit", 0, 0, rank,
                     flat_mean, top[:, :rank], scaling)
    mean_fit = z_t[fit].reshape(len(fit), -1).mean(dim=0)
    evaluate("mean_only", "mean_only", 0, 0, 0, mean_fit,
             mean_fit.new_zeros((mean_fit.numel(), 0)), None)
    return rows


def view_split(view_ids: np.ndarray, seed: int, fractions: dict) -> dict[str, np.ndarray]:
    """Fit/score/test split within every view."""
    generator = np.random.default_rng(seed)
    first = fractions["fit"]
    second = 1.0 - fractions["test"]
    parts: dict[str, list[np.ndarray]] = {"fit": [], "score": [], "test": []}
    for view in np.unique(view_ids):
        index = generator.permutation(np.flatnonzero(view_ids == view))
        n = len(index)
        a, b = int(round(first * n)), int(round(second * n))
        parts["fit"].append(index[:a])
        parts["score"].append(index[a:b])
        parts["test"].append(index[b:])
    return {k: np.sort(np.concatenate(v)) for k, v in parts.items()}


def montage(path: Path, clean: np.ndarray, z: np.ndarray, scale: np.ndarray,
            sigma2: float, cfg: dict, view_ids: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = torch.as_tensor(math.sqrt(sigma2) * scale.reshape(-1), dtype=torch.float32)
    z_t = torch.as_tensor(z)
    panels = {"Clean": torch.as_tensor(clean).reshape(len(clean), -1),
              "Colored noisy": z_t.reshape(len(z), -1)}
    raw = fit_pipeline(z, sigma2, cfg)
    panels["Unwhitened 2SDR"] = apply_basis(z_t, raw["mean_t"], raw["basis"]["2sdr"])
    zw = z_t / s.reshape(z.shape[1:])
    white = fit_pipeline(zw.numpy(), 1.0, cfg)
    for method, title in (("pca", "Whitened PCA"), ("mpca", "Whitened MPCA"),
                          ("2sdr", "Whitened 2SDR")):
        panels[title] = apply_basis(zw, white["mean_t"], white["basis"][method]) * s
    views = [v for v in (0, 17, 34) if np.any(view_ids == v)]
    picks = [int(np.flatnonzero(view_ids == v)[0]) for v in views]
    fig, axes = plt.subplots(len(panels), len(picks), squeeze=False,
                             figsize=(2.2 * len(picks), 2.2 * len(panels)))
    for row, (title, values) in enumerate(panels.items()):
        for col, index in enumerate(picks):
            image = white70s.q_coordinates_to_images(
                values[index].reshape(1, *z.shape[1:]).numpy())[0]
            ax = axes[row, col]
            ax.imshow(image, cmap="gray")
            ax.set_xticks([]), ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="an attempt-<UTC> directory is created inside it")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny configuration (1,000 particles, small grids) "
                             "for checking the installation; not a reported result")
    parser.add_argument("--rho", type=float, nargs="+")
    parser.add_argument("--realizations", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--snrs", type=float, nargs="+")
    parser.add_argument("--partb-estimated-psd", action="store_true",
                        help="Part B whitens and scores with the separable spectrum "
                             "estimated from noise-only images (test risk still uses "
                             "clean targets)")
    parser.add_argument("--skip-part-a", action="store_true")
    parser.add_argument("--skip-part-b", action="store_true")
    parser.add_argument("--skip-montage", action="store_true")
    parser.add_argument("--matching", choices=("marginal", "mahalanobis"),
                        default="mahalanobis")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cfg = dict(protocol["selection"])
    rhos = args.rho or protocol["noise"]["rho_primary"]
    realizations = args.realizations or protocol["realizations"]
    count = args.particles or protocol["dataset"]["particle_count"]
    snrs = args.snrs or protocol["dataset"]["snrs"]
    if args.smoke:
        cfg.update(SMOKE["selection"])
        rhos = args.rho or SMOKE["rhos"]
        snrs = args.snrs or SMOKE["snrs"]
        realizations = args.realizations or SMOKE["realizations"]
        count = args.particles or SMOKE["particles"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = args.output_root.resolve() / f"attempt-{stamp}"
    (attempt / "outputs").mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(attempt / "run.log"), logging.StreamHandler()])
    (attempt / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (attempt / "command.json").write_text(json.dumps(
        {"argv": [sys.executable, *sys.argv], "rhos": rhos, "snrs": snrs,
         "realizations": realizations, "particles": count, "smoke": args.smoke,
         "selection": cfg}, indent=2))

    datasets = white70s.discover_datasets(protocol)
    by_snr = {float(d.snr): d for d in datasets}
    stacks = {snr: white70s._mrc(d.paths["particles.mrcs"]) for snr, d in by_snr.items()}
    # Base noise level of the white reference, estimated without clean targets.
    base = white70s.estimate_base_noise_variance(
        stacks[0.04], stacks[0.06], low_snr=0.04, high_snr=0.06,
        count=protocol["dataset"]["particle_count"])
    logging.info("base noise variance of the white reference = %g", base)
    view_ids = np.asarray([int(r["gt_view_id"]) for r in by_snr[0.05].records])[:count]
    # The clean stack is identical across SNR; it generates data and scores.
    clean_spatial = np.asarray(
        white70s._mrc(by_snr[0.05].paths["clean_observed.mrcs"])[:count], dtype=np.float32)
    clean_q = white70s.images_to_q_coordinates(clean_spatial)
    clean_t = torch.as_tensor(clean_q)
    shape = clean_q.shape[1:]
    n_noise_only = int(protocol["noise"]["noise_only_images"])
    centered_energy = np.square(clean_q - clean_q.mean(axis=0), dtype=np.float64).mean(axis=0)
    fractions = protocol["part_b_split"]["fractions"]

    for rho in rhos:
        scale = separable_noise_scale(shape, rho)
        # kappa = tr(C^-1 Sigma_S)/tr(Sigma_S) matches the mean Mahalanobis
        # signal energy of the white reference (kappa = 1: marginal matching).
        kappa = (float((centered_energy / scale**2).sum() / centered_energy.sum())
                 if args.matching == "mahalanobis" else 1.0)
        logging.info("rho=%.1f matching=%s kappa=%.8f", rho, args.matching, kappa)
        for snr in snrs:
            sigma2 = base / snr * kappa
            for k in range(realizations):
                seed = int(protocol["seeds"]["noise_base"]) + 1000 * int(round(10 * rho)) \
                    + 100 * int(round(100 * snr)) + k
                started = time.perf_counter()
                z = clean_q + colored_noise(count, scale, sigma2, seed)
                noise_only = colored_noise(n_noise_only, scale, sigma2, seed + 500_000)
                scale_est = np.sqrt(estimate_separable_psd(noise_only))
                key = {"rho": rho, "white_reference_snr": snr, "kappa": kappa,
                       "sigma2": sigma2, "realization": k, "seed": seed}
                selections = {"white": "-", "raw": "-"}
                raw_rows = {"white_2sdr": float("nan"), "raw_2sdr": float("nan"),
                            "mean_only": float("nan")}
                if not args.skip_part_a:
                    (raw_rows, white_rows), selections = part_a(
                        z, clean_t, scale, scale_est, sigma2, cfg)
                    _csv_append(attempt / "outputs" / "part_a_pixel_mse.csv",
                                [{**key, **raw_rows}])
                    _csv_append(attempt / "outputs" / "part_a_whitened_mse.csv",
                                [{**key, **white_rows}])
                    _csv_append(attempt / "outputs" / "part_a_selections.csv",
                                [{**key, "variant": v, **sel} for v, sel in selections.items()])
                if not args.skip_part_b:
                    split = view_split(view_ids, seed, fractions)
                    rows = part_b(z, clean_t, scale, sigma2, split, cfg,
                                  s_full=scale_est if args.partb_estimated_psd else None)
                    _csv_append(attempt / "outputs" / "part_b_candidates.csv",
                                [{**key, **row} for row in rows])
                if (rho == rhos[0] and snr == 0.05 and k == 0
                        and not args.skip_part_a and not args.skip_montage):
                    montage(attempt / "outputs" / "montage.png", clean_q, z,
                            scale, sigma2, cfg, view_ids)
                logging.info(
                    "rho=%.1f snr=%.2f k=%d white=%s raw=%s %.0fs  mse white2sdr=%.4g "
                    "raw2sdr=%.4g mean=%.4g",
                    rho, snr, k, selections["white"], selections["raw"],
                    time.perf_counter() - started, raw_rows["white_2sdr"],
                    raw_rows["raw_2sdr"], raw_rows["mean_only"])
                del z, noise_only
    (attempt / "status.json").write_text(json.dumps({"status": "completed"}))
    logging.info("completed %s", attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
