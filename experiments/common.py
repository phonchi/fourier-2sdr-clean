"""Shared helpers for the paper's selection experiments.

Candidate
subspaces are fitted only on ``fit_images``.  SURE and GIC consume independent
observations, and refitting uses all non-test observations only after both
dimensions have been selected.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from fourier_2sdr.fourier import (
    _from_valid_rfft2,
    real_fourier_basis as minimal_real_fourier_basis,
    real_fourier_inverse as separable_real_fourier_inverse,
)
from fourier_2sdr.selection import (
    AffineEstimator,
    hhi_rank_upper_bound,
    restrict_kronecker_basis,
    select_pca_rank_hhi_gic,
    select_split_sure,
)
from fourier_2sdr.model import _fit_spatial_subspaces


def _tensor(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float64)
    array = np.asarray(value)
    if not array.flags.writeable:
        array = array.copy()
    return torch.as_tensor(array, dtype=torch.float64)


def circular_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    return ((yy - (height - 1) / 2) ** 2 + (xx - (width - 1) / 2) ** 2
            <= (min(shape) / 2) ** 2)


def real_fourier_frequency_labels(size: int) -> np.ndarray:
    """Physical frequency of columns [DC, cos1, sin1, ..., Nyquist]."""

    labels = [0]
    for frequency in range(1, (size - 1) // 2 + 1):
        labels.extend((frequency, frequency))
    if size % 2 == 0:
        labels.append(size // 2)
    result = np.asarray(labels)
    if len(result) != size:
        raise RuntimeError("real Fourier frequency-label dimension mismatch")
    return result


def radial_frequency_mask(shape: tuple[int, int], radius: float | None) -> np.ndarray:
    if radius is None:
        return np.ones(shape, dtype=bool)
    height, width = shape
    fy = real_fourier_frequency_labels(height)
    fx = real_fourier_frequency_labels(width)
    return (fy[:, None] ** 2 + fx[None, :] ** 2) <= radius**2


def periodic_ar1_eigenvalues(size: int, rho: float) -> np.ndarray:
    labels = real_fourier_frequency_labels(size)
    values = (1.0 - rho**2) / (
        1.0 + rho**2 - 2.0 * rho * np.cos(2 * np.pi * labels / size)
    )
    return values / values.mean()


def separable_periodic_ar1_covariance(
    shape: tuple[int, int], rho: float, sigma2: float = 1.0
) -> np.ndarray:
    left, right = fourier_matrices(shape)
    ey = torch.as_tensor(periodic_ar1_eigenvalues(shape[0], rho), dtype=torch.float64)
    ex = torch.as_tensor(periodic_ar1_eigenvalues(shape[1], rho), dtype=torch.float64)
    cy = (left * ey) @ left.mT
    cx = (right * ex) @ right.mT
    return (sigma2 * torch.kron(cy, cx)).numpy()


def fourier_matrices(shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        minimal_real_fourier_basis(shape[0], dtype=torch.float64),
        minimal_real_fourier_basis(shape[1], dtype=torch.float64),
    )


def transform_images(
    images: np.ndarray | torch.Tensor,
    *,
    spatial_mask: np.ndarray | torch.Tensor | None = None,
    frequency_mask: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = _tensor(images)
    if values.ndim != 3:
        raise ValueError("images must have shape (n,height,width)")
    if spatial_mask is not None:
        values = values * _tensor(spatial_mask)
    values = values.contiguous()
    shape = tuple(values.shape[-2:])
    # Production image code normally starts from rfft2.  Because Q_d.T equals
    # C_d F_d, the fixed conjugate-pair blocks below finish the square
    # separable real Fourier matrix of the paper with no further transform
    # and without constructing an HW-by-HW operator.
    if values.shape[0] == 0:
        # Some selection workflows deliberately defer test access by passing
        # an empty batch.  MKL's FFT backend rejects zero-batch transforms.
        coefficients = values.clone()
    else:
        half_spectra = torch.fft.rfft2(values, norm="ortho")
        coefficients = _from_valid_rfft2(
            half_spectra, spatial_shape=shape
        )
    mask = (
        torch.ones(values.shape[-2:], dtype=torch.bool)
        if frequency_mask is None
        else torch.as_tensor(frequency_mask, dtype=torch.bool)
    )
    return coefficients, coefficients[:, mask]


def inverse_observations(
    observations: np.ndarray | torch.Tensor,
    shape: tuple[int, int],
    *,
    frequency_mask: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Embed observable real-Fourier coordinates and invert to spatial images."""

    values = _tensor(observations)
    mask = (
        torch.ones(shape, dtype=torch.bool)
        if frequency_mask is None
        else torch.as_tensor(frequency_mask, dtype=torch.bool)
    )
    if values.ndim != 2 or values.shape[1] != int(mask.sum()):
        raise ValueError("observations must have shape (n, number of retained coordinates)")
    coefficients = torch.zeros(
        (len(values), *shape), dtype=values.dtype, device=values.device
    )
    coefficients[:, mask] = values
    return separable_real_fourier_inverse(coefficients)


def standard_ssim(
    estimate: np.ndarray, target: np.ndarray, *, data_range: float
) -> float:
    """Windowed SSIM with the standard 11-pixel Gaussian, sigma 1.5."""

    x = np.asarray(estimate, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or min(x.shape) < 11:
        raise ValueError("SSIM inputs must be matching images of at least 11x11")
    sigma, radius = 1.5, 5
    truncate = radius / sigma
    ux = gaussian_filter(x, sigma=sigma, truncate=truncate, mode="reflect")
    uy = gaussian_filter(y, sigma=sigma, truncate=truncate, mode="reflect")
    uxx = gaussian_filter(x * x, sigma=sigma, truncate=truncate, mode="reflect")
    uyy = gaussian_filter(y * y, sigma=sigma, truncate=truncate, mode="reflect")
    uxy = gaussian_filter(x * y, sigma=sigma, truncate=truncate, mode="reflect")
    vx = np.maximum(uxx - ux * ux, 0.0)
    vy = np.maximum(uyy - uy * uy, 0.0)
    vxy = uxy - ux * uy
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = (
        (2 * ux * uy + c1) * (2 * vxy + c2)
        / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    )
    return float(score[radius:-radius, radius:-radius].mean())


def observation_covariance(
    shape: tuple[int, int],
    sigma2: float,
    *,
    spatial_mask: np.ndarray | None = None,
    frequency_mask: np.ndarray | None = None,
    spatial_covariance: np.ndarray | None = None,
) -> torch.Tensor | float:
    """Return covariance in selected separable real-Fourier coordinates."""

    if sigma2 < 0:
        raise ValueError("sigma2 must be nonnegative")
    mask = (
        np.ones(shape, dtype=bool) if frequency_mask is None
        else np.asarray(frequency_mask, dtype=bool)
    )
    if spatial_mask is None and spatial_covariance is None:
        return float(sigma2)
    left, right = fourier_matrices(shape)
    # Rows are spatial pixels; columns are real Fourier coordinates.
    transform = torch.einsum("pa,qb->pqab", left, right).reshape(
        shape[0] * shape[1], shape[0] * shape[1]
    )
    transform = transform[:, torch.as_tensor(mask.reshape(-1))]
    if spatial_covariance is not None:
        covariance = _tensor(spatial_covariance)
        if covariance.shape != (shape[0] * shape[1],) * 2:
            raise ValueError("spatial_covariance has wrong shape")
        if spatial_mask is not None:
            weights = _tensor(np.asarray(spatial_mask, dtype=float).reshape(-1))
            covariance = weights[:, None] * covariance * weights[None, :]
        return transform.mT @ covariance @ transform
    weights = _tensor(np.asarray(spatial_mask, dtype=float).reshape(-1)) * sigma2
    return transform.mT @ (weights[:, None] * transform)


def _fit_candidate(
    coefficient_images: torch.Tensor,
    frequency_mask: torch.Tensor,
    p_rank: int,
    q_rank: int,
) -> tuple[AffineEstimator, torch.Tensor]:
    observable_images = coefficient_images.clone()
    observable_images[:, ~frequency_mask] = 0
    centered = observable_images - observable_images.mean(dim=0)
    left, right, _, _ = _fit_spatial_subspaces(
        (centered,), p_rank, q_rank, 30, 1e-6
    )
    restricted = restrict_kronecker_basis(
        left, right, observable_mask=frequency_mask
    )
    mean = observable_images[:, frequency_mask].mean(dim=0)
    estimator = AffineEstimator(
        label=f"p{p_rank}_q{q_rank}",
        mean=mean,
        basis=restricted.basis,
        effective_dimension=restricted.effective_dimension,
        p_rank=p_rank,
        q_rank=q_rank,
    )
    return estimator, restricted.basis


def serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return serializable(dataclasses.asdict(value))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def select_and_reconstruct(
    fit_images: np.ndarray | torch.Tensor,
    sure_images: np.ndarray | torch.Tensor,
    gic_images: np.ndarray | torch.Tensor,
    test_images: np.ndarray | torch.Tensor | Any,
    *,
    candidates: Sequence[tuple[int, int]],
    covariance: torch.Tensor | float,
    spatial_mask: np.ndarray | None = None,
    frequency_mask: np.ndarray | None = None,
    coordinate_system: str = "fourier",
    trace_method: str = "exact",
    probe_seed: int = 0,
) -> dict[str, Any]:
    """Select dimensions and reconstruct test observations without targets."""

    shape = tuple(np.asarray(fit_images).shape[-2:])
    if coordinate_system not in {"fourier", "spatial"}:
        raise ValueError("coordinate_system must be 'fourier' or 'spatial'")
    freq = (
        np.ones(shape, dtype=bool) if frequency_mask is None
        else np.asarray(frequency_mask, dtype=bool)
    )
    def coordinate_transform(images: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if coordinate_system == "fourier":
            return transform_images(
                images,
                spatial_mask=spatial_mask,
                frequency_mask=freq,
            )
        values = _tensor(images)
        if spatial_mask is not None:
            values = values * _tensor(spatial_mask)
        return values, values[:, torch.as_tensor(freq, dtype=torch.bool)]

    fit_coeff, _ = coordinate_transform(fit_images)
    _, sure_obs = coordinate_transform(sure_images)
    _, gic_obs = coordinate_transform(gic_images)
    freq_tensor = torch.as_tensor(freq, dtype=torch.bool)
    trained = [_fit_candidate(fit_coeff, freq_tensor, p, q)[0]
               for p, q in candidates]
    if trace_method != "exact":
        raise ValueError("the clean package supports exact holdout traces only")
    adaptive = None
    sure = select_split_sure(
        sure_obs, trained, covariance, trace_method=trace_method,
        num_probes=128, seed=probe_seed,
    )
    selected = next(item for item in trained if item.label == sure.selected_label)
    gic_cores_raw = (gic_obs - selected.mean) @ selected.basis
    gic_core_mean = gic_cores_raw.mean(dim=0)
    gic_cores = gic_cores_raw - gic_core_mean
    dimension = selected.effective_dimension or selected.basis.shape[1]
    upper = hhi_rank_upper_bound(len(gic_cores), dimension)
    gic = select_pca_rank_hhi_gic(
        gic_cores, min_rank=0, max_rank=upper, variant="hhi_2_over_n"
    )
    gic_sensitivity = select_pca_rank_hhi_gic(
        gic_cores, min_rank=0, max_rank=upper,
        variant="two_sdr_log_n_over_n_sensitivity",
    )

    # Test observations remain untouched until both dimensions are fixed.
    if callable(test_images):
        test_images = test_images()
    _, test_obs = coordinate_transform(test_images)

    rank = int(gic.selected_rank)
    if rank:
        _, _, frozen_vh = torch.linalg.svd(gic_cores, full_matrices=False)
        frozen_core_basis = frozen_vh[:rank].mT
        frozen_test_core = (test_obs - selected.mean) @ selected.basis
        frozen_test_core = (
            gic_core_mean
            + (
                (frozen_test_core - gic_core_mean) @ frozen_core_basis
            ) @ frozen_core_basis.mT
        )
        frozen_reconstructed = (
            selected.mean + frozen_test_core @ selected.basis.mT
        )
    else:
        frozen_core_basis = torch.empty(
            (selected.basis.shape[1], 0),
            dtype=selected.basis.dtype,
            device=selected.basis.device,
        )
        frozen_reconstructed = (
            selected.mean + gic_core_mean @ selected.basis.mT
        ).expand_as(test_obs)

    # Refit the selected Stage-I subspace and Stage-II PCA without test data.
    non_test = torch.cat((_tensor(fit_images), _tensor(sure_images), _tensor(gic_images)))
    non_test_coeff, non_test_obs = coordinate_transform(non_test)
    refit, basis = _fit_candidate(
        non_test_coeff, freq_tensor, int(selected.p_rank), int(selected.q_rank)
    )
    cores_raw = (non_test_obs - refit.mean) @ basis
    refit_core_mean = cores_raw.mean(dim=0)
    cores = cores_raw - refit_core_mean
    if rank:
        _, _, vh = torch.linalg.svd(cores, full_matrices=False)
        core_basis = vh[:rank].mT
        test_core = (test_obs - refit.mean) @ basis
        test_core = (
            refit_core_mean
            + ((test_core - refit_core_mean) @ core_basis) @ core_basis.mT
        )
        reconstructed = refit.mean + test_core @ basis.mT
    else:
        core_basis = torch.empty((basis.shape[1], 0), dtype=basis.dtype)
        reconstructed = (
            refit.mean + refit_core_mean @ basis.mT
        ).expand_as(test_obs)

    frozen_final_basis = selected.basis @ frozen_core_basis
    refit_final_basis = basis @ core_basis
    frozen_projector = frozen_final_basis @ frozen_final_basis.mT
    refit_projector = refit_final_basis @ refit_final_basis.mT
    projector_denominator = max(float(rank) ** 0.5, 1.0)
    projector_distance = float(
        torch.linalg.matrix_norm(frozen_projector - refit_projector)
        / projector_denominator
    )
    frozen_affine_mean = selected.mean + gic_core_mean @ selected.basis.mT
    refit_affine_mean = refit.mean + refit_core_mean @ basis.mT
    affine_mean_relative_difference = float(
        torch.linalg.vector_norm(frozen_affine_mean - refit_affine_mean)
        / torch.linalg.vector_norm(refit_affine_mean).clamp_min(
            torch.finfo(refit_affine_mean.dtype).eps
        )
    )
    prediction_relative_difference = float(
        torch.linalg.vector_norm(frozen_reconstructed - reconstructed)
        / torch.linalg.vector_norm(reconstructed).clamp_min(
            torch.finfo(reconstructed.dtype).eps
        )
    )
    return {
        "selected_p": int(selected.p_rank),
        "selected_q": int(selected.q_rank),
        "selected_r": rank,
        "selected_r_log_n_sensitivity": int(gic_sensitivity.selected_rank),
        "sure": serializable(sure),
        "sure_adaptive": serializable(adaptive) if adaptive is not None else None,
        "gic": serializable(gic),
        "gic_sensitivity": serializable(gic_sensitivity),
        "test_observations": test_obs.detach().cpu().numpy(),
        "frozen_test_reconstructions": frozen_reconstructed.detach().cpu().numpy(),
        "postselection_refit_test_reconstructions": reconstructed.detach().cpu().numpy(),
        "test_reconstructions_semantics": "postselection_refit_legacy_alias",
        "test_reconstructions": reconstructed.detach().cpu().numpy(),
        "frozen_mean": selected.mean.detach().cpu().numpy(),
        "frozen_gic_core_mean": gic_core_mean.detach().cpu().numpy(),
        "frozen_stage1_basis": selected.basis.detach().cpu().numpy(),
        "frozen_stage2_basis": frozen_core_basis.detach().cpu().numpy(),
        "refit_mean": refit.mean.detach().cpu().numpy(),
        "refit_core_mean": refit_core_mean.detach().cpu().numpy(),
        "refit_stage1_basis": basis.detach().cpu().numpy(),
        "refit_stage2_basis": core_basis.detach().cpu().numpy(),
        "refit_stability": {
            "final_projector_distance": projector_distance,
            "affine_mean_relative_difference": affine_mean_relative_difference,
            "test_prediction_relative_difference": prediction_relative_difference,
        },
    }
