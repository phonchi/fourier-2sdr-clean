"""Diagonal noise whitening in separable real Fourier coordinates.

Stationary noise on a periodic grid is diagonal in the real Fourier basis.  If
the noise is also separable, its variance at coordinate ``(k, l)`` is
``e_k f_l`` and whitening is the row/column scaling

    Z -> Z / S,    S[k, l] = sqrt(e_k f_l),

which keeps the matrix structure that 2SDR uses.  An orthogonal projector
``P = V V^T`` fitted in whitened coordinates acts on the original scale as the
affine map

    f(z) = mu + S P S^{-1} (z - mu),

whose linear part ``H = S P S^{-1}`` is idempotent and generally nonsymmetric
(it is symmetric exactly when ``P`` commutes with ``S^2``).  With a known diagonal noise
covariance ``Sigma = diag(omega)``, the corrected holdout score

    mean_i ||z_i - f(z_i)||^2 + 2 tr(Sigma H) - tr(Sigma)

is unbiased for the risk of ``f`` on the holdout sample.  Because ``S`` and
``Sigma`` are both diagonal, ``tr(Sigma S P S^{-1}) = tr(Sigma P) =
sum_j v_j^T Sigma v_j``, so the trace needs neither a dense ``D x D``
matrix nor the scaling itself.
"""

from __future__ import annotations

import numpy as np
import torch

Tensor = torch.Tensor

__all__ = [
    "affine_noise_trace",
    "corrected_holdout_score",
    "periodic_ar1_spectrum",
    "real_fourier_frequency_labels",
    "separable_noise_scale",
    "separable_periodic_ar1_spectrum",
    "whitened_projection",
    "whitened_projector_matrix",
]


def real_fourier_frequency_labels(size: int) -> np.ndarray:
    """Physical frequency of the real Fourier columns ``[DC, cos1, sin1, ..., Nyquist]``."""
    labels = [0]
    for frequency in range(1, (size - 1) // 2 + 1):
        labels.extend((frequency, frequency))
    if size % 2 == 0:
        labels.append(size // 2)
    result = np.asarray(labels)
    if len(result) != size:
        raise RuntimeError("real Fourier frequency-label dimension mismatch")
    return result


def periodic_ar1_spectrum(size: int, rho: float) -> np.ndarray:
    """Mean-one spectrum of periodic AR(1) noise on the real Fourier coordinates.

    The value at frequency ``k`` is proportional to
    ``(1 - rho^2) / (1 + rho^2 - 2 rho cos(2 pi k / size))``; the vector is
    normalized to mean one, so the marginal pixel variance is unchanged.
    """
    labels = real_fourier_frequency_labels(size)
    values = (1.0 - rho**2) / (
        1.0 + rho**2 - 2.0 * rho * np.cos(2 * np.pi * labels / size)
    )
    return values / values.mean()


def separable_periodic_ar1_spectrum(shape: tuple[int, int], rho: float) -> np.ndarray:
    """Per-coordinate variance ``e_k f_l`` of separable periodic AR(1) noise."""
    return np.outer(periodic_ar1_spectrum(shape[0], rho), periodic_ar1_spectrum(shape[1], rho))


def separable_noise_scale(shape: tuple[int, int], rho: float) -> np.ndarray:
    """Whitening scale ``S = sqrt(e_k f_l)`` of the mean-one separable AR(1) spectrum."""
    return np.sqrt(separable_periodic_ar1_spectrum(shape, rho))


def whitened_projection(
    observed: Tensor, mean: Tensor, basis: Tensor, scale: Tensor | None = None
) -> Tensor:
    """Apply ``f(z) = mean + S V V^T S^{-1} (z - mean)`` to vectorized rows.

    ``observed`` has shape ``(n, D)``, ``mean`` and ``scale`` have shape
    ``(D,)`` and ``basis`` is an orthonormal ``(D, r)`` matrix fitted to the
    whitened data.  ``scale=None`` gives the ordinary projection.  An empty
    basis returns the mean for every row.
    """
    if basis.shape[1] == 0:
        return mean.expand(len(observed), -1)
    inner = observed - mean if scale is None else (observed - mean) / scale
    out = (inner @ basis) @ basis.mT
    return mean + (out if scale is None else out * scale)


def whitened_projector_matrix(basis: Tensor, scale: Tensor | None = None) -> Tensor:
    """Dense linear part ``H = S V V^T S^{-1}`` (for small checks only)."""
    projector = basis @ basis.mT
    if scale is None:
        return projector
    return scale[:, None] * projector / scale[None, :]


def affine_noise_trace(basis: Tensor, noise_variance: Tensor) -> float:
    """``tr(diag(omega) S V V^T S^{-1}) = sum_j v_j^T diag(omega) v_j``.

    The value does not depend on the diagonal whitening scale ``S``.
    """
    if basis.shape[1] == 0:
        return 0.0
    return float((basis.square() * noise_variance[:, None]).sum())


def corrected_holdout_score(
    holdout: Tensor, estimate: Tensor, basis: Tensor, noise_variance: Tensor
) -> float:
    """Per-coordinate corrected holdout score of ``f(z) = a + S V V^T S^{-1} z``.

    ``holdout`` and ``estimate`` are ``(n, D)`` rows of independent holdout
    observations and ``f`` applied to them; ``basis`` is the orthonormal
    ``(D, r)`` basis defining ``f`` and ``noise_variance`` the diagonal
    ``omega`` of the known noise covariance on the scale of ``holdout``.
    Returns ``(mean ||z - f(z)||^2 + 2 tr(Sigma H) - tr(Sigma)) / D``.
    """
    residual = float(
        (holdout.reshape(len(holdout), -1) - estimate)
        .square()
        .sum(dim=1, dtype=torch.float64)
        .mean()
    )
    trace_h = affine_noise_trace(basis, noise_variance)
    trace_sigma = float(noise_variance.sum())
    return (residual + 2.0 * trace_h - trace_sigma) / noise_variance.numel()
