"""Alternating MPCA and fixed-rank 2SDR fits."""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


def _validate_matrices(values: Tensor) -> None:
    if not isinstance(values, Tensor):
        raise TypeError("coefficients must be a torch.Tensor")
    if values.ndim != 3:
        raise ValueError("coefficients must have shape (n, height, width)")
    if values.is_complex() or not values.is_floating_point():
        raise ValueError("coefficients must be real floating-point values")
    if len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("coefficients need at least two finite observations")


def _leading_eigenvectors(covariance: Tensor, rank: int) -> Tensor:
    _, vectors = torch.linalg.eigh(covariance)
    return vectors[:, -rank:].flip(1).contiguous()


def _projector_distance(new: Tensor, old: Tensor) -> Tensor:
    return torch.linalg.matrix_norm(new @ new.mT - old @ old.mT) / (2 * new.shape[1]) ** 0.5


def _fit_spatial_subspaces(
    centered: Tensor | tuple[Tensor, ...],
    p_rank: int,
    q_rank: int,
    max_iter: int,
    tol: float,
) -> tuple[Tensor, Tensor, int, bool]:
    if isinstance(centered, tuple):
        if len(centered) != 1:
            raise ValueError("the fit accepts one matrix channel")
        centered = centered[0]
    covariance_q = torch.einsum("npq,npr->qr", centered, centered)
    previous_p = previous_q = None
    converged = False
    for iteration in range(1, max_iter + 1):
        right = _leading_eigenvectors(covariance_q, q_rank)
        projected_q = torch.einsum("npq,qb->npb", centered, right)
        covariance_p = torch.einsum("npb,nrb->pr", projected_q, projected_q)
        left = _leading_eigenvectors(covariance_p, p_rank)
        projected_p = torch.einsum("pa,npq->naq", left, centered)
        covariance_q = torch.einsum("naq,nar->qr", projected_p, projected_p)
        if previous_p is not None:
            distance = torch.maximum(
                _projector_distance(left, previous_p),
                _projector_distance(right, previous_q),
            )
            if float(distance) <= tol:
                converged = True
                break
        previous_p, previous_q = left, right
    return left, right, iteration, converged


def _core(values: Tensor, left: Tensor, right: Tensor) -> Tensor:
    return torch.einsum("pa,npq,qb->nab", left, values, right)


def _ambient_basis(left: Tensor, right: Tensor) -> Tensor:
    return torch.einsum("pa,qb->pqab", left, right).reshape(
        left.shape[0] * right.shape[0], left.shape[1] * right.shape[1]
    )


@dataclass(frozen=True)
class TwoSDRFit:
    """A fitted affine 2SDR reconstruction in real Fourier coordinates."""

    mean: Tensor
    row_basis: Tensor
    column_basis: Tensor
    core_mean: Tensor
    core_basis: Tensor
    iterations: int
    converged: bool

    @property
    def ranks(self) -> tuple[int, int, int]:
        return (
            self.row_basis.shape[1],
            self.column_basis.shape[1],
            self.core_basis.shape[1],
        )

    @property
    def ambient_basis(self) -> Tensor:
        return _ambient_basis(self.row_basis, self.column_basis) @ self.core_basis

    def transform(self, coefficients: Tensor) -> Tensor:
        centered = coefficients - self.mean
        reduced = _core(centered, self.row_basis, self.column_basis).reshape(len(coefficients), -1)
        return (reduced - self.core_mean) @ self.core_basis

    def reconstruct(self, coefficients: Tensor) -> Tensor:
        centered = coefficients - self.mean
        reduced = _core(centered, self.row_basis, self.column_basis).reshape(len(coefficients), -1)
        restored = self.core_mean + ((reduced - self.core_mean) @ self.core_basis) @ self.core_basis.mT
        restored = restored.reshape(len(coefficients), *self.ranks[:2])
        return self.mean + torch.einsum(
            "pa,nab,qb->npq", self.row_basis, restored, self.column_basis
        )


def _make_fit(
    coefficients: Tensor,
    left: Tensor,
    right: Tensor,
    iterations: int,
    converged: bool,
    core_rank: int,
) -> TwoSDRFit:
    mean = coefficients.mean(dim=0)
    centered = coefficients - mean
    cores = _core(centered, left, right).reshape(len(coefficients), -1)
    core_mean = cores.mean(dim=0)
    centered_cores = cores - core_mean
    if core_rank:
        _, _, vh = torch.linalg.svd(centered_cores, full_matrices=False)
        core_basis = vh[:core_rank].mT.contiguous()
    else:
        core_basis = coefficients.new_empty((left.shape[1] * right.shape[1], 0))
    return TwoSDRFit(mean, left, right, core_mean, core_basis, iterations, converged)


@torch.no_grad()
def fit_2sdr(
    coefficients: Tensor,
    ranks: tuple[int, int, int],
    *,
    max_iter: int = 250,
    tol: float = 1e-5,
) -> TwoSDRFit:
    """Fit row, column, and core subspaces at fixed ranks ``(p, q, r)``."""
    _validate_matrices(coefficients)
    p_rank, q_rank, core_rank = map(int, ranks)
    n, height, width = coefficients.shape
    if not (0 < p_rank < height and 0 < q_rank < width):
        raise ValueError("row and column ranks must be strictly inside the matrix dimensions")
    if not 0 <= core_rank <= min(n - 1, p_rank * q_rank):
        raise ValueError("core rank is outside the centered-core range")
    if max_iter <= 0 or tol < 0:
        raise ValueError("max_iter must be positive and tol nonnegative")
    centered = coefficients - coefficients.mean(dim=0)
    left, right, iterations, converged = _fit_spatial_subspaces(
        centered, p_rank, q_rank, max_iter, tol
    )
    return _make_fit(coefficients, left, right, iterations, converged, core_rank)
