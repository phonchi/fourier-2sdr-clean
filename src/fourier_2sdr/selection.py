"""The two rank-selection routes reported in the Fourier 2SDR paper."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Mapping, Sequence

import torch

from .model import (
    TwoSDRFit,
    _ambient_basis,
    _core,
    _fit_spatial_subspaces,
    _make_fit,
    _validate_matrices,
)

Tensor = torch.Tensor
HHI_GIC = "hhi_2_over_n"
TWO_SDR_GIC_SENSITIVITY = "two_sdr_log_n_over_n_sensitivity"


@dataclass(frozen=True)
class Score:
    value: float
    residual: float | None = None
    correction: float | None = None
    degrees_of_freedom: float | None = None


@dataclass(frozen=True)
class RankSelection:
    """Selected fit plus the diagnostics needed to interpret its ranks."""

    fit: TwoSDRFit
    stage1_scores: Mapping[tuple[int, int], Score]
    core_scores: Mapping[int, Score]
    stage1_right_censored: tuple[bool, bool]
    core_right_censored: bool
    core_rank_ambiguous: bool
    tied_core_ranks: tuple[int, ...]
    method: str

    @property
    def ranks(self) -> tuple[int, int, int]:
        return self.fit.ranks


@dataclass(frozen=True)
class AffineEstimator:
    """A fixed affine estimator ``mean + H(z-mean)``."""

    label: str
    mean: Tensor
    basis: Tensor
    effective_dimension: int | None = None
    p_rank: int | None = None
    q_rank: int | None = None

    def apply_linear_rows(self, rows: Tensor) -> Tensor:
        if self.mean.ndim != 1 or self.basis.ndim != 2:
            raise ValueError("mean and basis must be one- and two-dimensional")
        if self.basis.shape[0] != len(self.mean) or rows.shape[1] != len(self.mean):
            raise ValueError("estimator dimensions do not match observations")
        return (rows @ self.basis) @ self.basis.mT


@dataclass(frozen=True)
class RestrictedKroneckerBasis:
    basis: Tensor
    effective_dimension: int


def restrict_kronecker_basis(
    left: Tensor,
    right: Tensor,
    *,
    observable_mask: Tensor | None = None,
    rtol: float | None = None,
) -> RestrictedKroneckerBasis:
    """Restrict ``kron(left, right)`` and return an orthonormal image basis."""
    raw = _ambient_basis(left, right)
    restricted = raw if observable_mask is None else raw[observable_mask.reshape(-1)]
    if restricted.shape[0] == 0:
        raise ValueError("the observable space must not be empty")
    u, singular_values, _ = torch.linalg.svd(restricted, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.numel() else 0.0
    relative = max(restricted.shape) * torch.finfo(restricted.dtype).eps if rtol is None else rtol
    rank = int((singular_values > relative * largest).sum())
    return RestrictedKroneckerBasis(u[:, :rank], rank)


@dataclass(frozen=True)
class SplitSUREScore:
    label: str
    score: float
    residual: float
    correction: float
    trace_covariance_operator: float
    trace_covariance: float
    correction_mcse: float
    effective_dimension: int
    p_rank: int | None
    q_rank: int | None


@dataclass(frozen=True)
class SplitSURESelection:
    selected_label: str
    scores: Mapping[str, SplitSUREScore]
    rank_ambiguous: bool
    ambiguous_labels: tuple[str, ...]
    trace_method: str
    num_probes: int
    seed: int
    tie_tolerance: float


@dataclass(frozen=True)
class GICScore:
    rank: int
    score: float
    log_determinant: float
    bias: float
    penalty_weight: float


@dataclass(frozen=True)
class GICSelection:
    selected_rank: int
    scores: Mapping[int, GICScore]
    eigenvalues: Tensor
    rank_upper_bound: int
    admissible_upper_bound: int
    search_upper_bound: int
    variant: str
    used_sample_gram: bool
    rank_ambiguous: bool
    tied_ranks: tuple[int, ...]
    selected_at_upper_bound: bool


def _eigen_gap_sum(eigenvalues: Tensor, selected: int) -> Tensor:
    leading = eigenvalues[:selected, None]
    trailing = eigenvalues[None, selected:]
    if trailing.numel() == 0:
        return eigenvalues.new_zeros(())
    gaps = leading - trailing
    tolerance = torch.finfo(eigenvalues.dtype).eps * eigenvalues.abs().max().clamp_min(1)
    if (gaps <= tolerance).any():
        return eigenvalues.new_tensor(float("inf"))
    return ((leading + trailing) / gaps).sum()


def _hhi_upper_bound(sample_count: int, dimension: int) -> int:
    if sample_count < 3 or dimension < 2:
        raise ValueError("HHI selection needs at least three observations and two coordinates")
    return min(100, dimension - 1, sample_count - 2)


def hhi_rank_admissible_upper_bound(sample_count: int, dimension: int) -> int:
    if sample_count < 3 or dimension < 2:
        raise ValueError("sample_count must be at least 3 and dimension at least 2")
    return min(dimension - 1, sample_count - 2)


def hhi_rank_upper_bound(sample_count: int, dimension: int) -> int:
    return min(100, hhi_rank_admissible_upper_bound(sample_count, dimension))


def _covariance_eigenvalues(samples: Tensor) -> Tensor:
    centered = samples - samples.mean(dim=0)
    n, dimension = centered.shape
    if n < dimension:
        values = torch.linalg.eigvalsh(centered @ centered.mT / n).flip(0).clamp_min(0)
        return torch.cat((values, samples.new_zeros(dimension - n)))
    return torch.linalg.eigvalsh(centered.mT @ centered / n).flip(0).clamp_min(0)


def _hhi_components(eigenvalues: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    dimension = len(eigenvalues)
    tail = eigenvalues[rank:]
    tail_mean = tail.mean()
    tolerance = 128 * torch.finfo(eigenvalues.dtype).eps * eigenvalues[0].clamp_min(1)
    if tail_mean <= tolerance or (rank and (eigenvalues[:rank] <= tolerance).any()):
        infinity = eigenvalues.new_tensor(float("inf"))
        return infinity, infinity
    log_determinant = eigenvalues[:rank].log().sum() + len(tail) * tail_mean.log()
    if rank:
        leading = eigenvalues[:rank, None]
        trailing = tail[None, :]
        gaps = leading - trailing
        if (gaps <= tolerance).any():
            infinity = eigenvalues.new_tensor(float("inf"))
            return infinity, infinity
        direction = (
            trailing * (leading - tail_mean) / (tail_mean * gaps)
        ).sum()
    else:
        direction = eigenvalues.new_zeros(())
    dispersion = tail.square().mean() / tail_mean.square()
    bias = rank * (rank - 1) / 2 + direction + rank + dispersion + dimension
    return log_determinant, bias


def _gic_components(eigenvalues: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Return log determinant and HHI bias without the mean contribution."""
    log_determinant, bias = _hhi_components(eigenvalues, rank)
    return log_determinant, bias - len(eigenvalues)


def _select_hhi(
    samples: Tensor, max_rank: int | None = None
) -> tuple[int, dict[int, Score], bool, tuple[int, ...]]:
    if samples.ndim != 2:
        raise ValueError("core samples must have shape (n, dimension)")
    n, dimension = samples.shape
    admissible = _hhi_upper_bound(n, dimension)
    upper = admissible if max_rank is None else int(max_rank)
    if not 0 <= upper <= admissible:
        raise ValueError(f"max_core_rank must lie in [0, {admissible}]")
    eigenvalues = _covariance_eigenvalues(samples)
    scores: dict[int, Score] = {}
    for rank in range(upper + 1):
        log_determinant, bias = _hhi_components(eigenvalues, rank)
        value = log_determinant + (2.0 / n) * bias
        scores[rank] = Score(float(value))
    finite = {rank: item for rank, item in scores.items() if torch.isfinite(torch.tensor(item.value))}
    if not finite:
        raise RuntimeError("all HHI candidates are numerically undefined")
    best = min(finite, key=lambda rank: finite[rank].value)
    scale = max(1.0, *(abs(item.value) for item in finite.values()))
    tolerance = 128 * torch.finfo(samples.dtype).eps * scale
    tied = tuple(
        rank for rank in sorted(finite)
        if finite[rank].value - finite[best].value <= tolerance
    )
    selected = min(tied)
    return selected, scores, selected == upper, tied


@torch.no_grad()
def select_pca_rank_hhi_gic(
    samples: Tensor,
    *,
    min_rank: int = 0,
    max_rank: int | None = None,
    variant: str = HHI_GIC,
    tie_tolerance: float | None = None,
) -> GICSelection:
    """Compatibility-level detailed HHI result used by experiment reports."""
    if samples.ndim != 2 or samples.is_complex() or not samples.is_floating_point():
        raise ValueError("samples must be a real floating tensor of shape (n, dimension)")
    n, dimension = samples.shape
    admissible = hhi_rank_admissible_upper_bound(n, dimension)
    registered = hhi_rank_upper_bound(n, dimension)
    upper = registered if max_rank is None else int(max_rank)
    if not 0 <= min_rank <= upper <= admissible:
        raise ValueError("invalid HHI candidate range")
    if variant not in (HHI_GIC, TWO_SDR_GIC_SENSITIVITY):
        raise ValueError(f"unknown GIC variant: {variant}")
    eigenvalues = _covariance_eigenvalues(samples)
    scores: dict[int, GICScore] = {}
    for rank in range(min_rank, upper + 1):
        log_determinant, base_bias = _gic_components(eigenvalues, rank)
        if variant == HHI_GIC:
            weight = 2.0
            bias = base_bias + dimension
        else:
            weight = log(n)
            bias = base_bias
        score = log_determinant + (weight / n) * bias
        scores[rank] = GICScore(
            rank,
            float(score),
            float(log_determinant),
            float(bias),
            float(weight),
        )
    finite = {rank: item for rank, item in scores.items() if torch.isfinite(torch.tensor(item.score))}
    if not finite:
        raise RuntimeError("all HHI candidates are numerically undefined")
    best = min(finite, key=lambda rank: finite[rank].score)
    scale = max(1.0, *(abs(item.score) for item in finite.values()))
    tolerance = (
        128 * torch.finfo(samples.dtype).eps * scale
        if tie_tolerance is None
        else float(tie_tolerance)
    )
    tied = tuple(
        rank for rank in sorted(finite)
        if finite[rank].score - finite[best].score <= tolerance
    )
    selected = min(tied)
    return GICSelection(
        selected,
        scores,
        eigenvalues,
        registered,
        admissible,
        upper,
        variant,
        n < dimension,
        len(tied) > 1,
        tied,
        selected == upper,
    )


def _covariance_traces(
    covariance: Tensor | float,
    basis: Tensor,
    dimension: int,
) -> tuple[float, float]:
    if isinstance(covariance, (int, float)):
        variance = float(covariance)
        if variance < 0:
            raise ValueError("covariance scalar must be nonnegative")
        return variance * basis.shape[1], variance * dimension
    if not isinstance(covariance, Tensor) or covariance.is_complex():
        raise ValueError("covariance must be a scalar or real tensor")
    if covariance.ndim == 1:
        if len(covariance) != dimension or (covariance < 0).any():
            raise ValueError("diagonal covariance has invalid shape or values")
        return float(torch.sum(covariance[:, None] * basis.square())), float(covariance.sum())
    if covariance.shape != (dimension, dimension):
        raise ValueError("dense covariance has invalid shape")
    tolerance = 128 * torch.finfo(covariance.dtype).eps * covariance.abs().max().clamp_min(1)
    if not torch.allclose(covariance, covariance.mT, atol=float(tolerance), rtol=0):
        raise ValueError("covariance must be symmetric")
    if torch.linalg.eigvalsh(covariance)[0] < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    return float(torch.sum(basis * (covariance @ basis))), float(torch.trace(covariance))


@torch.no_grad()
def select_split_sure(
    observations: Tensor,
    candidates: Sequence[AffineEstimator],
    covariance: Tensor | float,
    *,
    trace_method: str = "exact",
    num_probes: int = 128,
    seed: int = 0,
    ambiguity_z: float = 4.0,
    tie_tolerance: float | None = None,
) -> SplitSURESelection:
    """Score fixed affine candidates with the exact covariance correction."""
    if trace_method != "exact":
        raise ValueError("the clean package exposes exact traces only")
    if observations.ndim != 2 or observations.is_complex() or not observations.is_floating_point():
        raise ValueError("observations must be a real floating matrix")
    if not candidates:
        raise ValueError("candidates must not be empty")
    labels = [item.label for item in candidates]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique")
    dimension = observations.shape[1]
    scores: dict[str, SplitSUREScore] = {}
    for candidate in candidates:
        centered = observations - candidate.mean
        fitted = candidate.mean + candidate.apply_linear_rows(centered)
        residual = float((observations - fitted).square().sum(dim=1).mean())
        trace_operator, trace_covariance = _covariance_traces(
            covariance, candidate.basis, dimension
        )
        correction = 2 * trace_operator - trace_covariance
        effective = candidate.effective_dimension or candidate.basis.shape[1]
        scores[candidate.label] = SplitSUREScore(
            candidate.label,
            residual + correction,
            residual,
            correction,
            trace_operator,
            trace_covariance,
            0.0,
            int(effective),
            candidate.p_rank,
            candidate.q_rank,
        )
    best = min(scores, key=lambda label: scores[label].score)
    scale = max(1.0, *(abs(item.score) for item in scores.values()))
    tolerance = (
        128 * torch.finfo(observations.dtype).eps * scale
        if tie_tolerance is None
        else float(tie_tolerance)
    )
    by_label = {item.label: item for item in candidates}
    tied = [
        label for label, item in scores.items()
        if item.score - scores[best].score <= tolerance
    ]
    tie_key = lambda label: (
        by_label[label].effective_dimension or by_label[label].basis.shape[1],
        by_label[label].p_rank or dimension,
        by_label[label].q_rank or dimension,
        label,
    )
    ordered = tuple(sorted(tied, key=tie_key))
    return SplitSURESelection(
        ordered[0], scores, len(ordered) > 1, ordered, trace_method, 0, seed, tolerance
    )


def _validate_candidates(
    candidates: Sequence[tuple[int, int]], height: int, width: int
) -> tuple[tuple[int, int], ...]:
    result = tuple((int(p), int(q)) for p, q in candidates)
    if not result or len(set(result)) != len(result):
        raise ValueError("stage1_candidates must be nonempty and unique")
    if any(not (0 < p < height and 0 < q < width) for p, q in result):
        raise ValueError("candidate ranks must be strictly inside the matrix dimensions")
    return result


def _fit_envelope(
    coefficients: Tensor,
    upper: tuple[int, int],
    max_iter: int,
    tol: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int, bool]:
    mean = coefficients.mean(dim=0)
    centered = coefficients - mean
    left, right, iterations, converged = _fit_spatial_subspaces(
        centered, *upper, max_iter, tol
    )
    return mean, centered, left, right, iterations, converged


def _nested_tu_scores(
    centered: Tensor,
    left: Tensor,
    right: Tensor,
    candidates: Sequence[tuple[int, int]],
    sigma2: float,
) -> dict[tuple[int, int], Score]:
    n, height, width = centered.shape
    core = _core(centered, left, right)
    retained = core.to(torch.float64).square().sum(dim=0).div(n).cumsum(0).cumsum(1)
    total = float(centered.to(torch.float64).square().sum() / n)
    xb = centered @ right
    covariance_left = torch.einsum("npb,nrb->pr", xb, xb).to(torch.float64) / n
    ax = torch.einsum("pa,npq->naq", left, centered)
    covariance_right = torch.einsum("naq,nar->qr", ax, ax).to(torch.float64) / n
    eigen_left = torch.linalg.eigvalsh(covariance_left).flip(0)
    eigen_right = torch.linalg.eigvalsh(covariance_right).flip(0)
    scores: dict[tuple[int, int], Score] = {}
    for p_rank, q_rank in candidates:
        residual = total - float(retained[p_rank - 1, q_rank - 1])
        degrees = (
            height * width
            + (n - 1) * p_rank * q_rank
            + float(_eigen_gap_sum(eigen_left, p_rank))
            + float(_eigen_gap_sum(eigen_right, q_rank))
        )
        value = residual + (2.0 * sigma2 / n) * degrees - height * width * sigma2
        scores[(p_rank, q_rank)] = Score(value, residual=residual, degrees_of_freedom=degrees)
    return scores


def _select_pair(scores: Mapping[tuple[int, int], Score], dtype: torch.dtype) -> tuple[int, int]:
    minimum = min(item.value for item in scores.values())
    scale = max(1.0, *(abs(item.value) for item in scores.values()))
    tolerance = 128 * torch.finfo(dtype).eps * scale
    tied = [pair for pair, item in scores.items() if item.value - minimum <= tolerance]
    return min(tied, key=lambda pair: (pair[0] * pair[1], pair[0], pair[1]))


@torch.no_grad()
def select_tu_hhi(
    coefficients: Tensor,
    stage1_candidates: Sequence[tuple[int, int]],
    *,
    sigma2: float,
    max_core_rank: int | None = None,
    max_iter: int = 250,
    tol: float = 1e-5,
) -> RankSelection:
    """Run one nested upper-envelope Tu fit followed by practical HHI GIC."""
    _validate_matrices(coefficients)
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    candidates = _validate_candidates(stage1_candidates, *coefficients.shape[-2:])
    upper = (max(p for p, _ in candidates), max(q for _, q in candidates))
    _, centered, left, right, iterations, converged = _fit_envelope(
        coefficients, upper, max_iter, tol
    )
    stage1_scores = _nested_tu_scores(centered, left, right, candidates, sigma2)
    p_rank, q_rank = _select_pair(stage1_scores, coefficients.dtype)
    selected_left = left[:, :p_rank].contiguous()
    selected_right = right[:, :q_rank].contiguous()
    cores = _core(centered, selected_left, selected_right).reshape(len(coefficients), -1)
    core_rank, core_scores, core_censored, tied = _select_hhi(cores, max_core_rank)
    fit = _make_fit(
        coefficients, selected_left, selected_right, iterations, converged, core_rank
    )
    return RankSelection(
        fit=fit,
        stage1_scores=stage1_scores,
        core_scores=core_scores,
        stage1_right_censored=(p_rank == upper[0], q_rank == upper[1]),
        core_right_censored=core_censored,
        core_rank_ambiguous=len(tied) > 1,
        tied_core_ranks=tied,
        method="nested_tu_hhi",
    )
