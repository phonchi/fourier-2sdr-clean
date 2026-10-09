"""Known-truth models and fixed-envelope PCA/2SDR/MPCA fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


D1 = 24
D2 = 8
AMBIENT_DIM = D1 * D2
MODEL_SEED = 20260721
MASTER_SEED = 20260721
HELDOUT_N = 1000
MAX_ITER = 300
TOL = 1e-8

SCENARIO_SPECS = (
    ("pca_boundary", 24, 8, 4, "full-mode PCA boundary"),
    ("interior_row_thin", 2, 8, 4, "interior row-thin envelope"),
    ("interior_balanced", 4, 4, 4, "interior balanced envelope"),
    ("interior_col_thin", 8, 2, 4, "interior column-thin envelope"),
    ("mpca_boundary", 4, 4, 16, "full-core MPCA boundary"),
)


@dataclass(frozen=True)
class Scenario:
    scenario: str
    p: int
    q: int
    r: int
    role: str
    signal_basis: np.ndarray
    true_a: np.ndarray | None
    true_b: np.ndarray | None


@dataclass(frozen=True)
class Fit:
    method: str
    basis: np.ndarray
    iterations: int
    converged: bool


def leading_eigenvectors(matrix: np.ndarray, rank: int) -> np.ndarray:
    """Return leading eigenvectors of a real symmetric matrix."""
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    order = np.argsort(values)[::-1][:rank]
    return np.ascontiguousarray(vectors[:, order])


def projector_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Normalized Frobenius distance between possibly unequal-rank projectors."""
    cross = first.T @ second
    squared = first.shape[1] + second.shape[1] - 2.0 * np.sum(cross * cross)
    squared = max(float(squared), 0.0)
    # The trace identity above is efficient away from zero but loses relative
    # precision for algebraically identical boundary projectors.  Recompute
    # only that narrow case from the projector difference itself.
    if squared < 1e-10:
        difference = first @ first.T - second @ second.T
        return float(
            np.linalg.norm(difference, ord="fro")
            / np.sqrt(first.shape[1] + second.shape[1])
        )
    return float(np.sqrt(squared / (first.shape[1] + second.shape[1])))


def minimal_envelope_ranks(basis: np.ndarray) -> tuple[int, int]:
    matrices = [basis[:, index].reshape(D1, D2) for index in range(basis.shape[1])]
    row_envelope = np.concatenate(matrices, axis=1)
    column_envelope = np.concatenate(matrices, axis=0)
    return int(np.linalg.matrix_rank(row_envelope)), int(np.linalg.matrix_rank(column_envelope))


def build_scenarios(seed: int = MODEL_SEED) -> dict[str, Scenario]:
    rng = np.random.default_rng(seed)
    scenarios: dict[str, Scenario] = {}
    for name, p, q, r, role in SCENARIO_SPECS:
        if name == "pca_boundary":
            basis, _ = np.linalg.qr(rng.normal(size=(AMBIENT_DIM, r)))
            basis = np.ascontiguousarray(basis[:, :r])
            if minimal_envelope_ranks(basis) != (D1, D2):
                raise RuntimeError("PCA-boundary basis did not have a full mode envelope")
            true_a = true_b = None
        else:
            true_a, _ = np.linalg.qr(rng.normal(size=(D1, p)))
            true_b, _ = np.linalg.qr(rng.normal(size=(D2, q)))
            spatial = np.kron(true_a[:, :p], true_b[:, :q])
            if r == p * q:
                core = np.eye(r)
            else:
                core, _ = np.linalg.qr(rng.normal(size=(p * q, r)))
                core = core[:, :r]
            basis = np.ascontiguousarray(spatial @ core)
            if minimal_envelope_ranks(basis) != (p, q):
                raise RuntimeError(f"{name} did not realize its declared envelope")
            true_a = np.ascontiguousarray(true_a[:, :p])
            true_b = np.ascontiguousarray(true_b[:, :q])
        if not np.allclose(basis.T @ basis, np.eye(r), atol=1e-12):
            raise RuntimeError(f"{name} signal basis is not orthonormal")
        scenarios[name] = Scenario(name, p, q, r, role, basis, true_a, true_b)
    return scenarios


def signal_eigenvalue(snr: float, rank: int) -> float:
    return AMBIENT_DIM * float(snr) / int(rank)


def population_covariance(scenario: Scenario, snr: float) -> tuple[np.ndarray, np.ndarray]:
    value = signal_eigenvalue(snr, scenario.r)
    signal = value * (scenario.signal_basis @ scenario.signal_basis.T)
    return signal, signal + np.eye(AMBIENT_DIM)


def _bartlett_identity(dimension: int, degrees_of_freedom: int,
                       rng: np.random.Generator) -> np.ndarray:
    if degrees_of_freedom < dimension:
        raise ValueError("Wishart degrees of freedom must be at least the dimension")
    lower = np.zeros((dimension, dimension), dtype=np.float64)
    for index in range(dimension):
        lower[index, index] = np.sqrt(rng.chisquare(degrees_of_freedom - index))
        lower[index + 1:, index] = rng.normal(size=dimension - index - 1)
    return lower @ lower.T


def snr_seed_code(snr: float) -> int:
    """Encode a numeric SNR independently of its position in a design list."""
    code = int(round(float(snr) * 1_000_000))
    if not np.isclose(code / 1_000_000, float(snr), atol=1e-12, rtol=0.0):
        raise ValueError("SNR must be representable to six decimal places")
    return code


def replicate_seed(master_seed: int, scenario_index: int, snr: float,
                   sample_size: int, replicate: int) -> int:
    sequence = np.random.SeedSequence(
        [
            int(master_seed), int(scenario_index), snr_seed_code(snr),
            int(sample_size), int(replicate),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def training_scatter(scenario: Scenario, snr: float, sample_size: int,
                     rng: np.random.Generator) -> np.ndarray:
    _, covariance = population_covariance(scenario, snr)
    root = np.linalg.cholesky(covariance)
    standard = _bartlett_identity(AMBIENT_DIM, sample_size - 1, rng)
    scatter = root @ standard @ root.T
    return (scatter + scatter.T) / 2.0


def _fixed_mpca(scatter: np.ndarray, p: int, q: int,
                max_iter: int = MAX_ITER, tol: float = TOL) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Fit fixed-dimensional MPCA from a vectorized centered scatter."""
    if not (1 <= p <= D1 and 1 <= q <= D2):
        raise ValueError("invalid fixed envelope dimensions")
    covariance4 = scatter.reshape(D1, D2, D1, D2)
    if q == D2:
        b = np.eye(D2)
    else:
        column_marginal = np.einsum("ijil->jl", covariance4, optimize=True)
        b = leading_eigenvectors(column_marginal, q)
    previous_a = previous_b = None
    converged = False
    for iteration in range(1, max_iter + 1):
        if p == D1:
            a = np.eye(D1)
        else:
            pb = b @ b.T
            row_covariance = np.einsum("abcd,bd->ac", covariance4, pb, optimize=True)
            a = leading_eigenvectors(row_covariance, p)
        if q == D2:
            new_b = np.eye(D2)
        else:
            pa = a @ a.T
            column_covariance = np.einsum("abcd,ac->bd", covariance4, pa, optimize=True)
            new_b = leading_eigenvectors(column_covariance, q)
        b = new_b
        if previous_a is not None:
            distance = max(projector_distance(a, previous_a), projector_distance(b, previous_b))
            if distance <= tol:
                converged = True
                break
        previous_a, previous_b = a.copy(), b.copy()
        if p == D1 and q == D2:
            converged = True
            break
    return np.ascontiguousarray(a), np.ascontiguousarray(b), iteration, converged


def fit_methods(scatter: np.ndarray, scenario: Scenario) -> dict[str, Fit]:
    pca = leading_eigenvectors(scatter, scenario.r)
    a, b, iterations, converged = _fixed_mpca(scatter, scenario.p, scenario.q)
    spatial = np.kron(a, b)
    core_covariance = spatial.T @ scatter @ spatial
    g = leading_eigenvectors(core_covariance, scenario.r)
    twosdr = np.ascontiguousarray(spatial @ g)
    mpca = np.ascontiguousarray(spatial)
    for label, basis in (("pca", pca), ("twosdr", twosdr), ("mpca", mpca)):
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=2e-10):
            raise RuntimeError(f"{label} basis is not orthonormal")
    return {
        "pca": Fit("pca", pca, 1, True),
        "twosdr": Fit("twosdr", twosdr, iterations, converged),
        "mpca": Fit("mpca", mpca, iterations, converged),
    }


def heldout_draw(scenario: Scenario, snr: float, rng: np.random.Generator,
                 sample_size: int = HELDOUT_N) -> tuple[np.ndarray, np.ndarray]:
    value = signal_eigenvalue(snr, scenario.r)
    scores = rng.normal(size=(sample_size, scenario.r)) * np.sqrt(value)
    signal = scores @ scenario.signal_basis.T
    noise = rng.normal(size=(sample_size, AMBIENT_DIM))
    return signal, noise


def method_metrics(fit: Fit, scenario: Scenario, snr: float,
                   heldout_signal: np.ndarray, heldout_noise: np.ndarray) -> dict[str, float | int | bool]:
    cross = scenario.signal_basis.T @ fit.basis
    overlap = float(np.sum(cross * cross))
    rank = fit.basis.shape[1]
    eigenvalue = signal_eigenvalue(snr, scenario.r)
    total_signal = eigenvalue * scenario.r
    captured_signal = eigenvalue * overlap
    retained_noise = float(rank)
    population_risk = (total_signal - captured_signal + retained_noise) / total_signal
    signal_scores = heldout_signal @ fit.basis
    noise_scores = heldout_noise @ fit.basis
    test_total_signal = float(np.sum(heldout_signal * heldout_signal))
    heldout_risk = (
        test_total_signal - float(np.sum(signal_scores * signal_scores))
        + float(np.sum(noise_scores * noise_scores))
    ) / test_total_signal
    return {
        "method": fit.method,
        "output_rank": rank,
        "projector_frobenius_error": projector_distance(fit.basis, scenario.signal_basis),
        "clean_signal_capture": captured_signal / total_signal,
        "retained_noise_energy": retained_noise,
        "retained_noise_fraction": retained_noise / AMBIENT_DIM,
        "population_projection_risk": population_risk,
        "heldout_projection_risk": heldout_risk,
        "iterations": fit.iterations,
        "converged": fit.converged,
    }


def boundary_residual(first: Fit, second: Fit, first_risk: float,
                      second_risk: float) -> tuple[float, float]:
    return (
        projector_distance(first.basis, second.basis),
        abs(float(first_risk) - float(second_risk)),
    )
