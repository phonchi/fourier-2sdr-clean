from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from fourier_2sdr import (
    affine_noise_trace,
    corrected_holdout_score,
    periodic_ar1_spectrum,
    separable_noise_scale,
    whitened_projection,
    whitened_projector_matrix,
)
from fourier_2sdr.whitening import real_fourier_frequency_labels

ROOT = Path(__file__).resolve().parents[1]


def reference_ar1(size: int, rho: float) -> np.ndarray:
    """The periodic AR(1) formula used by the reported runs, written out."""
    labels = [0]
    for frequency in range(1, (size - 1) // 2 + 1):
        labels.extend((frequency, frequency))
    if size % 2 == 0:
        labels.append(size // 2)
    labels = np.asarray(labels)
    values = (1.0 - rho**2) / (1.0 + rho**2 - 2.0 * rho * np.cos(2 * np.pi * labels / size))
    return values / values.mean()


def random_basis(dim: int, rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(dim, rank, generator=generator, dtype=torch.float64)).Q


class SpectrumTest(unittest.TestCase):
    def test_matches_reference_formula(self) -> None:
        for size in (1, 2, 5, 8, 9, 32, 130):
            for rho in (0.0, 0.3, 0.6, 0.8):
                with self.subTest(size=size, rho=rho):
                    np.testing.assert_array_equal(
                        periodic_ar1_spectrum(size, rho), reference_ar1(size, rho))

    def test_matches_experiment_helper(self) -> None:
        sys.path.insert(0, str(ROOT))
        from experiments.common import periodic_ar1_eigenvalues

        for size, rho in ((7, 0.3), (16, 0.6), (130, 0.8)):
            np.testing.assert_array_equal(
                periodic_ar1_spectrum(size, rho), periodic_ar1_eigenvalues(size, rho))

    def test_mean_one_flat_at_zero_and_labels(self) -> None:
        self.assertEqual(real_fourier_frequency_labels(6).tolist(), [0, 1, 1, 2, 2, 3])
        self.assertAlmostEqual(float(periodic_ar1_spectrum(32, 0.6).mean()), 1.0, places=12)
        np.testing.assert_allclose(periodic_ar1_spectrum(10, 0.0), np.ones(10))
        scale = separable_noise_scale((6, 5), 0.6)
        np.testing.assert_allclose(
            scale**2, np.outer(periodic_ar1_spectrum(6, 0.6), periodic_ar1_spectrum(5, 0.6)))


class WhitenedProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (6, 5)
        self.dim = 30
        self.scale = torch.as_tensor(separable_noise_scale(self.shape, 0.6).reshape(-1))
        self.basis = random_basis(self.dim, 4, 0)

    def test_idempotent_and_nonsymmetric(self) -> None:
        h = whitened_projector_matrix(self.basis, self.scale)
        torch.testing.assert_close(h @ h, h)
        self.assertGreater(float((h - h.mT).abs().max()), 1e-3)
        plain = whitened_projector_matrix(self.basis)
        torch.testing.assert_close(plain, plain.mT)

    def test_back_transform_matches_dense_map(self) -> None:
        generator = torch.Generator().manual_seed(1)
        z = torch.randn(7, self.dim, generator=generator, dtype=torch.float64)
        mean = torch.randn(self.dim, generator=generator, dtype=torch.float64)
        h = whitened_projector_matrix(self.basis, self.scale)
        expected = mean + (z - mean) @ h.mT
        torch.testing.assert_close(whitened_projection(z, mean, self.basis, self.scale), expected)
        # Projecting in whitened coordinates then rescaling is the same map.
        whitened = whitened_projection(z / self.scale, mean / self.scale, self.basis)
        torch.testing.assert_close(whitened * self.scale, expected)
        # Applying the map twice changes nothing (idempotence of the affine map).
        once = whitened_projection(z, mean, self.basis, self.scale)
        torch.testing.assert_close(whitened_projection(once, mean, self.basis, self.scale), once)
        empty = self.basis[:, :0]
        torch.testing.assert_close(whitened_projection(z, mean, empty, self.scale),
                                   mean.expand(len(z), -1))


class CorrectedHoldoutTest(unittest.TestCase):
    def test_trace_shortcut_matches_dense(self) -> None:
        dim = 30
        omega = torch.as_tensor(separable_noise_scale((6, 5), 0.8).reshape(-1)) ** 2 * 0.7
        basis = random_basis(dim, 5, 2)
        for scale in (None, omega.sqrt(), torch.linspace(0.5, 2.0, dim, dtype=torch.float64)):
            dense = float(torch.trace(torch.diag(omega) @ whitened_projector_matrix(basis, scale)))
            self.assertAlmostEqual(affine_noise_trace(basis, omega), dense, places=10)
        self.assertEqual(affine_noise_trace(basis[:, :0], omega), 0.0)

    def test_score_matches_dense_formula(self) -> None:
        dim = 30
        omega = torch.as_tensor(separable_noise_scale((6, 5), 0.6).reshape(-1)) ** 2
        scale = omega.sqrt()
        basis = random_basis(dim, 3, 3)
        generator = torch.Generator().manual_seed(4)
        holdout = torch.randn(11, dim, generator=generator, dtype=torch.float64)
        mean = torch.randn(dim, generator=generator, dtype=torch.float64)
        estimate = whitened_projection(holdout, mean, basis, scale)
        h = whitened_projector_matrix(basis, scale)
        sigma = torch.diag(omega)
        residual = ((holdout - estimate) ** 2).sum(dim=1).mean()
        expected = float(residual + 2 * torch.trace(sigma @ h) - torch.trace(sigma)) / dim
        self.assertAlmostEqual(
            corrected_holdout_score(holdout, estimate, basis, omega), expected, places=12)

    def test_score_is_unbiased_for_the_risk(self) -> None:
        # Fixed affine map, Gaussian noise with diagonal covariance: the mean
        # corrected score over many holdout draws equals the per-coordinate risk.
        dim = 12
        omega = torch.linspace(0.5, 2.0, dim, dtype=torch.float64)
        scale = omega.sqrt()
        basis = random_basis(dim, 2, 5)
        mean = torch.zeros(dim, dtype=torch.float64)
        signal = torch.linspace(-1.0, 1.0, dim, dtype=torch.float64)
        h = whitened_projector_matrix(basis, scale)
        identity = torch.eye(dim, dtype=torch.float64)
        risk = float(((h - identity) @ signal).square().sum()
                     + (h * omega[None, :] * h).sum()) / dim
        generator = torch.Generator().manual_seed(6)
        noise = torch.randn(200_000, dim, generator=generator, dtype=torch.float64) * scale
        holdout = signal + noise
        estimate = whitened_projection(holdout, mean, basis, scale)
        score = corrected_holdout_score(holdout, estimate, basis, omega)
        self.assertAlmostEqual(score, risk, delta=0.01 * risk)


if __name__ == "__main__":
    unittest.main()
