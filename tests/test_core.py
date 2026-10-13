from __future__ import annotations

import unittest
from math import sqrt
from unittest import mock

import torch

from fourier_2sdr import (
    fit_2sdr,
    real_fourier_inverse,
    real_fourier_transform,
    select_tu_hhi,
)
from fourier_2sdr.fourier import real_fourier_basis
from fourier_2sdr.fourier import _from_rfft2, _from_valid_rfft2


class FourierCoordinatesTest(unittest.TestCase):
    @staticmethod
    def scalar_reference(
        spectra: torch.Tensor, shape: tuple[int, int]
    ) -> torch.Tensor:
        height, width = shape
        rows = [spectra[..., 0, :].unsqueeze(-2)]
        for frequency in range(1, (height - 1) // 2 + 1):
            upper = spectra[..., frequency, :]
            lower = spectra[..., height - frequency, :]
            rows.extend(
                (
                    ((upper + lower) / sqrt(2.0)).unsqueeze(-2),
                    (1j * (upper - lower) / sqrt(2.0)).unsqueeze(-2),
                )
            )
        if height % 2 == 0:
            rows.append(spectra[..., height // 2, :].unsqueeze(-2))
        first_axis = torch.cat(rows, dim=-2)

        columns = [first_axis[..., 0].real.unsqueeze(-1)]
        for frequency in range(1, (width - 1) // 2 + 1):
            columns.extend(
                (
                    (sqrt(2.0) * first_axis[..., frequency].real).unsqueeze(-1),
                    (-sqrt(2.0) * first_axis[..., frequency].imag).unsqueeze(-1),
                )
            )
        if width % 2 == 0:
            columns.append(first_axis[..., width // 2].real.unsqueeze(-1))
        return torch.cat(columns, dim=-1)

    def test_block_implementation_matches_dense_basis_and_inverts(self) -> None:
        generator = torch.Generator().manual_seed(20260804)
        for height, width in (
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (3, 4),
            (4, 3),
            (5, 7),
            (6, 8),
            (5, 8),
            (6, 7),
        ):
            with self.subTest(shape=(height, width)):
                images = torch.randn(
                    3, height, width, generator=generator, dtype=torch.float64
                )
                left = real_fourier_basis(height, dtype=torch.float64)
                right = real_fourier_basis(width, dtype=torch.float64)
                dense = torch.einsum("pa,npq,qb->nab", left, images, right)
                coordinates = real_fourier_transform(images)
                torch.testing.assert_close(coordinates, dense)
                torch.testing.assert_close(real_fourier_inverse(coordinates), images)
                torch.testing.assert_close(
                    coordinates.square().sum(), images.square().sum()
                )

    def test_vectorized_pack_is_bitwise_scalar_reference(self) -> None:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        shapes = (
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (3, 4),
            (4, 3),
            (5, 7),
            (6, 8),
            (5, 8),
            (6, 7),
            (130, 130),
        )
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                generator = torch.Generator(device=device).manual_seed(20260818)
                for shape in shapes:
                    with self.subTest(device=device, dtype=dtype, shape=shape):
                        images = torch.randn(
                            3,
                            *shape,
                            generator=generator,
                            dtype=dtype,
                            device=device,
                        )
                        spectra = torch.fft.rfft2(images, norm="ortho")
                        expected = self.scalar_reference(spectra, shape)
                        actual = _from_valid_rfft2(spectra, shape)
                        self.assertTrue(torch.equal(actual, expected))

    def test_validated_half_spectrum_still_rejects_bad_boundary(self) -> None:
        images = torch.randn(2, 5, 8, dtype=torch.float64)
        spectra = torch.fft.rfft2(images, norm="ortho")
        spectra[..., 1, 0] += 1j
        with self.assertRaisesRegex(ValueError, "Hermitian symmetry"):
            _from_rfft2(spectra, (5, 8))

    def test_internal_rfft_route_skips_redundant_half_spectrum_validation(self) -> None:
        images = torch.randn(2, 5, 8, dtype=torch.float64)
        with mock.patch(
            "fourier_2sdr.fourier._validate_half_spectrum",
            side_effect=AssertionError("unexpected validation"),
        ):
            coordinates = real_fourier_transform(images)
        self.assertEqual(coordinates.shape, images.shape)

    def test_vectorized_transform_keeps_autograd(self) -> None:
        images = torch.randn(3, 5, 8, dtype=torch.float64, requires_grad=True)
        real_fourier_transform(images).square().sum().backward()
        self.assertIsNotNone(images.grad)
        self.assertTrue(torch.isfinite(images.grad).all())


class TwoSDRTest(unittest.TestCase):
    @staticmethod
    def sample() -> torch.Tensor:
        generator = torch.Generator().manual_seed(7)
        left, _ = torch.linalg.qr(
            torch.randn(6, 2, generator=generator, dtype=torch.float64)
        )
        right, _ = torch.linalg.qr(
            torch.randn(7, 3, generator=generator, dtype=torch.float64)
        )
        scores = torch.randn(48, 2, 3, generator=generator, dtype=torch.float64)
        signal = torch.einsum("pa,nab,qb->npq", left, scores, right)
        return signal + 0.2 * torch.randn(
            48, 6, 7, generator=generator, dtype=torch.float64
        )

    def test_fixed_rank_fit_has_requested_dimensions(self) -> None:
        values = self.sample()
        fit = fit_2sdr(values, (2, 3, 4), max_iter=50)
        self.assertEqual(fit.ranks, (2, 3, 4))
        self.assertEqual(fit.transform(values).shape, (48, 4))
        self.assertEqual(fit.reconstruct(values).shape, values.shape)
        torch.testing.assert_close(fit.ambient_basis.mT @ fit.ambient_basis, torch.eye(4, dtype=values.dtype))

    def test_tu_hhi_selection_returns_valid_fit(self) -> None:
        values = self.sample()
        nested = select_tu_hhi(
            values,
            ((2, 2), (2, 3), (3, 2), (3, 3)),
            sigma2=0.04,
            max_core_rank=4,
            max_iter=50,
        )
        self.assertEqual(nested.method, "nested_tu_hhi")
        self.assertIn(nested.ranks[:2], ((2, 2), (2, 3), (3, 2), (3, 3)))
        self.assertLessEqual(nested.ranks[2], 4)


if __name__ == "__main__":
    unittest.main()
