"""Complete real Fourier coordinates used by Fourier-domain 2SDR."""

from __future__ import annotations

from math import sqrt

import torch

Tensor = torch.Tensor


def _require_real(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.is_complex() or not value.is_floating_point():
        raise ValueError(f"{name} must be a real floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def real_fourier_basis(
    size: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return ``Q_d`` with columns DC, cosine/sine pairs, then Nyquist."""
    if size <= 0:
        raise ValueError("size must be positive")
    dtype = dtype or torch.get_default_dtype()
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    locations = torch.arange(size, dtype=dtype, device=device)
    columns = [torch.ones(size, dtype=dtype, device=device) / sqrt(size)]
    scale = sqrt(2.0 / size)
    for frequency in range(1, (size - 1) // 2 + 1):
        angle = 2.0 * torch.pi * frequency * locations / size
        columns.extend((scale * angle.cos(), scale * angle.sin()))
    if size % 2 == 0:
        columns.append(((-1.0) ** locations) / sqrt(size))
    return torch.stack(columns, dim=1)


def _pack_last_axis(spectra: Tensor, size: int) -> Tensor:
    output = torch.empty(
        (*spectra.shape[:-1], size),
        dtype=spectra.real.dtype,
        device=spectra.device,
    )
    output[..., 0] = spectra[..., 0].real
    paired = (size - 1) // 2
    if paired:
        interior = spectra[..., 1 : paired + 1]
        output[..., 1 : 2 * paired + 1 : 2] = sqrt(2.0) * interior.real
        output[..., 2 : 2 * paired + 1 : 2] = -sqrt(2.0) * interior.imag
    if size % 2 == 0:
        output[..., -1] = spectra[..., size // 2].real
    return output


def _unpack_last_axis(coordinates: Tensor) -> Tensor:
    size = coordinates.shape[-1]
    zeros = torch.zeros_like(coordinates[..., 0])
    pieces = [torch.complex(coordinates[..., 0], zeros).unsqueeze(-1)]
    offset = 1
    for _ in range(1, (size - 1) // 2 + 1):
        cosine = coordinates[..., offset] / sqrt(2.0)
        sine = coordinates[..., offset + 1] / sqrt(2.0)
        pieces.append(torch.complex(cosine, -sine).unsqueeze(-1))
        offset += 2
    if size % 2 == 0:
        pieces.append(torch.complex(coordinates[..., offset], zeros).unsqueeze(-1))
    return torch.cat(pieces, dim=-1)


def _validate_half_spectrum(
    spectra: Tensor, spatial_shape: tuple[int, int]
) -> tuple[int, int]:
    height, width = map(int, spatial_shape)
    if height <= 0 or width <= 0:
        raise ValueError("spatial dimensions must be positive")
    if spectra.ndim < 2 or spectra.shape[-2:] != (height, width // 2 + 1):
        raise ValueError("rfft2 shape does not match spatial_shape")
    if not torch.isfinite(spectra.real).all() or not torch.isfinite(spectra.imag).all():
        raise ValueError("spectra must contain only finite values")
    tolerance = 128 * torch.finfo(spectra.real.dtype).eps * spectra.abs().max().clamp_min(1)
    reflected = (-torch.arange(height, device=spectra.device)) % height
    boundary_columns = [0] + ([width // 2] if width % 2 == 0 else [])
    for index in boundary_columns:
        column = spectra[..., :, index]
        if not torch.allclose(
            column,
            column.index_select(-1, reflected).conj(),
            atol=float(tolerance),
            rtol=0,
        ):
            raise ValueError("rfft2 boundary column violates Hermitian symmetry")
    return height, width


def _apply_first_axis_blocks(spectra: Tensor, height: int) -> Tensor:
    output = torch.empty_like(spectra)
    output[..., 0, :] = spectra[..., 0, :]
    paired = (height - 1) // 2
    if paired:
        upper = spectra[..., 1 : paired + 1, :]
        lower = spectra[..., height - paired : height, :].flip(-2)
        output[..., 1 : 2 * paired + 1 : 2, :] = (upper + lower) / sqrt(2.0)
        output[..., 2 : 2 * paired + 1 : 2, :] = (
            1j * (upper - lower) / sqrt(2.0)
        )
    if height % 2 == 0:
        output[..., -1, :] = spectra[..., height // 2, :]
    return output


def _from_valid_rfft2(
    spectra: Tensor, spatial_shape: tuple[int, int]
) -> Tensor:
    """Finish real coordinates from an internally produced rFFT2 array."""
    height, width = map(int, spatial_shape)
    expected = (height, width // 2 + 1)
    if not isinstance(spectra, Tensor) or not spectra.is_complex():
        raise TypeError("spectra must be a complex torch.Tensor")
    if height <= 0 or width <= 0 or spectra.ndim < 2:
        raise ValueError("spatial dimensions and spectra must be nonempty")
    if spectra.shape[-2:] != expected:
        raise ValueError("rfft2 shape does not match spatial_shape")
    return _pack_last_axis(_apply_first_axis_blocks(spectra, height), width)


def _from_rfft2(spectra: Tensor, spatial_shape: tuple[int, int]) -> Tensor:
    """Finish ``Q_h.T @ X @ Q_w`` from an existing orthonormal rFFT2."""
    if not isinstance(spectra, Tensor) or not spectra.is_complex():
        raise TypeError("spectra must be a complex torch.Tensor")
    height, width = _validate_half_spectrum(spectra, spatial_shape)
    return _from_valid_rfft2(spectra, (height, width))


def real_fourier_transform(images: Tensor) -> Tensor:
    """Map real ``(..., h, w)`` images to complete Parseval coordinates.

    The implementation performs one orthonormal ``rfft2`` and then only fixed
    conjugate-pair block operations.  The result equals ``Q_h.T @ X @ Q_w``.
    """
    _require_real(images, "images")
    if images.ndim < 2 or min(images.shape[-2:]) <= 0:
        raise ValueError("images must end in two nonempty spatial dimensions")
    shape = tuple(int(value) for value in images.shape[-2:])
    return _from_valid_rfft2(torch.fft.rfft2(images, norm="ortho"), shape)


def real_fourier_inverse(coefficients: Tensor) -> Tensor:
    """Invert complete real Fourier coordinates to real images."""
    _require_real(coefficients, "coefficients")
    if coefficients.ndim < 2 or min(coefficients.shape[-2:]) <= 0:
        raise ValueError("coefficients must end in two nonempty dimensions")
    height = coefficients.shape[-2]
    height_spectra = _unpack_last_axis(coefficients.movedim(-2, -1))
    width_coordinates = torch.fft.irfft(
        height_spectra, n=height, dim=-1, norm="ortho"
    ).movedim(-1, -2)
    row_spectra = _unpack_last_axis(width_coordinates)
    return torch.fft.irfft(
        row_spectra, n=coefficients.shape[-1], dim=-1, norm="ortho"
    )
