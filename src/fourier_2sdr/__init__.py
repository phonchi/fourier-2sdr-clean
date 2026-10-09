"""Small public interface for Fourier-domain two-stage dimension reduction."""

from .fourier import real_fourier_inverse, real_fourier_transform
from .model import TwoSDRFit, fit_2sdr
from .selection import RankSelection, select_corrected_holdout, select_tu_hhi

__all__ = [
    "RankSelection",
    "TwoSDRFit",
    "fit_2sdr",
    "real_fourier_inverse",
    "real_fourier_transform",
    "select_corrected_holdout",
    "select_tu_hhi",
]

from .whitening import (
    affine_noise_trace,
    corrected_holdout_score,
    periodic_ar1_spectrum,
    separable_noise_scale,
    separable_periodic_ar1_spectrum,
    whitened_projection,
    whitened_projector_matrix,
)

__all__ += [
    "affine_noise_trace",
    "corrected_holdout_score",
    "periodic_ar1_spectrum",
    "separable_noise_scale",
    "separable_periodic_ar1_spectrum",
    "whitened_projection",
    "whitened_projector_matrix",
]
