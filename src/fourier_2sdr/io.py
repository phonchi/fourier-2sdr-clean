"""Minimal local I/O needed by the synthetic 70S application."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def mrc_header(path: str | Path) -> tuple[dict[str, object], int]:
    """Return the mode-2 stack shape and byte offset without reading images."""
    source = Path(path)
    with source.open("rb") as stream:
        header = stream.read(1024)
    if len(header) != 1024:
        raise RuntimeError(f"truncated MRC header: {source}")
    nx, ny, nz, mode = struct.unpack_from("<4i", header, 0)
    nsymbt = struct.unpack_from("<i", header, 92)[0]
    x_length = struct.unpack_from("<f", header, 40)[0]
    if min(nx, ny, nz) <= 0 or mode != 2 or nsymbt < 0:
        raise RuntimeError(f"unsupported MRC header: {source}")
    return {
        "shape": (nz, ny, nx),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "mode": mode,
        "pixel_size": float(x_length / nx) if x_length > 0 else 0.0,
    }, 1024 + nsymbt


def read_mrc_stack(
    path: str | Path, indices: np.ndarray | list[int] | None = None
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Read little-endian float32 MRC mode-2 images and basic metadata."""
    source = Path(path)
    metadata, offset = mrc_header(source)
    nx, ny, nz = (int(metadata[key]) for key in ("nx", "ny", "nz"))
    with source.open("rb") as stream:
        stream.seek(offset)
        values = np.fromfile(stream, dtype="<f4", count=nx * ny * nz)
    if values.size != nx * ny * nz:
        raise RuntimeError(f"truncated MRC data: {source}")
    stack = values.reshape(nz, ny, nx)
    if indices is not None:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.size and (selected.min() < 0 or selected.max() >= nz):
            raise IndexError("MRC particle indices are out of bounds")
        stack = stack[selected]
    return stack, metadata
