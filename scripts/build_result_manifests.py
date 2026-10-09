#!/usr/bin/env python3
"""Rebuild the SHA-256 manifest of every folder in results/.

Each manifest lists the files of one study with their sizes and SHA-256
digests, together with the name of the formal run that produced them.
tests/test_results.py checks the committed files against these manifests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SOURCES = {
    "controlled-projection-family": "Section 6.2; formal run controlled_projection_family_snr004_006_seed20260721_r100",
    "same-sample-selection": "Section 6.3; formal run formal_nested_v1/attempt-20260804T024153Z",
    "seventy-s-same-sample": "Section 6.4; formal run formal_blockalgebra_v2/attempt-20260803T081656Z",
    "whitened-calibration": "Section 6.5 calibration; formal runs formal_rho03, formal_rho06, formal_rho08",
    "colored-70s": "Section 6.5 colored-noise 70S study; one folder per formal run",
    "paper-assets": "reference copies of the generated tables and figures; see scripts/make_paper_assets.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for directory in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        files = {
            str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "result_manifest.json"
        }
        manifest = {
            "schema_version": 2,
            "study": directory.name,
            "source": SOURCES.get(directory.name, ""),
            "files": files,
        }
        (directory / "result_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"{directory.name}: {len(files)} files")


if __name__ == "__main__":
    main()
