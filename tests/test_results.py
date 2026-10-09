from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ResultManifestTest(unittest.TestCase):
    def test_compact_result_files_match_manifests(self) -> None:
        manifests = sorted((ROOT / "results").glob("*/result_manifest.json"))
        folders = sorted(p for p in (ROOT / "results").iterdir() if p.is_dir())
        self.assertEqual(len(manifests), len(folders))
        for manifest_path in manifests:
            with self.subTest(study=manifest_path.parent.name):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["study"], manifest_path.parent.name)
                on_disk = {
                    str(p.relative_to(manifest_path.parent))
                    for p in manifest_path.parent.rglob("*")
                    if p.is_file() and p.name != "result_manifest.json"
                }
                self.assertEqual(set(manifest["files"]), on_disk)
                for name, expected in manifest["files"].items():
                    path = manifest_path.parent / name
                    self.assertEqual(path.stat().st_size, expected["bytes"])
                    self.assertEqual(
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        expected["sha256"],
                    )



class PaperAssetTest(unittest.TestCase):
    def test_paper_assets_regenerate_from_results(self) -> None:
        import subprocess
        import sys
        import tempfile

        script = ROOT / "scripts" / "make_paper_assets.py"
        if not (ROOT / "experiments" / "colored_70s" / "make_tables.py").exists():
            self.skipTest("colored_70s table generator not present")
        with tempfile.TemporaryDirectory() as out:
            result = subprocess.run(
                [sys.executable, str(script), "--output-dir", out, "--check"],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
