from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STUDY = ROOT / "experiments" / "colored_70s"
TABLES = (
    "paper1_colored_calibration.tex",
    "paper1_colored_70s.tex",
    "paper1_colored_70s_partA.tex",
)
RUNS = ("anchor_rho00", "formal_rho06", "formal_rho06_s006", "sensitivity_rho03",
        "sensitivity_rho08", "formal_rho06_partb_estpsd")


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _results_present() -> bool:
    calibration = all((RESULTS / "whitened-calibration" / f"rho{tag}" / "candidate_scores.csv")
                      .is_file() for tag in ("03", "06", "08"))
    colored = all((RESULTS / "colored-70s" / run / "part_b_candidates.csv").is_file()
                  for run in RUNS)
    return calibration and colored


@unittest.skipUnless(_results_present(), "frozen colored-noise results are not present")
class FrozenTablesTest(unittest.TestCase):
    def test_tables_reproduce_manuscript(self) -> None:
        make_tables = _module("colored_70s_make_tables", STUDY / "make_tables.py")
        with tempfile.TemporaryDirectory() as tmp:
            make_tables.write_tables(RESULTS, Path(tmp))
            for name in TABLES:
                with self.subTest(table=name):
                    data = (Path(tmp) / name).read_bytes()
                    # results/paper-assets holds the copies used by the manuscript.
                    self.assertEqual(data, (RESULTS / "paper-assets" / name).read_bytes())

    def test_estimated_spectrum_statements(self) -> None:
        summary = _module("colored_70s_estimated_spectrum",
                          STUDY / "estimated_spectrum_summary.py").summarize(RESULTS)
        self.assertEqual(summary["paired"], 15)
        self.assertEqual(summary["same_envelope"], 9)
        self.assertEqual(summary["same_envelope_r"], {(49, 49): 9})
        rises = [round(100 * v, 2) for v in summary["mean_test_rise"].values()]
        self.assertEqual(rises, [0.76, 0.37, 0.55])
        self.assertLess(summary["max_within_lowest"], 3e-4)

    def test_nontest_reference_ranges(self) -> None:
        summary = _module("colored_70s_nontest", STUDY / "nontest_reference.py").summarize(
            RESULTS / "colored-70s" / "reference_tuhhi_nontest" / "reference_tuhhi_nontest.csv")
        self.assertEqual(summary["realizations"], 75)
        self.assertEqual([round(100 * summary[k]) for k in
                          ("white_min", "white_max", "colored_min", "colored_max")], [32, 41, 39, 68])


if __name__ == "__main__":
    unittest.main()
