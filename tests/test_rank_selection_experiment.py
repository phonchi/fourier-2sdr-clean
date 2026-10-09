import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.same_sample_selection.run_experiment import (
    MPCA_MAX_ITER,
    MPCA_TOL,
    _ambient_tensor_basis,
    _cells,
    _expected_checkpoints,
    _seed,
    _select_stage1,
    _truncated_mpca_basis,
    _validate_checkpoint,
    conditional_clean_prediction_risk,
    confirmation_decision,
    fit_and_score,
    generate_cell,
)


HERE = Path(__file__).resolve().parents[1] / "experiments" / "same_sample_selection"


class GeneratorTests(unittest.TestCase):
    def test_registered_generator_has_exact_population_snr_and_basis(self):
        generated = generate_cell(
            shape=(32, 32),
            n=12,
            snr=0.04,
            spectrum=(1.0, 0.7, 0.5, 0.35),
            eta=0.02,
            seed=8,
        )
        self.assertAlmostEqual(generated.achieved_snr, 0.04, places=12)
        basis = generated.true_signal_basis
        self.assertTrue(torch.allclose(
            basis.mT @ basis,
            torch.eye(5, dtype=basis.dtype),
            atol=2e-12,
            rtol=0,
        ))
        self.assertAlmostEqual(
            sum(generated.component_variances), 32 * 32 * 0.04, places=11
        )

    def test_tensor_basis_matches_c_order_outer_products(self):
        left = torch.eye(3, dtype=torch.float64)[:, :2]
        right = torch.eye(4, dtype=torch.float64)[:, :2]
        basis = _ambient_tensor_basis(left, right)
        expected = torch.outer(left[:, 1], right[:, 0]).reshape(-1)
        self.assertTrue(torch.equal(basis[:, 2], expected))


class RiskTests(unittest.TestCase):
    def test_conditional_risk_matches_monte_carlo(self):
        dtype = torch.float64
        covariance = torch.diag(torch.tensor([4.0, 2.0, 1.0], dtype=dtype))
        basis = torch.tensor([[1.0], [0.0], [0.0]], dtype=dtype)
        mean = torch.tensor([0.0, 0.2, -0.1], dtype=dtype)
        analytic = conditional_clean_prediction_risk(
            basis, mean, covariance, sigma2=1.0
        )
        generator = torch.Generator().manual_seed(19)
        draws = 300000
        clean = torch.randn(draws, 3, generator=generator, dtype=dtype)
        clean = clean * torch.tensor([2.0, math.sqrt(2.0), 1.0], dtype=dtype)
        noise = torch.randn(draws, 3, generator=generator, dtype=dtype)
        projection = basis @ basis.mT
        reconstructed = (
            (clean + noise) @ projection
            + mean @ (torch.eye(3, dtype=dtype) - projection)
        )
        empirical = float((reconstructed - clean).square().mean())
        self.assertAlmostEqual(empirical, analytic, delta=0.012)

    def test_registered_smoke_recovers_balanced_envelope_and_advantage(self):
        generated = generate_cell(
            shape=(32, 32),
            n=96,
            snr=0.04,
            spectrum=(1.0, 1.0, 1.0, 1.0),
            eta=0.0,
            seed=1985248429,
        )
        result = fit_and_score(generated)
        self.assertEqual(
            (result["selected_p"], result["selected_q"], result["selected_r"]),
            (4, 4, 4),
        )
        self.assertLess(
            result["risks"]["two_sdr_sure_hhi"],
            result["risks"]["vector_pca_hhi"],
        )
        self.assertLess(
            result["risks"]["two_sdr_sure_hhi"],
            result["risks"]["mpca_sure_full_core"],
        )


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (HERE / "protocol.json").read_text(encoding="utf-8")
        )

    def test_formal_has_eight_primary_and_eight_boundary_cells(self):
        cells = _cells(self.protocol, "formal")
        self.assertEqual(sum(cell["eta"] == 0.0 for cell in cells), 8)
        self.assertEqual(sum(cell["eta"] == 0.02 for cell in cells), 8)
        self.assertEqual(self.protocol["modes"]["formal"]["replicates"], 100)

    def test_discovery_covers_every_formal_cell_with_separate_replicates(self):
        discovery = {
            (cell["n"], cell["snr"], cell["spectrum"], cell["eta"])
            for cell in _cells(self.protocol, "discovery")
        }
        formal = {
            (cell["n"], cell["snr"], cell["spectrum"], cell["eta"])
            for cell in _cells(self.protocol, "formal")
        }
        self.assertEqual(discovery, formal)
        self.assertEqual(self.protocol["modes"]["discovery"]["replicates"], 2)

    def test_seed_namespaces_are_disjoint(self):
        ids = [
            self.protocol["modes"][mode]["seed_namespace"]
            for mode in ("smoke", "discovery", "formal")
        ]
        self.assertEqual(len(ids), len(set(ids)))
        seeds = {
            _seed(namespace, "same_cell", 0)
            for namespace in ids
        }
        self.assertEqual(len(seeds), 3)

    def test_registered_mpca_solver_settings_match_runner(self):
        self.assertEqual(
            self.protocol["selection"]["mpca_max_iter"], MPCA_MAX_ITER
        )
        self.assertEqual(
            self.protocol["selection"]["mpca_tolerance"], MPCA_TOL
        )

    def test_only_formal_can_return_confirmation_decision(self):
        summary = [
            {
                "cell_id": "primary",
                "comparison": comparison,
                "eta": 0.0,
                "replicates": 100,
                "ci_upper": -0.01,
            }
            for comparison in ("two_sdr_minus_pca", "two_sdr_minus_mpca")
        ]
        self.assertEqual(
            confirmation_decision(
                summary,
                mode="discovery",
                expected_replicates=100,
                expected_primary_cell_ids={"primary"},
            )["status"],
            "not_evaluated",
        )
        self.assertEqual(
            confirmation_decision(
                summary,
                mode="formal",
                expected_replicates=100,
                expected_primary_cell_ids={"primary"},
            )["status"],
            "pass",
        )
        self.assertEqual(
            confirmation_decision(
                summary[:1],
                mode="formal",
                expected_replicates=100,
                expected_primary_cell_ids={"primary"},
            )["status"],
            "incomplete",
        )

    def test_checkpoint_identity_validation_rejects_stale_payload(self):
        cell = {"n": 625, "snr": 0.04, "spectrum": "equal", "eta": 0.0}
        expected = _expected_checkpoints([cell], 1, 731101)
        filename, metadata = next(iter(expected.items()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / filename
            payload = {
                **metadata,
                "risks": {},
                "selected_p": 4,
                "selected_q": 4,
                "selected_r": 4,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_validate_checkpoint(path, metadata)["seed"], metadata["seed"])
            payload["seed"] += 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                _validate_checkpoint(path, metadata)


class NestedStageOneTests(unittest.TestCase):
    """The Stage-I pair now comes from one over-specified fit, as in 2SDR."""

    def _observed(self):
        generated = generate_cell(
            shape=(32, 32),
            n=96,
            snr=0.04,
            spectrum=(1.0, 1.0, 1.0, 1.0),
            eta=0.0,
            seed=1985248429,
        )
        return generated.observed

    def test_candidate_bases_are_truncations_of_one_upper_envelope(self):
        from fourier_2sdr.model import _fit_spatial_subspaces

        observed = self._observed()
        _, _, envelope = _select_stage1(observed, (2, 4, 8))

        # The envelope is the fit at the largest candidate pair.
        centered = observed - observed.mean(dim=0)
        left, right, _, _ = _fit_spatial_subspaces(
            (centered,), 8, 8, MPCA_MAX_ITER, MPCA_TOL
        )
        self.assertTrue(torch.allclose(envelope["left"], left, atol=1e-10))
        self.assertTrue(torch.allclose(envelope["right"], right, atol=1e-10))

        # Every candidate basis is a leading block of that same fit.
        for p, q in ((2, 2), (4, 4), (8, 4)):
            basis, _, _ = _truncated_mpca_basis(envelope, p, q)
            expected = _ambient_tensor_basis(
                envelope["left"][:, :p], envelope["right"][:, :q]
            )
            self.assertTrue(torch.equal(basis, expected))
            self.assertEqual(basis.shape[1], p * q)

    def test_one_fit_serves_every_candidate_score(self):
        observed = self._observed()
        _, records, envelope = _select_stage1(observed, (2, 4, 8))
        self.assertEqual(len(records), 9)
        # A single alternating path means one convergence record, shared.
        for record in records.values():
            self.assertEqual(record["iterations"], envelope["iterations"])
            self.assertEqual(record["converged"], envelope["converged"])

    def test_registered_generator_selects_the_generating_pair(self):
        # The criterion is written for unit-variance white noise, which is what
        # the registered generator supplies, so a favourable cell should land
        # on the 4-by-4 envelope the generator used.
        generated = generate_cell(
            shape=(32, 32),
            n=1000,
            snr=0.06,
            spectrum=(1.0, 1.0, 1.0, 1.0),
            eta=0.0,
            seed=1985248429,
        )
        selected, _, _ = _select_stage1(generated.observed, (2, 4, 8))
        self.assertEqual(selected, (4, 4))


if __name__ == "__main__":
    unittest.main()
