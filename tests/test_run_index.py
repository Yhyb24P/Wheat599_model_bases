"""Run index derives fields read-only from run artifacts and is deterministic."""
import json
import tempfile
import unittest
from pathlib import Path

from genopred.evaluation import run_index


def _run(results, rel, prov=None, status=None, summary=False, failures=None):
    d = results / rel
    d.mkdir(parents=True)
    if prov is not None:
        (d / "provenance.json").write_text(json.dumps(prov))
    if status is not None:
        (d / "execution_status.json").write_text(json.dumps(status))
    if summary:
        (d / "summary.json").write_text("{}")
    if failures is not None:
        (d / "failures.json").write_text(json.dumps(failures))
    return d


class RunIndexTest(unittest.TestCase):
    def test_derivation_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            _run(results, "wheat599/neural/quantile_tabm_final_20261106",
                 prov={"config": {"model": "quantile_tabm_final"}, "git_commit": "abc"},
                 status={"status": "complete", "completed": 80, "expected": 80})
            _run(results, "wheat599/baselines/gsformer_nsa_outer_20261115",
                 prov={"config": {"model": "gsformer_nsa"}}, summary=True)
            _run(results, "wheat599/baselines/background_ridge_20261102",
                 prov={"config": {"model": None}}, summary=True)
            _run(results, "wheat599/diagnostics/bimsgp_diag_ftt_20261029",
                 prov={"config": {"model": "bimsgp"}}, summary=True)
            _run(results, "wheat599/kernels/robust_icm_development_20261103",
                 prov={"config": {"model": "robust_icm"}, "phase": "dev"},
                 status={"status": "running"})
            _run(results, "wheat2000/wheat2000_huber_dev_20261115",
                 prov={"config": {"model": "wheat2000_huber"}, "phase": "dev"}, summary=True)
            _run(results, "wheat599/neural/fail_run_20261101",
                 prov={"config": {"model": "x"}}, failures={"failures": [{"fold": 0}]})
            # nested summary under a provenance run must not be counted twice
            _run(results, "wheat599/neural/quantile_tabm_final_20261106/test", summary=True)

            index = run_index.build_index(root)
            again = run_index.build_index(root)
            self.assertEqual(index, again)
            by_id = {r["run_id"]: r for r in index["runs"]}
            self.assertEqual(index["n_runs"], 7)
            self.assertNotIn("wheat599/neural/quantile_tabm_final_20261106/test", by_id)

            q = by_id["wheat599/neural/quantile_tabm_final_20261106"]
            self.assertEqual((q["method"], q["phase"], q["status"], q["role"]),
                             ("quantile_tabm_final", "refit", "complete", "candidate"))
            self.assertEqual(q["source_commit"], "abc")
            g = by_id["wheat599/baselines/gsformer_nsa_outer_20261115"]
            self.assertEqual((g["phase"], g["role"], g["status"]), ("outer_eval", "baseline", "complete"))
            b = by_id["wheat599/baselines/background_ridge_20261102"]
            self.assertEqual(b["method"], "background_ridge")
            d = by_id["wheat599/diagnostics/bimsgp_diag_ftt_20261029"]
            self.assertEqual((d["phase"], d["role"]), ("diagnostic", "diagnostic"))
            r = by_id["wheat599/kernels/robust_icm_development_20261103"]
            self.assertEqual((r["phase"], r["status"]), ("development", "running"))
            w = by_id["wheat2000/wheat2000_huber_dev_20261115"]
            self.assertEqual((w["phase"], w["role"]), ("development", "cross_dataset"))
            f = by_id["wheat599/neural/fail_run_20261101"]
            self.assertEqual(f["status"], "failed")

    def test_main_writes_only_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _run(root / "results", "wheat599/neural/m_dev_20261101",
                       prov={"config": {"model": "m"}}, summary=True)
            before = {p: p.read_bytes() for p in run.iterdir()}
            run_index.main(["--root", str(root)])
            self.assertTrue((root / "results" / "run_index.json").exists())
            for p, data in before.items():
                self.assertEqual(p.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()