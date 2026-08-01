from __future__ import annotations

import unittest

from tepid_h1.stage_gates import EXPECTED_STAGE_ORDER, audit_stage_gates, load_stage_gates


class StageGateTests(unittest.TestCase):
    def test_example_stage_gates_pass(self) -> None:
        report = audit_stage_gates(load_stage_gates("configs/stage_gates.json"))

        self.assertTrue(report.passed)
        self.assertEqual(tuple(gate.name for gate in report.gates), EXPECTED_STAGE_ORDER)
        self.assertEqual(report.errors, ())

    def test_missing_stage_gate_fails(self) -> None:
        payload = load_stage_gates("configs/stage_gates.json")
        payload.pop("M5_formal_training")

        report = audit_stage_gates(payload)

        self.assertFalse(report.passed)
        self.assertTrue(any("missing stage gates" in error for error in report.errors))

    def test_empty_deliverables_fail_closed(self) -> None:
        payload = load_stage_gates("configs/stage_gates.json")
        payload["M0_data_tokenizer"]["deliverables"] = []

        report = audit_stage_gates(payload)

        self.assertFalse(report.passed)
        self.assertIn("M0_data_tokenizer.deliverables", report.errors[0])


if __name__ == "__main__":
    unittest.main()
