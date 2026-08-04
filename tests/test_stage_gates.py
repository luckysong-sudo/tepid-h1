from __future__ import annotations

import unittest

from tepid_h1.stage_gates import EXPECTED_STAGE_ORDER, audit_stage_gates, load_stage_gates


class StageGateTests(unittest.TestCase):
    def test_example_stage_gates_pass(self) -> None:
        report = audit_stage_gates(
            load_stage_gates("configs/stage_gates.json"),
            evidence_root=".",
        )

        self.assertTrue(report.passed)
        self.assertEqual(tuple(gate.name for gate in report.gates), EXPECTED_STAGE_ORDER)
        self.assertEqual(report.errors, ())
        self.assertTrue(all(gate.evidence_refs for gate in report.gates))

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

    def test_missing_evidence_refs_fail_closed(self) -> None:
        payload = load_stage_gates("configs/stage_gates.json")
        payload["M4_moe_prototype"]["evidence_refs"] = []

        report = audit_stage_gates(payload)

        self.assertFalse(report.passed)
        self.assertTrue(any("M4_moe_prototype.evidence_refs" in error for error in report.errors))

    def test_unsupported_evidence_ref_scheme_fails(self) -> None:
        payload = load_stage_gates("configs/stage_gates.json")
        payload["M4_moe_prototype"]["evidence_refs"] = ["url:https://example.invalid/report"]

        report = audit_stage_gates(payload)

        self.assertFalse(report.passed)
        self.assertTrue(any("unsupported scheme" in error for error in report.errors))

    def test_invalid_cli_evidence_ref_fails_when_validator_is_supplied(self) -> None:
        payload = load_stage_gates("configs/stage_gates.json")
        payload["M4_moe_prototype"]["evidence_refs"] = ["cli:tepid-h1 missing-command"]

        report = audit_stage_gates(
            payload,
            cli_validator=lambda command: "unknown command" if command else None,
        )

        self.assertFalse(report.passed)
        self.assertTrue(any("invalid CLI ref" in error for error in report.errors))

    def test_missing_file_evidence_ref_fails_when_root_is_checked(self) -> None:
        payload = load_stage_gates("configs/stage_gates.json")
        payload["M4_moe_prototype"]["evidence_refs"] = ["file:docs/DOES_NOT_EXIST.md"]

        report = audit_stage_gates(payload, evidence_root=".")

        self.assertFalse(report.passed)
        self.assertTrue(any("file ref does not exist" in error for error in report.errors))

    def test_escaping_file_evidence_ref_fails_when_root_is_checked(self) -> None:
        payload = load_stage_gates("configs/stage_gates.json")
        payload["M4_moe_prototype"]["evidence_refs"] = ["file:../outside.md"]

        report = audit_stage_gates(payload, evidence_root=".")

        self.assertFalse(report.passed)
        self.assertTrue(any("escapes evidence root" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
