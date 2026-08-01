from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPECTED_STAGE_ORDER = (
    "M0_data_tokenizer",
    "M1_350m_prototype",
    "M2_1_3b_ablation",
    "M3_7b_product_evidence",
    "M4_moe_prototype",
    "M5_formal_training",
)


@dataclass(frozen=True)
class StageGate:
    name: str
    deliverables: tuple[str, ...]
    exit_criteria: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageGateReport:
    schema_version: int
    passed: bool
    gates: tuple[StageGate, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "gates": [gate.to_dict() for gate in self.gates],
            "errors": list(self.errors),
        }


def load_stage_gates(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("stage gate config must be a JSON object")
    return payload


def audit_stage_gates(payload: dict[str, Any]) -> StageGateReport:
    errors: list[str] = []
    gates: list[StageGate] = []

    actual_order = tuple(payload)
    if actual_order != EXPECTED_STAGE_ORDER:
        errors.append(
            f"stage gate keys must appear in canonical M0-M5 order: {list(EXPECTED_STAGE_ORDER)}"
        )

    missing = [name for name in EXPECTED_STAGE_ORDER if name not in payload]
    if missing:
        errors.append(f"missing stage gates: {missing}")

    unexpected = [name for name in payload if name not in EXPECTED_STAGE_ORDER]
    if unexpected:
        errors.append(f"unexpected stage gates: {unexpected}")

    for name in EXPECTED_STAGE_ORDER:
        if name not in payload:
            continue
        gate = payload[name]
        if not isinstance(gate, dict):
            errors.append(f"{name} must be an object")
            continue
        deliverables = _string_tuple(gate.get("deliverables"), f"{name}.deliverables", errors)
        exit_criteria = _string_tuple(gate.get("exit_criteria"), f"{name}.exit_criteria", errors)
        gates.append(
            StageGate(
                name=name,
                deliverables=deliverables,
                exit_criteria=exit_criteria,
            )
        )

    return StageGateReport(
        schema_version=1,
        passed=not errors,
        gates=tuple(gates),
        errors=tuple(errors),
    )


def _string_tuple(value: Any, field_name: str, errors: list[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        errors.append(f"{field_name} must be a non-empty list")
        return ()
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{field_name}[{index}] must be a non-empty string")
            continue
        items.append(item)
    return tuple(items)
