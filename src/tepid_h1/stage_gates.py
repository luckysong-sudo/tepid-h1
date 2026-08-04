from __future__ import annotations

import json
from collections.abc import Callable
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
    evidence_refs: tuple[str, ...]

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


def audit_stage_gates(
    payload: dict[str, Any],
    *,
    evidence_root: str | Path | None = None,
    cli_validator: Callable[[str], str | None] | None = None,
) -> StageGateReport:
    errors: list[str] = []
    gates: list[StageGate] = []
    root = Path(evidence_root).resolve() if evidence_root is not None else None

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
        evidence_refs = _string_tuple(gate.get("evidence_refs"), f"{name}.evidence_refs", errors)
        _validate_evidence_refs(
            evidence_refs,
            name,
            errors,
            evidence_root=root,
            cli_validator=cli_validator,
        )
        gates.append(
            StageGate(
                name=name,
                deliverables=deliverables,
                exit_criteria=exit_criteria,
                evidence_refs=evidence_refs,
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


def _validate_evidence_refs(
    refs: tuple[str, ...],
    gate_name: str,
    errors: list[str],
    *,
    evidence_root: Path | None,
    cli_validator: Callable[[str], str | None] | None,
) -> None:
    for index, ref in enumerate(refs):
        if ":" not in ref:
            errors.append(f"{gate_name}.evidence_refs[{index}] must include a ref scheme")
            continue
        scheme, value = ref.split(":", 1)
        if scheme not in {"cli", "file"}:
            errors.append(
                f"{gate_name}.evidence_refs[{index}] uses unsupported scheme: {scheme!r}"
            )
            continue
        if not value.strip():
            errors.append(f"{gate_name}.evidence_refs[{index}] must include a ref target")
            continue
        if scheme == "file" and evidence_root is not None:
            _validate_file_ref(value, gate_name, index, errors, evidence_root=evidence_root)
        if scheme == "cli" and cli_validator is not None:
            validation_error = cli_validator(value.strip())
            if validation_error is not None:
                errors.append(
                    f"{gate_name}.evidence_refs[{index}] invalid CLI ref: "
                    f"{validation_error}"
                )


def _validate_file_ref(
    value: str,
    gate_name: str,
    index: int,
    errors: list[str],
    *,
    evidence_root: Path,
) -> None:
    path = Path(value)
    if path.is_absolute():
        errors.append(f"{gate_name}.evidence_refs[{index}] file ref must be repository-relative")
        return
    resolved = (evidence_root / path).resolve()
    if not resolved.is_relative_to(evidence_root):
        errors.append(f"{gate_name}.evidence_refs[{index}] file ref escapes evidence root")
        return
    if not resolved.exists():
        errors.append(f"{gate_name}.evidence_refs[{index}] file ref does not exist: {value}")
