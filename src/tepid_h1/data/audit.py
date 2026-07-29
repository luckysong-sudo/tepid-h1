from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ALLOWED_LICENSE_CATEGORIES = {
    "permissive",
    "public_domain",
    "contracted",
    "restricted",
    "prohibited",
    "unknown",
}
APPROVED_LICENSE_CATEGORIES = {"permissive", "public_domain", "contracted"}
ALLOWED_PII_STATUSES = {"absent", "removed", "present", "unassessed"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    code: str
    message: str
    source_id: str | None = None


@dataclass(frozen=True)
class AuditReport:
    inventory_id: str
    passed: bool
    source_count: int
    approved_source_count: int
    estimated_tokens: int
    findings: tuple[AuditFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["findings"] = [asdict(finding) for finding in self.findings]
        return payload


def load_inventory(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("data inventory root must be a JSON object")
    return payload


def audit_inventory(payload: dict[str, Any]) -> AuditReport:
    findings: list[AuditFinding] = []
    inventory_id = _text(payload.get("inventory_id")) or "unknown"
    if payload.get("schema_version") != 1:
        findings.append(_error("schema_version", "schema_version must equal 1"))
    for field in ("inventory_id", "owner", "generated_at"):
        if not _text(payload.get(field)):
            findings.append(_error("required_field", f"inventory field {field!r} is required"))

    sources = payload.get("sources")
    if not isinstance(sources, list):
        findings.append(_error("sources_type", "sources must be an array"))
        sources = []
    if not sources:
        findings.append(_error("sources_empty", "at least one data source is required"))

    seen_ids: set[str] = set()
    approved_count = 0
    estimated_tokens = 0
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            findings.append(_error("source_type", f"source at index {index} must be an object"))
            continue
        source_id = _text(source.get("id")) or f"index:{index}"
        source_findings = _audit_source(source, source_id)
        findings.extend(source_findings)
        if source_id in seen_ids:
            findings.append(_error("duplicate_source_id", "source id must be unique", source_id))
        seen_ids.add(source_id)
        estimated_tokens += _positive_int(source.get("estimated_tokens")) or 0
        if not any(item.severity == "error" for item in source_findings):
            approved_count += 1

    findings.extend(_audit_repository_decontamination(payload.get("repository_decontamination")))
    return AuditReport(
        inventory_id=inventory_id,
        passed=not any(item.severity == "error" for item in findings),
        source_count=len(sources),
        approved_source_count=approved_count,
        estimated_tokens=estimated_tokens,
        findings=tuple(findings),
    )


def _audit_source(source: dict[str, Any], source_id: str) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for field in ("id", "name", "uri", "snapshot", "license_id"):
        if not _text(source.get(field)):
            findings.append(_error("source_required_field", f"{field!r} is required", source_id))

    category = source.get("license_category")
    if category not in ALLOWED_LICENSE_CATEGORIES:
        findings.append(
            _error(
                "license_category",
                f"license_category must be one of {sorted(ALLOWED_LICENSE_CATEGORIES)}",
                source_id,
            )
        )
    elif category not in APPROVED_LICENSE_CATEGORIES:
        findings.append(
            _error(
                "license_not_approved",
                f"license category {category!r} is not approved for training",
                source_id,
            )
        )
    if source.get("commercial_use") is not True:
        findings.append(
            _error(
                "commercial_use",
                "commercial_use must be explicitly true before inclusion",
                source_id,
            )
        )
    if not _text(source.get("rights_evidence")):
        findings.append(_error("rights_evidence", "rights_evidence is required", source_id))

    checksum = _text(source.get("sha256"))
    if not checksum or not SHA256_PATTERN.fullmatch(checksum):
        findings.append(
            _error("sha256", "sha256 must be a lowercase 64-character digest", source_id)
        )
    if _positive_int(source.get("estimated_tokens")) is None:
        findings.append(
            _error("estimated_tokens", "estimated_tokens must be a positive integer", source_id)
        )
    for field in ("languages", "domains"):
        value = source.get(field)
        if not isinstance(value, list) or not value or not all(_text(item) for item in value):
            findings.append(
                _error(
                    "source_classification", f"{field} must be a non-empty string array", source_id
                )
            )

    pii_status = source.get("pii_status")
    if pii_status not in ALLOWED_PII_STATUSES:
        findings.append(
            _error(
                "pii_status",
                f"pii_status must be one of {sorted(ALLOWED_PII_STATUSES)}",
                source_id,
            )
        )
    elif pii_status in {"present", "unassessed"}:
        findings.append(
            _error("pii_not_cleared", f"PII status {pii_status!r} blocks inclusion", source_id)
        )
    elif pii_status == "removed" and not _text(source.get("pii_report")):
        findings.append(
            _error("pii_report", "pii_report is required when PII was removed", source_id)
        )

    if source.get("quality_status") not in {"accepted", "conditional"}:
        findings.append(
            _error(
                "quality_status",
                "quality_status must be 'accepted' or 'conditional'",
                source_id,
            )
        )
    return findings


def _audit_repository_decontamination(value: Any) -> list[AuditFinding]:
    if not isinstance(value, dict):
        return [_error("decontamination", "repository_decontamination object is required")]
    findings: list[AuditFinding] = []
    if value.get("status") != "complete":
        findings.append(
            _error("decontamination_status", "repository decontamination must be complete")
        )
    for field in ("method", "report_uri", "completed_at"):
        if not _text(value.get(field)):
            findings.append(
                _error("decontamination_field", f"decontamination field {field!r} is required")
            )
    benchmark_sets = value.get("benchmark_sets")
    if not isinstance(benchmark_sets, list) or not benchmark_sets:
        findings.append(
            _error("benchmark_sets", "at least one decontamination benchmark set is required")
        )
    return findings


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _error(code: str, message: str, source_id: str | None = None) -> AuditFinding:
    return AuditFinding(severity="error", code=code, message=message, source_id=source_id)
