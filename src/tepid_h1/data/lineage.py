"""Data lineage tracking and license compatibility checking.

This module provides tools to track the lineage of data sources through
processing pipelines and to verify that license categories are compatible
with intended usage (e.g., commercial training).  These tools complement
the existing inventory audit and decontamination checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


# License category compatibility matrix
# Maps each license category to the set of usage contexts it permits
_LICENSE_COMPATIBILITY: dict[str, frozenset[str]] = {
    "permissive": frozenset({"commercial", "research", "redistribution"}),
    "public_domain": frozenset({"commercial", "research", "redistribution"}),
    "contracted": frozenset({"commercial", "research", "redistribution"}),
    "restricted": frozenset({"research"}),
    "prohibited": frozenset(),
    "unknown": frozenset(),
}

_VALID_USAGE_CONTEXTS = frozenset({"commercial", "research", "redistribution"})


@dataclass(frozen=True)
class LineageEntry:
    """A single lineage transformation record."""

    stage: str
    input_source_ids: tuple[str, ...]
    output_source_id: str
    transformation: str
    timestamp: str
    operator: str = "unknown"
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LineageReport:
    """Report of a complete data lineage chain."""

    schema_version: int
    source_id: str
    entries: tuple[LineageEntry, ...]
    chain_length: int
    root_source_ids: tuple[str, ...]
    passed: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "chain_length": self.chain_length,
            "root_source_ids": list(self.root_source_ids),
            "passed": self.passed,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class LicenseCompatibilityReport:
    """Report of license compatibility checks for a set of sources."""

    schema_version: int
    usage_context: str
    checked_sources: int
    compatible_sources: int
    incompatible_sources: int
    passed: bool
    findings: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "usage_context": self.usage_context,
            "checked_sources": self.checked_sources,
            "compatible_sources": self.compatible_sources,
            "incompatible_sources": self.incompatible_sources,
            "passed": self.passed,
            "findings": [dict(f) for f in self.findings],
        }


class LineageTracker:
    """Tracks data lineage through processing stages.

    Records each transformation applied to a data source, building an
    auditable chain from root sources to final outputs.  The tracker
    validates that every input source has been registered before it can
    be used as an input to a transformation.
    """

    def __init__(self) -> None:
        self._entries: list[LineageEntry] = []
        self._registered_sources: set[str] = set()
        self._root_sources: set[str] = set()

    def register_root_source(self, source_id: str) -> None:
        """Register a root data source that has no upstream inputs."""
        if not source_id or not source_id.strip():
            raise ValueError("source_id must not be empty")
        source_id = source_id.strip()
        if source_id in self._registered_sources:
            raise ValueError(f"source {source_id!r} is already registered")
        self._registered_sources.add(source_id)
        self._root_sources.add(source_id)

    def record_transformation(
        self,
        stage: str,
        input_source_ids: tuple[str, ...] | list[str],
        output_source_id: str,
        transformation: str,
        *,
        operator: str = "unknown",
        parameters: dict[str, Any] | None = None,
    ) -> LineageEntry:
        """Record a data transformation in the lineage chain."""
        if not stage.strip():
            raise ValueError("stage must not be empty")
        if not input_source_ids:
            raise ValueError("input_source_ids must not be empty")
        if not output_source_id.strip():
            raise ValueError("output_source_id must not be empty")
        if not transformation.strip():
            raise ValueError("transformation must not be empty")

        input_ids = tuple(input_source_ids)
        for input_id in input_ids:
            if input_id not in self._registered_sources:
                raise ValueError(
                    f"input source {input_id!r} is not registered; "
                    "register it before using it as a transformation input"
                )

        output_id = output_source_id.strip()
        if output_id in self._registered_sources:
            raise ValueError(f"output source {output_id!r} is already registered")

        entry = LineageEntry(
            stage=stage.strip(),
            input_source_ids=input_ids,
            output_source_id=output_id,
            transformation=transformation.strip(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            operator=operator,
            parameters=dict(parameters or {}),
        )
        self._entries.append(entry)
        self._registered_sources.add(output_id)
        return entry

    def get_lineage(self, source_id: str) -> LineageReport:
        """Get the full lineage chain for a source."""
        if source_id not in self._registered_sources:
            raise ValueError(f"source {source_id!r} is not registered")

        entries: list[LineageEntry] = []
        errors: list[str] = []
        root_ids = self._trace_roots(source_id, entries, errors, set())

        return LineageReport(
            schema_version=1,
            source_id=source_id,
            entries=tuple(entries),
            chain_length=len(entries),
            root_source_ids=tuple(sorted(root_ids)),
            passed=not errors,
            errors=tuple(errors),
        )

    def _trace_roots(
        self,
        source_id: str,
        entries: list[LineageEntry],
        errors: list[str],
        visited: set[str],
    ) -> set[str]:
        """Recursively trace root sources for a given source ID."""
        if source_id in visited:
            errors.append(f"cycle detected at source {source_id!r}")
            return set()
        visited.add(source_id)

        if source_id in self._root_sources:
            return {source_id}

        roots: set[str] = set()
        for entry in self._entries:
            if entry.output_source_id == source_id:
                entries.append(entry)
                for input_id in entry.input_source_ids:
                    roots.update(
                        self._trace_roots(input_id, entries, errors, visited.copy())
                    )
        if not roots and source_id not in self._root_sources:
            errors.append(f"source {source_id!r} has no recorded inputs and is not a root")
        return roots

    @property
    def registered_sources(self) -> frozenset[str]:
        return frozenset(self._registered_sources)

    @property
    def root_sources(self) -> frozenset[str]:
        return frozenset(self._root_sources)


def check_license_compatibility(
    sources: list[dict[str, Any]],
    *,
    usage_context: str = "commercial",
) -> LicenseCompatibilityReport:
    """Check that all sources have licenses compatible with the intended usage.

    Args:
        sources: List of source dictionaries with 'id' and 'license_category' keys.
        usage_context: The intended usage context ('commercial', 'research',
            or 'redistribution').

    Returns:
        A LicenseCompatibilityReport with per-source findings.
    """
    if usage_context not in _VALID_USAGE_CONTEXTS:
        raise ValueError(
            f"usage_context must be one of {sorted(_VALID_USAGE_CONTEXTS)}"
        )
    if not sources:
        raise ValueError("sources must not be empty")

    findings: list[dict[str, Any]] = []
    compatible = 0
    incompatible = 0

    for index, source in enumerate(sources):
        source_id = source.get("id", f"index:{index}")
        category = source.get("license_category", "unknown")

        if category not in _LICENSE_COMPATIBILITY:
            findings.append(
                {
                    "source_id": source_id,
                    "license_category": category,
                    "compatible": False,
                    "reason": f"unknown license category {category!r}",
                }
            )
            incompatible += 1
            continue

        allowed_contexts = _LICENSE_COMPATIBILITY[category]
        is_compatible = usage_context in allowed_contexts
        if is_compatible:
            findings.append(
                {
                    "source_id": source_id,
                    "license_category": category,
                    "compatible": True,
                    "reason": "",
                }
            )
            compatible += 1
        else:
            findings.append(
                {
                    "source_id": source_id,
                    "license_category": category,
                    "compatible": False,
                    "reason": (
                        f"license category {category!r} does not permit "
                        f"{usage_context!r} usage"
                    ),
                }
            )
            incompatible += 1

    return LicenseCompatibilityReport(
        schema_version=1,
        usage_context=usage_context,
        checked_sources=len(sources),
        compatible_sources=compatible,
        incompatible_sources=incompatible,
        passed=incompatible == 0,
        findings=tuple(findings),
    )


