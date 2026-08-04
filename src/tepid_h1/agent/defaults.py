from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .protocols import (
    AgentAction,
    FinalAnswer,
    PolicyDecision,
    RuntimeState,
    ToolCall,
    ToolResult,
)


class PolicyEngineLike(Protocol):
    """Structural protocol for policy engines usable in composite policies."""

    def authorize(self, state: RuntimeState, call: ToolCall) -> PolicyDecision: ...


class StateContextBuilder:
    """Context builder that returns the runtime state directly.

    The simplest context builder: the model receives the full ``RuntimeState``,
    including observations and actions, and must decide the next action from it.
    """

    def build(self, state: RuntimeState) -> RuntimeState:
        return state


class AllowlistPolicy:
    """Policy engine that permits a fixed set of tool names.

    Calls to unregistered tools are denied with a descriptive reason. This is a
    fail-closed default: an empty allowlist denies every tool call.
    """

    def __init__(self, allowed_tools: Mapping[str, str] | set[str] | None = None) -> None:
        if allowed_tools is None:
            self._descriptions: dict[str, str] = {}
        elif isinstance(allowed_tools, Mapping):
            self._descriptions = dict(allowed_tools)
        else:
            self._descriptions = {name: "" for name in allowed_tools}

    @property
    def allowed_tools(self) -> frozenset[str]:
        return frozenset(self._descriptions)

    def authorize(self, state: RuntimeState, call: ToolCall) -> PolicyDecision:
        if call.tool_name in self._descriptions:
            return PolicyDecision(allowed=True)
        return PolicyDecision(
            allowed=False,
            reason=f"tool {call.tool_name!r} is not in the allowlist",
        )


class RateLimitPolicy:
    """Policy that limits how many times a tool can be called per run.

    Wraps an optional inner policy: when the inner policy allows a call, the
    rate limiter checks whether the tool has already been invoked up to its
    configured maximum.  This is a fail-closed default: an empty rate map
    denies every tool call unless no limits are configured.
    """

    def __init__(
        self,
        limits: Mapping[str, int] | None = None,
        *,
        inner: PolicyEngineLike | None = None,
    ) -> None:
        self._limits: dict[str, int] = {}
        if limits is not None:
            for name, count in limits.items():
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError(f"rate limit for {name!r} must be a non-negative integer")
                self._limits[name] = count
        self._inner = inner
        self._call_counts: dict[str, int] = {}

    def authorize(self, state: RuntimeState, call: ToolCall) -> PolicyDecision:
        if self._inner is not None:
            inner_decision = self._inner.authorize(state, call)
            if not inner_decision.allowed:
                return inner_decision
        limit = self._limits.get(call.tool_name)
        if limit is None:
            return PolicyDecision(allowed=True)
        current = self._call_counts.get(call.tool_name, 0)
        if current >= limit:
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"tool {call.tool_name!r} has reached its rate limit "
                    f"of {limit} calls"
                ),
            )
        self._call_counts[call.tool_name] = current + 1
        return PolicyDecision(allowed=True)

    def reset(self) -> None:
        """Clear call counts so the policy can be reused across runs."""
        self._call_counts.clear()


class CompositePolicy:
    """Policy that requires every inner policy to allow the call.

    The first denial short-circuits and its reason is returned.  This lets
    callers compose allowlist, rate-limit and custom policies without
    modifying any single policy implementation.
    """

    def __init__(self, policies: Sequence[PolicyEngineLike]) -> None:
        if not policies:
            raise ValueError("CompositePolicy requires at least one policy")
        self._policies = list(policies)

    def authorize(self, state: RuntimeState, call: ToolCall) -> PolicyDecision:
        for policy in self._policies:
            decision = policy.authorize(state, call)
            if not decision.allowed:
                return decision
        return PolicyDecision(allowed=True)


class ToolSchemaValidator:
    """Validates tool call arguments against a per-tool required-keys schema.

    This is a fail-closed validator: if a tool is registered with a schema,
    every call to that tool must include all required keys.  Unknown tools
    with no schema are allowed to pass through.
    """

    def __init__(self, schemas: Mapping[str, frozenset[str]] | None = None) -> None:
        self._schemas: dict[str, frozenset[str]] = dict(schemas or {})

    def validate(self, call: ToolCall) -> ToolResult | None:
        """Return a failure ToolResult if validation fails, else None."""
        schema = self._schemas.get(call.tool_name)
        if schema is None:
            return None
        missing = sorted(schema - set(call.arguments))
        if missing:
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=(
                    f"tool {call.tool_name!r} is missing required arguments: "
                    f"{', '.join(missing)}"
                ),
            )
        return None


class ToolRegistry:
    """Tool gateway backed by a registry of callables.

    Each registered callable receives the call's ``arguments`` mapping and
    returns the tool output. Callables that raise convert the exception into a
    ``ToolResult`` with ``ok=False`` and the error message, so the runtime loop
    never crashes on a tool failure.
    """

    def __init__(
        self,
        tools: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None,
        *,
        schema_validator: ToolSchemaValidator | None = None,
    ) -> None:
        self._tools: dict[str, Callable[[Mapping[str, Any]], Any]] = dict(tools or {})
        self._schema_validator = schema_validator

    def register(
        self,
        name: str,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> None:
        if not name.strip():
            raise ValueError("tool name must be non-empty")
        self._tools[name] = handler

    @property
    def registered_tools(self) -> frozenset[str]:
        return frozenset(self._tools)

    def execute(self, call: ToolCall) -> ToolResult:
        handler = self._tools.get(call.tool_name)
        if handler is None:
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=f"tool {call.tool_name!r} is not registered",
            )
        if self._schema_validator is not None:
            validation_failure = self._schema_validator.validate(call)
            if validation_failure is not None:
                return validation_failure
        try:
            output = handler(call.arguments)
        except Exception as error:  # noqa: BLE001 - tool errors are observable, not fatal
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=f"{type(error).__name__}: {error}",
            )
        return ToolResult(call_id=call.call_id, ok=True, output=output)


class EvidenceVerifier:
    """Verifier that requires every evidence call id to have a successful result.

    A final answer is accepted only when each id in ``evidence_call_ids`` maps to
    a ``ToolResult`` with ``ok=True`` in the runtime observations. This binds a
    claim to its supporting tool evidence without coupling the verifier to any
    specific tool semantics.
    """

    def verify_final(self, state: RuntimeState, answer: FinalAnswer) -> PolicyDecision:
        successful = {
            item.call_id for item in state.observations if isinstance(item, ToolResult) and item.ok
        }
        missing = sorted(set(answer.evidence_call_ids) - successful)
        if missing:
            return PolicyDecision(
                allowed=False,
                reason=f"missing successful evidence for call ids: {', '.join(missing)}",
            )
        return PolicyDecision(allowed=True)


class ContentLengthVerifier:
    """Verifier that enforces minimum and maximum content length on final answers.

    This verifier can be composed with ``EvidenceVerifier`` via a custom
    verifier implementation or used standalone to guard against degenerate
    empty or overly long answers.
    """

    def __init__(
        self,
        *,
        min_length: int = 1,
        max_length: int = 10_000,
        inner: "VerifierLike | None" = None,
    ) -> None:
        if not isinstance(min_length, int) or isinstance(min_length, bool) or min_length < 0:
            raise ValueError("min_length must be a non-negative integer")
        if (
            not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or max_length < min_length
        ):
            raise ValueError("max_length must be an integer >= min_length")
        self._min_length = min_length
        self._max_length = max_length
        self._inner = inner

    def verify_final(self, state: RuntimeState, answer: FinalAnswer) -> PolicyDecision:
        if self._inner is not None:
            inner_decision = self._inner.verify_final(state, answer)
            if not inner_decision.allowed:
                return inner_decision
        length = len(answer.content)
        if length < self._min_length:
            return PolicyDecision(
                allowed=False,
                reason=f"answer content length {length} is below minimum {self._min_length}",
            )
        if length > self._max_length:
            return PolicyDecision(
                allowed=False,
                reason=f"answer content length {length} exceeds maximum {self._max_length}",
            )
        return PolicyDecision(allowed=True)


class VerifierLike(Protocol):
    """Structural protocol for verifiers usable in composite verifiers."""

    def verify_final(self, state: RuntimeState, answer: FinalAnswer) -> PolicyDecision: ...


@dataclass
class ListTelemetry:
    """Telemetry sink that appends every event to an in-memory list.

    Useful for tests and audits: the full action/result trace is available for
    inspection after the run completes.
    """

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, state: RuntimeState, action: AgentAction, result: Any) -> None:
        self.events.append(
            {
                "step": state.steps,
                "action": _serialize_action(action),
                "result": _serialize_result(result),
            }
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact summary of the recorded telemetry events."""
        total = len(self.events)
        tool_calls = sum(1 for e in self.events if e["action"]["kind"] == "tool_call")
        final_answers = sum(1 for e in self.events if e["action"]["kind"] == "final_answer")
        failures = sum(
            1
            for e in self.events
            if e["result"].get("kind") == "tool_result" and not e["result"].get("ok", True)
        )
        return {
            "total_events": total,
            "tool_calls": tool_calls,
            "final_answers": final_answers,
            "tool_failures": failures,
        }


def _serialize_action(action: AgentAction) -> dict[str, Any]:
    if isinstance(action, ToolCall):
        return {
            "kind": "tool_call",
            "call_id": action.call_id,
            "tool_name": action.tool_name,
            "arguments": dict(action.arguments),
        }
    if isinstance(action, FinalAnswer):
        return {
            "kind": "final_answer",
            "content": action.content,
            "evidence_call_ids": list(action.evidence_call_ids),
        }
    return {"kind": "unknown", "type": type(action).__name__}


def _serialize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, ToolResult):
        return {
            "kind": "tool_result",
            "call_id": result.call_id,
            "ok": result.ok,
            "error": result.error,
        }
    if isinstance(result, Mapping):
        return {"kind": "mapping", "data": dict(result)}
    return {"kind": "other", "type": type(result).__name__}


