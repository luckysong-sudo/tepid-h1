from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .protocols import (
    AgentAction,
    FinalAnswer,
    PolicyDecision,
    RuntimeState,
    ToolCall,
    ToolResult,
)


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


class ToolRegistry:
    """Tool gateway backed by a registry of callables.

    Each registered callable receives the call's ``arguments`` mapping and
    returns the tool output. Callables that raise convert the exception into a
    ``ToolResult`` with ``ok=False`` and the error message, so the runtime loop
    never crashes on a tool failure.
    """

    def __init__(
        self, tools: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None
    ) -> None:
        self._tools: dict[str, Callable[[Mapping[str, Any]], Any]] = dict(tools or {})

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
