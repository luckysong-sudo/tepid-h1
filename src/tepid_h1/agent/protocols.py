from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    ok: bool
    output: Any = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalAnswer:
    content: str
    evidence_call_ids: tuple[str, ...] = ()


AgentAction: TypeAlias = ToolCall | FinalAnswer


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str = ""


@dataclass
class RuntimeState:
    task: str
    max_steps: int
    steps: int = 0
    observations: list[Any] = field(default_factory=list)
    actions: list[AgentAction] = field(default_factory=list)


class AgentModel(Protocol):
    def generate_action(self, context: Any) -> AgentAction: ...


class ContextBuilder(Protocol):
    def build(self, state: RuntimeState) -> Any: ...


class PolicyEngine(Protocol):
    def authorize(self, state: RuntimeState, call: ToolCall) -> PolicyDecision: ...


class ToolGateway(Protocol):
    def execute(self, call: ToolCall) -> ToolResult: ...


class Verifier(Protocol):
    def verify_final(self, state: RuntimeState, answer: FinalAnswer) -> PolicyDecision: ...


class Telemetry(Protocol):
    def record(self, state: RuntimeState, action: AgentAction, result: Any) -> None: ...

