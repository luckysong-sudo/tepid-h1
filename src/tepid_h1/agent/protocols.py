from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias


# --- Role enum ---


class Role(str, Enum):
    """Message role in agent conversations."""

    USER = "user"
    ASSISTANT = "assistant"
    OBSERVATION = "observation"
    SYSTEM = "system"


# --- Validation helpers ---


def _validate_non_empty(value: str, name: str) -> str:
    """Validate a string is non-empty and strip whitespace."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must not be empty")
    return stripped


def _validate_positive(value: int, name: str) -> int:
    """Validate an integer is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


# --- Data classes ---


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_id", _validate_non_empty(self.call_id, "ToolCall.call_id"))
        object.__setattr__(
            self, "tool_name", _validate_non_empty(self.tool_name, "ToolCall.tool_name")
        )


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "content", _validate_non_empty(self.content, "FinalAnswer.content")
        )


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _validate_non_empty(self.task, "RuntimeState.task"))
        object.__setattr__(
            self, "max_steps", _validate_positive(self.max_steps, "RuntimeState.max_steps")
        )

    def can_proceed(self) -> bool:
        """Check if the agent can take more steps."""
        return self.steps < self.max_steps

    def record_observation(self, observation: Any) -> None:
        """Append an observation to the state."""
        self.observations.append(observation)

    def record_action(self, action: AgentAction) -> None:
        """Append an action to the state and increment step counter."""
        self.actions.append(action)
        self.steps += 1


class AgentModel(Protocol):
    """Protocol for models that generate agent actions."""

    def generate_action(self, context: Any) -> AgentAction: ...


class ContextBuilder(Protocol):
    """Protocol for building context from runtime state."""

    def build(self, state: RuntimeState) -> Any: ...


class PolicyEngine(Protocol):
    """Protocol for authorizing tool calls."""

    def authorize(self, state: RuntimeState, call: ToolCall) -> PolicyDecision: ...


class ToolGateway(Protocol):
    """Protocol for executing tool calls."""

    def execute(self, call: ToolCall) -> ToolResult: ...


class Verifier(Protocol):
    """Protocol for verifying final answers."""

    def verify_final(self, state: RuntimeState, answer: FinalAnswer) -> PolicyDecision: ...


class Telemetry(Protocol):
    """Protocol for recording telemetry data."""

    def record(self, state: RuntimeState, action: AgentAction, result: Any) -> None: ...


class ModelValidationError(ValueError):
    """Raised when model output violates protocol constraints."""

    pass
