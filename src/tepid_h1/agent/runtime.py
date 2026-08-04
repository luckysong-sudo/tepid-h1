from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .protocols import (
    AgentAction,
    AgentModel,
    ContextBuilder,
    FinalAnswer,
    ModelValidationError,
    PolicyEngine,
    RuntimeState,
    Telemetry,
    ToolCall,
    ToolGateway,
    Verifier,
)

T = TypeVar("T")


class BudgetExceeded(RuntimeError):
    """Raised when the agent exhausts its step budget."""


class RetryExhausted(RuntimeError):
    """Raised when an operation exceeds its retry limit."""


@dataclass(frozen=True)
class RuntimeDependencies:
    """Dependencies required to run the agent loop."""
    model: AgentModel
    context_builder: ContextBuilder
    policy: PolicyEngine
    tools: ToolGateway
    verifier: Verifier
    telemetry: Telemetry | None = None


class AgentRuntime:
    """Bounded, policy-checked and verifier-gated execution loop."""

    def __init__(
        self,
        dependencies: RuntimeDependencies,
        *,
        max_retries: int = 3,
        retry_backoff: float = 0.1,
    ) -> None:
        self.dependencies = dependencies
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must be non-negative")
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    def run(self, task: str, *, max_steps: int = 32) -> FinalAnswer:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        state = RuntimeState(task=task, max_steps=max_steps)

        while state.can_proceed():
            context = self._run_with_retry(self.dependencies.context_builder.build, state)
            try:
                action = self.dependencies.model.generate_action(context)
            except Exception as exc:
                raise ModelValidationError(
                    f"model generated invalid action: {exc!r}"
                ) from exc

            state.record_action(action)

            if isinstance(action, FinalAnswer):
                verdict = self._run_with_retry(
                    self.dependencies.verifier.verify_final, state, action
                )
                self._record(state, action, verdict)
                if verdict.allowed:
                    return action
                state.record_observation({"verification_error": verdict.reason})
                continue

            if not isinstance(action, ToolCall):
                raise ModelValidationError(
                    f"model emitted unsupported action type: {type(action)!r}"
                )

            decision = self._run_with_retry(self.dependencies.policy.authorize, state, action)
            if not decision.allowed:
                observation = {"policy_denial": decision.reason, "call_id": action.call_id}
                state.record_observation(observation)
                self._record(state, action, observation)
                continue

            result = self._run_with_retry(self.dependencies.tools.execute, action)
            if result.call_id != action.call_id:
                raise ModelValidationError(
                    "tool result call_id does not match the requested call"
                )
            state.record_observation(result)
            self._record(state, action, result)

        raise BudgetExceeded(
            f"agent exhausted {max_steps} steps without a verified final answer"
        )

    def _run_with_retry(
        self, operation: Callable[..., T], *args: object, **kwargs: object
    ) -> T:
        for attempt in range(self._max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except (ModelValidationError, RetryExhausted):
                raise
            except Exception as exc:
                if attempt == self._max_retries:
                    raise RetryExhausted(
                        f"operation failed after {self._max_retries + 1} attempts: {exc!r}"
                    ) from exc
                time.sleep(self._retry_backoff * (2 ** attempt))

        raise AssertionError("retry loop did not execute")

    def _record(self, state: RuntimeState, action: AgentAction, result: object) -> None:
        if self.dependencies.telemetry is not None:
            self.dependencies.telemetry.record(state, action, result)
