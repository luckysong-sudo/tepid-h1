from __future__ import annotations

from dataclasses import dataclass

from .protocols import (
    AgentModel,
    ContextBuilder,
    FinalAnswer,
    PolicyEngine,
    RuntimeState,
    Telemetry,
    ToolCall,
    ToolGateway,
    Verifier,
)


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeDependencies:
    model: AgentModel
    context_builder: ContextBuilder
    policy: PolicyEngine
    tools: ToolGateway
    verifier: Verifier
    telemetry: Telemetry | None = None


class AgentRuntime:
    """Bounded, policy-checked and verifier-gated execution loop."""

    def __init__(self, dependencies: RuntimeDependencies) -> None:
        self.dependencies = dependencies

    def run(self, task: str, *, max_steps: int = 32) -> FinalAnswer:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        state = RuntimeState(task=task, max_steps=max_steps)

        while state.steps < state.max_steps:
            context = self.dependencies.context_builder.build(state)
            action = self.dependencies.model.generate_action(context)
            state.steps += 1
            state.actions.append(action)

            if isinstance(action, FinalAnswer):
                verdict = self.dependencies.verifier.verify_final(state, action)
                self._record(state, action, verdict)
                if verdict.allowed:
                    return action
                state.observations.append({"verification_error": verdict.reason})
                continue

            if not isinstance(action, ToolCall):
                raise TypeError(f"model emitted unsupported action: {type(action)!r}")

            decision = self.dependencies.policy.authorize(state, action)
            if not decision.allowed:
                observation = {"policy_denial": decision.reason, "call_id": action.call_id}
                state.observations.append(observation)
                self._record(state, action, observation)
                continue

            result = self.dependencies.tools.execute(action)
            if result.call_id != action.call_id:
                raise ValueError("tool result call_id does not match the requested call")
            state.observations.append(result)
            self._record(state, action, result)

        raise BudgetExceeded(f"agent exhausted {max_steps} steps without a verified final answer")

    def _record(self, state: RuntimeState, action: object, result: object) -> None:
        if self.dependencies.telemetry is not None:
            self.dependencies.telemetry.record(state, action, result)

