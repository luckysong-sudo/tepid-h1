"""Demonstration of Tepid-H1 agent runtime."""
from __future__ import annotations

from tepid_h1 import (
    AgentRuntime,
    FinalAnswer,
    RuntimeDependencies,
    RuntimeState,
    ToolCall,
    ToolResult,
)
from tepid_h1.agent.protocols import (
    ContextBuilder,
    PolicyDecision,
    PolicyEngine,
    Telemetry,
    Verifier,
)


class MockModel:
    """Mock LLM for demonstration."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    def generate_action(self, context: RuntimeState) -> ToolCall | FinalAnswer:
        if self.call_count >= len(self.responses):
            return FinalAnswer(content="No more responses")
        response = self.responses[self.call_count]
        self.call_count += 1

        if response.startswith("TOOL:"):
            tool_name = response[5:].strip()
            return ToolCall(tool_name=tool_name, arguments={"query": "test"})
        return FinalAnswer(content=response)


class MockContextBuilder(ContextBuilder):
    """Builds context from runtime state."""

    def build(self, state: RuntimeState) -> dict:
        return {
            "task": state.task,
            "steps": state.steps_used,
            "observations": state.observations,
        }


class MockPolicy(PolicyEngine):
    """Always allows tool calls."""

    def authorize(self, state: RuntimeState, action: ToolCall) -> PolicyDecision:
        return PolicyDecision(allowed=True, reason="policy allows all")


class MockTools:
    """Mock tool executor."""

    def execute(self, action: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=action.call_id,
            content=f"Result from {action.tool_name}",
            ok=True,
        )


class MockVerifier(Verifier):
    """Simple verifier that accepts any final answer."""

    def verify_final(self, state: RuntimeState, answer: FinalAnswer) -> PolicyDecision:
        return PolicyDecision(allowed=True, reason="answer accepted")


class MockTelemetry(Telemetry):
    """Records telemetry events."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, state: RuntimeState, action: object, result: object) -> None:
        self.events.append({
            "action_type": type(action).__name__,
            "steps": state.steps_used,
            "result_ok": getattr(result, "ok", None),
        })


def demo_basic_agent() -> None:
    """Demo basic agent execution."""
    model = MockModel([
        "TOOL: search",
        "Final answer: The result is 42",
    ])

    deps = RuntimeDependencies(
        model=model,
        context_builder=MockContextBuilder(),
        policy=MockPolicy(),
        tools=MockTools(),
        verifier=MockVerifier(),
    )

    runtime = AgentRuntime(deps)
    answer = runtime.run("What is the meaning of life?", max_steps=5)
    print(f"Agent answer: {answer.content}")
    print(f"Model calls: {model.call_count}")


def demo_tool_retry() -> None:
    """Demo agent with retry logic."""
    call_count = 0

    class FlakyTools:
        def execute(self, action: ToolCall) -> ToolResult:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Transient error")
            return ToolResult(
                call_id=action.call_id,
                content=f"Success after {call_count} attempts",
                ok=True,
            )

    model = MockModel(["TOOL: compute"])
    deps = RuntimeDependencies(
        model=model,
        context_builder=MockContextBuilder(),
        policy=MockPolicy(),
        tools=FlakyTools(),
        verifier=MockVerifier(),
    )

    runtime = AgentRuntime(deps, max_retries=3, retry_backoff=0.01)
    answer = runtime.run("Compute the result", max_steps=5)
    print(f"Retry demo answer: {answer.content}")
    print(f"Tool attempts: {call_count}")


def demo_telemetry() -> None:
    """Demo telemetry recording."""
    telemetry = MockTelemetry()
    model = MockModel([
        "TOOL: search",
        "Final answer: Done",
    ])

    deps = RuntimeDependencies(
        model=model,
        context_builder=MockContextBuilder(),
        policy=MockPolicy(),
        tools=MockTools(),
        verifier=MockVerifier(),
        telemetry=telemetry,
    )

    runtime = AgentRuntime(deps)
    runtime.run("Test task")
    print(f"Telemetry events: {len(telemetry.events)}")
    for event in telemetry.events:
        print(f"  - {event}")


if __name__ == "__main__":
    print("=" * 50)
    print("Tepid-H1 Agent Demo")
    print("=" * 50)

    print("\n1. Basic Agent Execution")
    demo_basic_agent()

    print("\n2. Tool Retry Demo")
    demo_tool_retry()

    print("\n3. Telemetry Demo")
    demo_telemetry()

    print("\n" + "=" * 50)
    print("Agent demo completed!")
    print("=" * 50)