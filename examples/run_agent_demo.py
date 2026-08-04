"""
Tepid-H1 Agent Runtime Example

Demonstrates the bounded agent execution loop with policy checks.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

from tepid_h1.agent import (
    AgentRuntime,
    AllowlistPolicy,
    EvidenceVerifier,
    FinalAnswer,
    ListTelemetry,
    RuntimeDependencies,
    StateContextBuilder,
    ToolCall,
    ToolRegistry,
)


@dataclass
class SimpleModel:
    """A mock model that generates simple responses."""

    def generate_action(self, context: Any) -> ToolCall | FinalAnswer:
        if not context.observations:
            return ToolCall(call_id="1", tool_name="compute", arguments={"task": context.task})
        return FinalAnswer(content="The answer is 4", evidence_call_ids=("1",))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Tepid-H1 agent demo")
    parser.add_argument("--task", type=str, default="Compute 2+2")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=3)
    return run_agent(parser.parse_args())


def run_agent(args: argparse.Namespace) -> int:
    """Run a simple agent loop."""
    # Setup tools
    tools = ToolRegistry()

    def compute_tool(arguments: dict[str, Any]) -> str:
        return "The answer is 4"

    tools.register("compute", compute_tool)

    # Setup dependencies
    model = SimpleModel()
    context_builder = StateContextBuilder()
    policy = AllowlistPolicy(allowed_tools={"compute"})
    verifier = EvidenceVerifier()
    telemetry = ListTelemetry()

    deps = RuntimeDependencies(
        model=model,
        context_builder=context_builder,
        policy=policy,
        tools=tools,
        verifier=verifier,
        telemetry=telemetry,
    )

    # Run with retries
    runtime = AgentRuntime(deps, max_retries=args.max_retries)

    answer = runtime.run(args.task, max_steps=args.max_steps)
    print(f"Task: {args.task}")
    print(f"Answer: {answer.content}")
    print(f"Telemetry entries: {len(telemetry.events)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
