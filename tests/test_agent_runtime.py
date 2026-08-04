import unittest

from tepid_h1.agent import (
    AgentRuntime,
    FinalAnswer,
    PolicyDecision,
    RetryExhausted,
    RuntimeDependencies,
    ToolCall,
    ToolResult,
)


class Context:
    def build(self, state):
        return state


class Model:
    def generate_action(self, state):
        if not state.observations:
            return ToolCall(call_id="call-1", tool_name="read_file", arguments={"path": "x"})
        return FinalAnswer(content="verified", evidence_call_ids=("call-1",))


class Policy:
    def authorize(self, state, call):
        return PolicyDecision(allowed=call.tool_name == "read_file")


class Tools:
    def execute(self, call):
        return ToolResult(call_id=call.call_id, ok=True, output="contents")


class Verifier:
    def verify_final(self, state, answer):
        observed = {item.call_id for item in state.observations if isinstance(item, ToolResult)}
        return PolicyDecision(allowed=set(answer.evidence_call_ids) <= observed)


class RetryingTools:
    def __init__(self):
        self.calls = 0

    def execute(self, call):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return ToolResult(call_id=call.call_id, ok=True, output="contents")


class FailingTools:
    def __init__(self):
        self.calls = 0

    def execute(self, call):
        self.calls += 1
        raise RuntimeError("persistent failure")


class AgentRuntimeTests(unittest.TestCase):
    def test_verified_tool_loop(self):
        runtime = AgentRuntime(
            RuntimeDependencies(
                model=Model(),
                context_builder=Context(),
                policy=Policy(),
                tools=Tools(),
                verifier=Verifier(),
            )
        )
        answer = runtime.run("read x", max_steps=3)
        self.assertEqual(answer.content, "verified")

    def test_retries_transient_tool_gateway_failure(self):
        tools = RetryingTools()
        runtime = AgentRuntime(
            RuntimeDependencies(
                model=Model(),
                context_builder=Context(),
                policy=Policy(),
                tools=tools,
                verifier=Verifier(),
            ),
            retry_backoff=0,
        )
        answer = runtime.run("read x", max_steps=3)
        self.assertEqual(answer.content, "verified")
        self.assertEqual(tools.calls, 2)

    def test_raises_after_retry_budget_is_exhausted(self):
        tools = FailingTools()
        runtime = AgentRuntime(
            RuntimeDependencies(
                model=Model(),
                context_builder=Context(),
                policy=Policy(),
                tools=tools,
                verifier=Verifier(),
            ),
            max_retries=1,
            retry_backoff=0,
        )
        with self.assertRaises(RetryExhausted):
            runtime.run("read x", max_steps=3)
        self.assertEqual(tools.calls, 2)


if __name__ == "__main__":
    unittest.main()
