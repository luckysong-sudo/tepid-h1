import unittest

from tepid_h1.agent import (
    AgentRuntime,
    FinalAnswer,
    PolicyDecision,
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


if __name__ == "__main__":
    unittest.main()

