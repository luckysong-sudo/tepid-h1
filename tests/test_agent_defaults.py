import unittest

from tepid_h1.agent import (
    AgentRuntime,
    AllowlistPolicy,
    BudgetExceeded,
    EvidenceVerifier,
    FinalAnswer,
    ListTelemetry,
    RuntimeDependencies,
    StateContextBuilder,
    ToolCall,
    ToolRegistry,
    ToolResult,
)


class ScriptedModel:
    """Model that replays a fixed action script and then answers."""

    def __init__(self, script):
        self._script = list(script)
        self._index = 0

    def generate_action(self, state):
        if self._index < len(self._script):
            action = self._script[self._index]
            self._index += 1
            return action
        return FinalAnswer(content="done", evidence_call_ids=("call-1",))


class AgentDefaultsTests(unittest.TestCase):
    def test_allowlist_policy_permits_registered_tools(self):
        policy = AllowlistPolicy({"read_file": "read a file", "search": "search the web"})
        self.assertEqual(policy.allowed_tools, frozenset({"read_file", "search"}))
        allowed = policy.authorize(
            None,
            ToolCall(call_id="c1", tool_name="read_file", arguments={}),
        )
        self.assertTrue(allowed.allowed)
        denied = policy.authorize(
            None,
            ToolCall(call_id="c2", tool_name="delete_file", arguments={}),
        )
        self.assertFalse(denied.allowed)
        self.assertIn("delete_file", denied.reason)

    def test_empty_allowlist_denies_everything(self):
        policy = AllowlistPolicy()
        decision = policy.authorize(
            None,
            ToolCall(call_id="c1", tool_name="any", arguments={}),
        )
        self.assertFalse(decision.allowed)

    def test_tool_registry_executes_registered_handler(self):
        registry = ToolRegistry({"echo": lambda args: args.get("value")})
        result = registry.execute(
            ToolCall(call_id="c1", tool_name="echo", arguments={"value": 42}),
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.output, 42)
        self.assertEqual(result.call_id, "c1")

    def test_tool_registry_reports_unregistered_tool(self):
        registry = ToolRegistry()
        result = registry.execute(
            ToolCall(call_id="c1", tool_name="missing", arguments={}),
        )
        self.assertFalse(result.ok)
        self.assertIn("not registered", result.error)

    def test_tool_registry_captures_handler_exception(self):
        def bad_handler(args):
            raise ValueError("boom")

        registry = ToolRegistry({"bad": bad_handler})
        result = registry.execute(
            ToolCall(call_id="c1", tool_name="bad", arguments={}),
        )
        self.assertFalse(result.ok)
        self.assertIn("ValueError", result.error)
        self.assertIn("boom", result.error)

    def test_tool_registry_rejects_empty_name(self):
        registry = ToolRegistry()
        with self.assertRaisesRegex(ValueError, "non-empty"):
            registry.register("  ", lambda args: None)

    def test_evidence_verifier_requires_successful_results(self):
        from tepid_h1.agent.protocols import RuntimeState

        state = RuntimeState(task="t", max_steps=5)
        state.observations.append(ToolResult(call_id="call-1", ok=True, output="x"))
        state.observations.append(ToolResult(call_id="call-2", ok=False, error="err"))
        verifier = EvidenceVerifier()

        ok = verifier.verify_final(
            state,
            FinalAnswer(content="ok", evidence_call_ids=("call-1",)),
        )
        self.assertTrue(ok.allowed)

        missing = verifier.verify_final(
            state,
            FinalAnswer(content="bad", evidence_call_ids=("call-1", "call-2", "call-3")),
        )
        self.assertFalse(missing.allowed)
        self.assertIn("call-2", missing.reason)
        self.assertIn("call-3", missing.reason)

    def test_list_telemetry_records_events(self):
        from tepid_h1.agent.protocols import RuntimeState

        telemetry = ListTelemetry()
        state = RuntimeState(task="t", max_steps=5)
        state.steps = 1
        action = ToolCall(call_id="c1", tool_name="echo", arguments={"v": 1})
        result = ToolResult(call_id="c1", ok=True, output=1)
        telemetry.record(state, action, result)
        self.assertEqual(len(telemetry.events), 1)
        event = telemetry.events[0]
        self.assertEqual(event["step"], 1)
        self.assertEqual(event["action"]["kind"], "tool_call")
        self.assertEqual(event["action"]["tool_name"], "echo")
        self.assertEqual(event["result"]["kind"], "tool_result")
        self.assertTrue(event["result"]["ok"])

    def test_full_runtime_with_defaults(self):
        registry = ToolRegistry({"read_file": lambda args: f"contents of {args['path']}"})
        runtime = AgentRuntime(
            RuntimeDependencies(
                model=ScriptedModel(
                    [ToolCall(call_id="call-1", tool_name="read_file", arguments={"path": "x"})]
                ),
                context_builder=StateContextBuilder(),
                policy=AllowlistPolicy({"read_file": ""}),
                tools=registry,
                verifier=EvidenceVerifier(),
                telemetry=ListTelemetry(),
            )
        )
        answer = runtime.run("read x", max_steps=3)
        self.assertEqual(answer.content, "done")

    def test_runtime_rejects_invalid_retry_controls(self):
        dependencies = RuntimeDependencies(
            model=ScriptedModel([]),
            context_builder=StateContextBuilder(),
            policy=AllowlistPolicy(),
            tools=ToolRegistry(),
            verifier=EvidenceVerifier(),
        )

        with self.assertRaisesRegex(TypeError, "max_retries"):
            AgentRuntime(dependencies, max_retries=True)
        with self.assertRaisesRegex(TypeError, "retry_backoff"):
            AgentRuntime(dependencies, retry_backoff=True)
        with self.assertRaisesRegex(TypeError, "retry_backoff"):
            AgentRuntime(dependencies, retry_backoff="0.1")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "retry_backoff"):
            AgentRuntime(dependencies, retry_backoff=float("inf"))

    def test_runtime_rejects_invalid_step_budget_types(self):
        runtime = AgentRuntime(
            RuntimeDependencies(
                model=ScriptedModel([]),
                context_builder=StateContextBuilder(),
                policy=AllowlistPolicy(),
                tools=ToolRegistry(),
                verifier=EvidenceVerifier(),
            )
        )

        with self.assertRaisesRegex(TypeError, "max_steps"):
            runtime.run("read x", max_steps=True)
        with self.assertRaisesRegex(TypeError, "max_steps"):
            runtime.run("read x", max_steps=1.0)  # type: ignore[arg-type]

    def test_runtime_denies_unauthorized_tool(self):
        registry = ToolRegistry({"read_file": lambda args: "ok"})
        runtime = AgentRuntime(
            RuntimeDependencies(
                model=ScriptedModel(
                    [ToolCall(call_id="call-1", tool_name="delete_file", arguments={})]
                ),
                context_builder=StateContextBuilder(),
                policy=AllowlistPolicy({"read_file": ""}),
                tools=registry,
                verifier=EvidenceVerifier(),
            )
        )
        with self.assertRaises(BudgetExceeded):
            runtime.run("delete", max_steps=2)

    def test_runtime_exhausts_budget_without_answer(self):
        runtime = AgentRuntime(
            RuntimeDependencies(
                model=ScriptedModel(
                    [
                        ToolCall(call_id=f"c{i}", tool_name="read_file", arguments={})
                        for i in range(10)
                    ]
                ),
                context_builder=StateContextBuilder(),
                policy=AllowlistPolicy({"read_file": ""}),
                tools=ToolRegistry({"read_file": lambda args: "ok"}),
                verifier=EvidenceVerifier(),
            )
        )
        with self.assertRaises(BudgetExceeded):
            runtime.run("loop", max_steps=3)


if __name__ == "__main__":
    unittest.main()
