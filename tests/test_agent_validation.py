"""Tests for agent protocol validation enhancements."""

import pytest

from tepid_h1.agent import (
    AgentRuntime,
    FinalAnswer,
    ModelValidationError,
    RuntimeDependencies,
    RuntimeState,
    ToolCall,
    ToolResult,
)
from tepid_h1.agent.protocols import PolicyDecision


class TestProtocolValidation:
    """Test protocol validation in data classes."""

    def test_toolcall_requires_non_empty_call_id(self):
        with pytest.raises(ValueError, match="call_id must not be empty"):
            ToolCall(call_id="", tool_name="search", arguments={})

    def test_toolcall_requires_non_empty_tool_name(self):
        with pytest.raises(ValueError, match="tool_name must not be empty"):
            ToolCall(call_id="abc", tool_name="", arguments={})

    def test_toolcall_strips_whitespace(self):
        call = ToolCall(call_id="  abc  ", tool_name="  search  ", arguments={})
        assert call.call_id == "abc"
        assert call.tool_name == "search"

    def test_final_answer_requires_non_empty_content(self):
        with pytest.raises(ValueError, match="content must not be empty"):
            FinalAnswer(content="")

    def test_final_answer_strips_whitespace(self):
        answer = FinalAnswer(content="  hello  ")
        assert answer.content == "hello"

    def test_runtime_state_requires_non_empty_task(self):
        with pytest.raises(ValueError, match="task must not be empty"):
            RuntimeState(task="", max_steps=10)

    def test_runtime_state_requires_positive_max_steps(self):
        with pytest.raises(ValueError, match="max_steps must be positive"):
            RuntimeState(task="test", max_steps=0)
        with pytest.raises(ValueError, match="max_steps must be positive"):
            RuntimeState(task="test", max_steps=-1)


class TestRuntimeStateHelpers:
    """Test RuntimeState helper methods."""

    def test_can_proceed_true(self):
        state = RuntimeState(task="test", max_steps=10)
        assert state.can_proceed() is True

    def test_can_proceed_false_when_steps_exhausted(self):
        state = RuntimeState(task="test", max_steps=2)
        state.steps = 2
        assert state.can_proceed() is False

    def test_record_action_increments_steps(self):
        state = RuntimeState(task="test", max_steps=10)
        action = ToolCall(call_id="c1", tool_name="search", arguments={})
        state.record_action(action)
        assert state.steps == 1
        assert len(state.actions) == 1

    def test_record_observation_appends(self):
        state = RuntimeState(task="test", max_steps=10)
        state.record_observation({"key": "value"})
        assert state.observations == [{"key": "value"}]


class TestAgentRuntimeErrors:
    """Test runtime error handling with new validation."""

    def test_model_exception_raises_model_validation_error(self):
        """Model exceptions should be wrapped as ModelValidationError."""
        class FailingModel:
            def generate_action(self, context):
                raise RuntimeError("model crashed")

        class StubContextBuilder:
            def build(self, state):
                return state

        class StubPolicy:
            def authorize(self, state, call):
                return PolicyDecision(allowed=True)

        class StubTools:
            def execute(self, call):
                return ToolResult(call_id=call.call_id, ok=True, output="result")

        class StubVerifier:
            def verify_final(self, state, answer):
                return PolicyDecision(allowed=True)

        runtime = AgentRuntime(RuntimeDependencies(
            model=FailingModel(),
            context_builder=StubContextBuilder(),
            policy=StubPolicy(),
            tools=StubTools(),
            verifier=StubVerifier(),
        ))

        with pytest.raises(ModelValidationError, match="model generated invalid action"):
            runtime.run("test task")

    def test_invalid_action_type_raises_model_validation_error(self):
        """Model returning invalid action type should raise ModelValidationError."""
        class BadModel:
            def generate_action(self, context):
                return "not a valid action"  # type: ignore[return-value]

        class StubContextBuilder:
            def build(self, state):
                return state

        class StubPolicy:
            def authorize(self, state, call):
                return PolicyDecision(allowed=True)

        class StubTools:
            def execute(self, call):
                return ToolResult(call_id=call.call_id, ok=True)

        class StubVerifier:
            def verify_final(self, state, answer):
                return PolicyDecision(allowed=True)

        runtime = AgentRuntime(RuntimeDependencies(
            model=BadModel(),
            context_builder=StubContextBuilder(),
            policy=StubPolicy(),
            tools=StubTools(),
            verifier=StubVerifier(),
        ))

        with pytest.raises(ModelValidationError):
            runtime.run("test task")

    def test_mismatched_call_id_raises_model_validation_error(self):
        """Tool returning wrong call_id should raise ModelValidationError."""
        step = 0

        class CounterModel:
            def generate_action(self, context):
                nonlocal step
                step += 1
                if step >= 2:
                    return FinalAnswer(content="done", evidence_call_ids=("c1",))
                return ToolCall(call_id="c1", tool_name="do", arguments={})

        class StubContextBuilder:
            def build(self, state):
                return state

        class StubPolicy:
            def authorize(self, state, call):
                return PolicyDecision(allowed=True)

        class StubTools:
            def execute(self, call):
                return ToolResult(call_id="wrong_id", ok=True)  # mismatched

        class StubVerifier:
            def verify_final(self, state, answer):
                return PolicyDecision(allowed=True)

        runtime = AgentRuntime(RuntimeDependencies(
            model=CounterModel(),
            context_builder=StubContextBuilder(),
            policy=StubPolicy(),
            tools=StubTools(),
            verifier=StubVerifier(),
        ))

        with pytest.raises(ModelValidationError, match="call_id does not match"):
            runtime.run("test task")
