"""Tests for agent policy and verifier extensions."""

import pytest

from tepid_h1.agent import (
    AllowlistPolicy,
    CompositePolicy,
    ContentLengthVerifier,
    EvidenceVerifier,
    FinalAnswer,
    ListTelemetry,
    RateLimitPolicy,
    RuntimeState,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolSchemaValidator,
)


class TestRateLimitPolicy:
    """Test the RateLimitPolicy class."""

    def test_allows_calls_under_limit(self):
        policy = RateLimitPolicy({"search": 3})
        state = RuntimeState(task="t", max_steps=10)
        call = ToolCall(call_id="c1", tool_name="search", arguments={})
        for _ in range(3):
            decision = policy.authorize(state, call)
            assert decision.allowed
        decision = policy.authorize(state, call)
        assert not decision.allowed
        assert "rate limit" in decision.reason

    def test_no_limit_means_unlimited(self):
        policy = RateLimitPolicy()
        state = RuntimeState(task="t", max_steps=10)
        call = ToolCall(call_id="c1", tool_name="search", arguments={})
        for _ in range(100):
            assert policy.authorize(state, call).allowed

    def test_wraps_inner_policy(self):
        inner = AllowlistPolicy({"search": ""})
        policy = RateLimitPolicy({"search": 2}, inner=inner)
        state = RuntimeState(task="t", max_steps=10)
        call = ToolCall(call_id="c1", tool_name="search", arguments={})
        assert policy.authorize(state, call).allowed
        assert policy.authorize(state, call).allowed
        assert not policy.authorize(state, call).allowed

    def test_inner_deny_short_circuits(self):
        inner = AllowlistPolicy({"read": ""})
        policy = RateLimitPolicy({"search": 2}, inner=inner)
        state = RuntimeState(task="t", max_steps=10)
        call = ToolCall(call_id="c1", tool_name="search", arguments={})
        decision = policy.authorize(state, call)
        assert not decision.allowed
        assert "not in the allowlist" in decision.reason

    def test_reset_clears_counts(self):
        policy = RateLimitPolicy({"search": 1})
        state = RuntimeState(task="t", max_steps=10)
        call = ToolCall(call_id="c1", tool_name="search", arguments={})
        assert policy.authorize(state, call).allowed
        assert not policy.authorize(state, call).allowed
        policy.reset()
        assert policy.authorize(state, call).allowed

    def test_rejects_negative_limit(self):
        with pytest.raises(ValueError, match="non-negative"):
            RateLimitPolicy({"search": -1})

    def test_rejects_boolean_limit(self):
        with pytest.raises(ValueError, match="non-negative"):
            RateLimitPolicy({"search": True})


class TestCompositePolicy:
    """Test the CompositePolicy class."""

    def test_all_pass_means_allowed(self):
        p1 = AllowlistPolicy({"search": ""})
        p2 = RateLimitPolicy({"search": 5})
        composite = CompositePolicy([p1, p2])
        state = RuntimeState(task="t", max_steps=10)
        call = ToolCall(call_id="c1", tool_name="search", arguments={})
        assert composite.authorize(state, call).allowed

    def test_first_deny_short_circuits(self):
        p1 = AllowlistPolicy({"read": ""})
        p2 = RateLimitPolicy({"search": 5})
        composite = CompositePolicy([p1, p2])
        state = RuntimeState(task="t", max_steps=10)
        call = ToolCall(call_id="c1", tool_name="search", arguments={})
        decision = composite.authorize(state, call)
        assert not decision.allowed
        assert "not in the allowlist" in decision.reason

    def test_second_deny_returns_its_reason(self):
        p1 = AllowlistPolicy({"search": ""})
        p2 = RateLimitPolicy({"search": 0})
        composite = CompositePolicy([p1, p2])
        state = RuntimeState(task="t", max_steps=10)
        call = ToolCall(call_id="c1", tool_name="search", arguments={})
        decision = composite.authorize(state, call)
        assert not decision.allowed
        assert "rate limit" in decision.reason

    def test_empty_policies_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            CompositePolicy([])


class TestToolSchemaValidator:
    """Test the ToolSchemaValidator class."""

    def test_validates_required_keys(self):
        validator = ToolSchemaValidator({"read_file": frozenset({"path"})})
        call = ToolCall(call_id="c1", tool_name="read_file", arguments={"path": "x"})
        assert validator.validate(call) is None

    def test_reports_missing_keys(self):
        validator = ToolSchemaValidator({"read_file": frozenset({"path", "mode"})})
        call = ToolCall(call_id="c1", tool_name="read_file", arguments={"path": "x"})
        result = validator.validate(call)
        assert result is not None
        assert not result.ok
        assert "mode" in result.error

    def test_unknown_tool_passes(self):
        validator = ToolSchemaValidator({"read_file": frozenset({"path"})})
        call = ToolCall(call_id="c1", tool_name="unknown", arguments={})
        assert validator.validate(call) is None

    def test_integration_with_tool_registry(self):
        validator = ToolSchemaValidator({"read_file": frozenset({"path"})})
        registry = ToolRegistry(
            {"read_file": lambda args: f"contents of {args['path']}"},
            schema_validator=validator,
        )
        ok_call = ToolCall(call_id="c1", tool_name="read_file", arguments={"path": "x"})
        result = registry.execute(ok_call)
        assert result.ok
        assert result.output == "contents of x"

        bad_call = ToolCall(call_id="c2", tool_name="read_file", arguments={})
        result = registry.execute(bad_call)
        assert not result.ok
        assert "path" in result.error


class TestContentLengthVerifier:
    """Test the ContentLengthVerifier class."""

    def test_accepts_valid_length(self):
        verifier = ContentLengthVerifier(min_length=1, max_length=100)
        state = RuntimeState(task="t", max_steps=5)
        answer = FinalAnswer(content="hello world")
        assert verifier.verify_final(state, answer).allowed

    def test_rejects_too_short(self):
        verifier = ContentLengthVerifier(min_length=10, max_length=100)
        state = RuntimeState(task="t", max_steps=5)
        answer = FinalAnswer(content="short")
        decision = verifier.verify_final(state, answer)
        assert not decision.allowed
        assert "below minimum" in decision.reason

    def test_rejects_too_long(self):
        verifier = ContentLengthVerifier(min_length=1, max_length=5)
        state = RuntimeState(task="t", max_steps=5)
        answer = FinalAnswer(content="this is way too long")
        decision = verifier.verify_final(state, answer)
        assert not decision.allowed
        assert "exceeds maximum" in decision.reason

    def test_wraps_inner_verifier(self):
        inner = EvidenceVerifier()
        verifier = ContentLengthVerifier(min_length=1, max_length=100, inner=inner)
        state = RuntimeState(task="t", max_steps=5)
        answer = FinalAnswer(content="ok", evidence_call_ids=("call-1",))
        decision = verifier.verify_final(state, answer)
        assert not decision.allowed
        assert "missing successful evidence" in decision.reason

    def test_rejects_invalid_min_length(self):
        with pytest.raises(ValueError, match="min_length"):
            ContentLengthVerifier(min_length=-1)

    def test_rejects_invalid_max_length(self):
        with pytest.raises(ValueError, match="max_length"):
            ContentLengthVerifier(min_length=10, max_length=5)

    def test_rejects_boolean_lengths(self):
        with pytest.raises(ValueError, match="min_length"):
            ContentLengthVerifier(min_length=True)


class TestListTelemetrySummary:
    """Test the ListTelemetry summary method."""

    def test_summary_reports_counts(self):
        telemetry = ListTelemetry()
        state = RuntimeState(task="t", max_steps=5)
        state.steps = 1
        action = ToolCall(call_id="c1", tool_name="echo", arguments={})
        result = ToolResult(call_id="c1", ok=True, output=1)
        telemetry.record(state, action, result)

        state.steps = 2
        action2 = ToolCall(call_id="c2", tool_name="fail", arguments={})
        result2 = ToolResult(call_id="c2", ok=False, error="boom")
        telemetry.record(state, action2, result2)

        state.steps = 3
        action3 = FinalAnswer(content="done")
        telemetry.record(state, action3, {"verification": "ok"})

        summary = telemetry.summary()
        assert summary["total_events"] == 3
        assert summary["tool_calls"] == 2
        assert summary["final_answers"] == 1
        assert summary["tool_failures"] == 1


