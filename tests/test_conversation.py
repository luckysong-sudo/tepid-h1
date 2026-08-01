"""Tests for conversation module."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tepid_h1.agent.conversation import Conversation, ConversationAgent, ConversationTurn, Message
from tepid_h1.agent.protocols import FinalAnswer, Role, ToolCall


class TestMessage:
    def test_basic_message(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert msg.tool_response is None
        assert msg.timestamp == 0.0

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(call_id="c1", tool_name="search", arguments={"q": "test"})
        msg = Message(role=Role.ASSISTANT, content="", tool_calls=[tc])
        assert msg.tool_calls[0].call_id == "c1"

    def test_message_with_response(self) -> None:
        msg = Message(role=Role.OBSERVATION, content="result", tool_response={"data": 1})
        assert msg.tool_response == {"data": 1}


class TestConversationTurn:
    def test_basic_turn(self) -> None:
        turn = ConversationTurn(user_message=Message(role=Role.USER, content="hi"))
        assert turn.assistant_message is None
        assert turn.observations == []
        assert turn.final_answer is None
        assert turn.step_count == 0
        assert turn.completed is False

    def test_completed_turn(self) -> None:
        ans = FinalAnswer(content="answer", evidence_call_ids=("c1",))
        turn = ConversationTurn(
            user_message=Message(role=Role.USER, content="hi"),
            assistant_message=Message(role=Role.ASSISTANT, content="answer"),
            final_answer=ans,
            step_count=3,
            completed=True,
        )
        assert turn.completed is True
        assert turn.final_answer is ans


class TestConversation:
    def test_empty_conversation(self) -> None:
        conv = Conversation(task="solve math")
        assert conv.task == "solve math"
        assert conv.turns == []
        assert conv.current_turn is None
        assert conv.history == []

    def test_add_turn(self) -> None:
        conv = Conversation(task="task")
        user_msg = Message(role=Role.USER, content="hello")
        turn = conv.add_turn(user_msg)
        assert len(conv.turns) == 1
        assert turn.user_message.content == "hello"

    def test_history_with_turns(self) -> None:
        conv = Conversation(task="task")
        conv.add_turn(Message(role=Role.USER, content="q1"))
        conv.add_turn(
            Message(role=Role.USER, content="q2"),
            assistant_message=Message(role=Role.ASSISTANT, content="a2"),
        )
        history = conv.history
        assert len(history) == 3
        assert history[0].content == "q1"
        assert history[2].content == "a2"

    def test_current_turn(self) -> None:
        conv = Conversation(task="task")
        assert conv.current_turn is None
        conv.add_turn(Message(role=Role.USER, content="q"))
        assert conv.current_turn is not None
        assert conv.current_turn.user_message.content == "q"

    def test_get_context_for_model(self) -> None:
        conv = Conversation(task="solve")
        conv.add_turn(Message(role=Role.USER, content="what is 2+2?"))
        ctx = conv.get_context_for_model()
        assert "Task: solve" in ctx
        assert "what is 2+2?" in ctx

    def test_get_context_with_assistant(self) -> None:
        conv = Conversation(task="task")
        conv.add_turn(
            Message(role=Role.USER, content="hi"),
            assistant_message=Message(role=Role.ASSISTANT, content="hello"),
        )
        ctx = conv.get_context_for_model()
        assert "Assistant: hello" in ctx


class TestConversationAgent:
    def test_create_conversation(self) -> None:
        runtime = MagicMock()
        runtime.run.return_value = FinalAnswer(content="done")
        agent = ConversationAgent(runtime)
        cid = agent.create_conversation("task")
        assert cid == "default"
        assert agent.list_conversations() == ["default"]

    def test_send_message(self) -> None:
        runtime = MagicMock()
        runtime.run.return_value = FinalAnswer(content="answer")
        agent = ConversationAgent(runtime)
        agent.create_conversation("task")
        turn = agent.send_message("default", "what is 2+2?")
        assert turn.completed is True
        assert turn.user_message.content == "what is 2+2?"
        assert turn.assistant_message is not None

    def test_send_message_invalid_conversation(self) -> None:
        runtime = MagicMock()
        agent = ConversationAgent(runtime)
        with pytest.raises(ValueError, match="does not exist"):
            agent.send_message("missing", "hi")

    def test_get_history(self) -> None:
        runtime = MagicMock()
        runtime.run.return_value = FinalAnswer(content="a")
        agent = ConversationAgent(runtime)
        agent.create_conversation("task")
        agent.send_message("default", "q")
        history = agent.get_history("default")
        assert len(history) >= 2

    def test_get_history_invalid(self) -> None:
        agent = ConversationAgent(MagicMock())
        with pytest.raises(ValueError, match="does not exist"):
            agent.get_history("missing")