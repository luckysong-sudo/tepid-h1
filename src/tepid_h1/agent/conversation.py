"""Multi-turn conversation support for Tepid-H1 Agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .protocols import FinalAnswer, Role, ToolCall
from .runtime import AgentRuntime


@dataclass
class Message:
    """A single message in a conversation turn."""

    role: Role
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_response: Any = None
    timestamp: float = 0.0


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation."""

    user_message: Message
    assistant_message: Message | None = None
    observations: list[Any] = field(default_factory=list)
    final_answer: FinalAnswer | None = None
    step_count: int = 0
    completed: bool = False


@dataclass
class Conversation:
    """Multi-turn conversation with history tracking."""

    task: str
    turns: list[ConversationTurn] = field(default_factory=list)
    _current_turn_index: int = 0

    @property
    def current_turn(self) -> ConversationTurn | None:
        if self.turns:
            return self.turns[-1]
        return None

    @property
    def history(self) -> list[Message]:
        """Get all messages in conversation order."""
        messages: list[Message] = []
        for turn in self.turns:
            messages.append(turn.user_message)
            if turn.assistant_message:
                messages.append(turn.assistant_message)
            for obs in turn.observations:
                if isinstance(obs, dict) and "source" in obs:
                    messages.append(Message(
                        role=Role.OBSERVATION,
                        content=str(obs.get("content", obs)),
                    ))
        return messages

    def add_turn(
        self,
        user_message: Message,
        *,
        assistant_message: Message | None = None,
        observations: list[Any] | None = None,
        final_answer: FinalAnswer | None = None,
        step_count: int = 0,
        completed: bool = False,
    ) -> ConversationTurn:
        turn = ConversationTurn(
            user_message=user_message,
            assistant_message=assistant_message,
            observations=observations or [],
            final_answer=final_answer,
            step_count=step_count,
            completed=completed,
        )
        self.turns.append(turn)
        return turn

    def get_context_for_model(self) -> str:
        """Build context string for the model from conversation history."""
        lines: list[str] = [f"Task: {self.task}"]
        for turn in self.turns:
            lines.append(f"\n--- Turn {len(self.turns)} ---")
            lines.append(f"User: {turn.user_message.content}")
            if turn.assistant_message:
                if turn.assistant_message.tool_calls:
                    for tc in turn.assistant_message.tool_calls:
                        lines.append(f"Assistant called: {tc.arguments}")  # type: ignore[attr-defined]
                else:
                    lines.append(f"Assistant: {turn.assistant_message.content}")
            for obs in turn.observations:
                if isinstance(obs, dict):
                    lines.append(f"Observation: {obs.get('content', obs)}")
        return "\n".join(lines)


class ConversationAgent:
    """Agent with multi-turn conversation support."""

    def __init__(
        self,
        runtime: AgentRuntime,
        max_history_length: int = 10,
    ) -> None:
        self.runtime = runtime
        self.max_history_length = max_history_length
        self._conversations: dict[str, Conversation] = {}

    def create_conversation(self, task: str, conversation_id: str = "default") -> str:
        conversation = Conversation(task=task)
        self._conversations[conversation_id] = conversation
        return conversation_id

    def send_message(
        self,
        conversation_id: str,
        message: str,
        *,
        max_steps: int = 32,
    ) -> ConversationTurn:
        if conversation_id not in self._conversations:
            raise ValueError(f"Conversation '{conversation_id}' does not exist")
        conversation = self._conversations[conversation_id]
        user_msg = Message(role=Role.USER, content=message)
        conversation.add_turn(user_msg)

        try:
            result = self.runtime.run(
                f"{conversation.task}\n\nContext: {conversation.get_context_for_model()}",
                max_steps=max_steps,
            )
        except Exception:
            turn = conversation.current_turn
            if turn is not None:
                turn.completed = True
            raise

        assistant_msg = Message(
            role=Role.ASSISTANT,
            content=str(result) if not isinstance(result, FinalAnswer) else result.content,
        )
        conversation.add_turn(
            user_msg,
            assistant_message=assistant_msg,
            final_answer=result if isinstance(result, FinalAnswer) else None,
            completed=True,
        )
        turn = conversation.current_turn
        assert turn is not None
        return turn

    def list_conversations(self) -> list[str]:
        return list(self._conversations.keys())

    def get_history(self, conversation_id: str) -> list[Message]:
        if conversation_id not in self._conversations:
            raise ValueError(f"Conversation '{conversation_id}' does not exist")
        return self._conversations[conversation_id].history