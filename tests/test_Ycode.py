import pytest
from Ycode import AgentStop, Agent, Thought, ToolCall

class FakeBrain:
    """fake brain for testing - returns predictable responses"""
    def __init__(self, responses=None):
        self.responses = responses or [Thought(text="Fake response")]
        self.call_count = 0
        self.last_conversation = None

    def think(self, conversation):
        self.last_conversation = list(conversation) # Store a copy
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return Thought(text="No more responses")


def test_quit_command_raises_agent_stop():
    """Verify /q raises AgentStop exception."""
    agent = Agent(brain=FakeBrain())
    with pytest.raises(AgentStop):
        agent.handle_input("/q")

def test_quit_command_with_whitespace():
    """Verify /q works with surrounding whitespace."""
    agent = Agent(brain=FakeBrain())
    with pytest.raises(AgentStop):
        agent.handle_input("  /q  ")

def test_empty_input_returns_empty_string():
    """Verify empty/whitespace input returns empty string."""
    agent = Agent(brain=FakeBrain())
    assert agent.handle_input("") == ""
    assert agent.handle_input("   ") == ""
    assert agent.handle_input("\n") == ""

def test_handle_input_returns_brain_response():
    """Verify handle_input returns the brain's response text."""
    brain = FakeBrain(responses=[Thought(text="Hello from brain")])
    agent = Agent(brain=brain)
    result = agent.handle_input("hi")
    assert result == "Hello from brain"

def test_conversation_accumulates():
    """Verify conversation list grows with each interaction."""
    brain = FakeBrain(responses=[
        Thought(text="Response 1"),
        Thought(text="Response 2")
    ])
    agent = Agent(brain=brain)

    agent.handle_input("First Message")
    assert len(agent.conversation) == 2 # user + assistant

    agent.handle_input("Second Message")
    assert len(agent.conversation) == 4 # 2 users + 2 asssistants

def test_conversation_contains_correct_roles():
    """Verify conversation has correct role alternation."""
    brain = FakeBrain(responses=[Thought(text="AI response")])
    agent = Agent(brain=brain)

    agent.handle_input("User message")

    assert agent.conversation[0]["role"] == "user"
    assert agent.conversation[0]["content"] == "User message"
    assert agent.conversation[1]["role"] == "assistant"
    assert agent.conversation[1]["content"] == "AI response"


def test_brain_receives_conversation():
    """Verify brain.think is called with the conversation list."""
    brain = FakeBrain()
    agent = Agent(brain=brain)

    agent.handle_input("Test message")

    assert brain.last_conversation is not None
    assert len(brain.last_conversation) == 1
    assert brain.last_conversation[0]["content"] == "Test message"    

def test_failed_brain_call_removes_user_message():
    """Verify failed brain call removes the user message from history."""
    class FailingBrain:
        def think(self, conversation):
            raise Exception("API Error")

    agent = Agent(brain=FailingBrain())
    result = agent.handle_input("Test message")

    assert "Error" in result
    assert len(agent.conversation) == 0  # Message should be removed