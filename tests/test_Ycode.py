import pytest
from Ycode import AgentStop, Agent

def test_handle_input_returns_strings():
    "Verify handle input returns a string for normal input"
    agent = Agent()
    result = agent.handle_input("hello")
    assert isinstance(result, str)
    assert "hello" in result

def test_empty_input_returns_empty_string():
    "Verify empty/whitespace input returns empty string"
    agent = Agent()
    assert agent.handle_input("") == ""
    assert agent.handle_input(" ") == ""
    assert agent.handle_input("\n") == ""

def test_quite_command_raises_agent_stop():
    "Verify /q raises AgentStop exception"
    agent = Agent()
    with pytest.raises(AgentStop):
        agent.handle_input("/q")

def test_quit_command_with_whitespace():
    "Verify /q works with surrounding whitespace"
    agent = Agent()
    with pytest.raises(AgentStop):
        agent.handle_input(" /q ")