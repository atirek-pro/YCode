import os
import tempfile

import pytest

from Ycode import (
    AgentStop,
    Agent,
    Thought,
    ToolCall,
    Brain,
    BRAINS,
    ReadFile,
    WriteFile,
    Gemini,
    tool_definitions,
    tools,
)


# ============================================================
# Fake Brain
# ============================================================

class FakeBrain(Brain):
    """Fake brain for testing predictable agent behavior."""

    def __init__(self, responses=None):
        self.responses = responses or [
            Thought(text="Fake response")
        ]
        self.call_count = 0
        self.last_conversation = None

    def think(self, conversation):
        self.last_conversation = list(conversation)

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response

        return Thought(text="No more responses")

    def _parse_response(self, response):
        """Fake implementation required by Brain interface."""
        raise NotImplementedError


# ============================================================
# Basic Agent Tests
# ============================================================

def test_quit_command_raises_agent_stop():
    """Verify /q raises AgentStop exception."""
    agent = Agent(
        brain=FakeBrain(),
        tools=tools
    )

    with pytest.raises(AgentStop):
        agent.handle_input("/q")


def test_quit_command_with_whitespace():
    """Verify /q works with surrounding whitespace."""
    agent = Agent(
        brain=FakeBrain(),
        tools=tools
    )

    with pytest.raises(AgentStop):
        agent.handle_input("  /q  ")


def test_empty_input_returns_empty_string():
    """Verify empty/whitespace input returns empty string."""
    agent = Agent(
        brain=FakeBrain(),
        tools=tools
    )

    assert agent.handle_input("") == ""
    assert agent.handle_input("   ") == ""
    assert agent.handle_input("\n") == ""


def test_handle_input_returns_brain_response():
    """Verify handle_input returns the brain's response text."""

    brain = FakeBrain(
        responses=[
            Thought(text="Hello from brain")
        ]
    )

    agent = Agent(
        brain=brain,
        tools=tools
    )

    result = agent.handle_input("hi")

    assert result == "Hello from brain"


def test_conversation_accumulates():
    """Verify conversation list grows with each interaction."""

    brain = FakeBrain(
        responses=[
            Thought(text="Response 1"),
            Thought(text="Response 2"),
        ]
    )

    agent = Agent(
        brain=brain,
        tools=tools
    )

    agent.handle_input("First Message")

    # Only the user message is stored because the
    # updated agentic loop does not append normal
    # assistant text to conversation.
    assert len(agent.conversation) == 1

    agent.handle_input("Second Message")

    assert len(agent.conversation) == 2


# ============================================================
# Multiple Brain Tests
# ============================================================

def test_agent_stores_brain_name():
    """Verify agent stores the brain name."""

    agent = Agent(
        brain=FakeBrain(),
        tools=tools,
        brain_name="claude"
    )

    assert agent.brain_name == "claude"

    agent = Agent(
        brain=FakeBrain(),
        tools=tools,
        brain_name="deepseek"
    )

    assert agent.brain_name == "deepseek"


def test_brains_registry_has_expected_providers():
    """Verify BRAINS registry contains expected providers."""

    assert "gemini" in BRAINS

# ============================================================
# Conversation Tests
# ============================================================

def test_conversation_contains_correct_roles():
    """
    Verify user input is stored correctly.

    Normal assistant responses are no longer stored as
    plain assistant messages because the agentic loop only
    preserves model responses when tool calls occur.
    """

    brain = FakeBrain(
        responses=[
            Thought(text="AI response")
        ]
    )

    agent = Agent(
        brain=brain,
        tools=tools
    )

    agent.handle_input("User message")

    assert agent.conversation[0]["role"] == "user"
    assert agent.conversation[0]["content"] == "User message"


def test_brain_receives_conversation():
    """Verify brain.think receives the conversation list."""

    brain = FakeBrain()

    agent = Agent(
        brain=brain,
        tools=tools
    )

    agent.handle_input("Test message")

    assert brain.last_conversation is not None
    assert len(brain.last_conversation) == 1
    assert brain.last_conversation[0]["role"] == "user"
    assert brain.last_conversation[0]["content"] == "Test message"


def test_failed_brain_call_preserves_user_message():
    """
    Verify failed brain calls return an error.

    The updated handle_input() no longer removes the user
    message automatically, so the failed request remains
    in conversation history.
    """

    class FailingBrain(Brain):

        def think(self, conversation):
            raise Exception("API Error")

        def _parse_response(self, response):
            raise NotImplementedError

    agent = Agent(
        brain=FailingBrain(),
        tools=tools
    )

    result = agent.handle_input("Test message")

    assert "Error" in result

    # Updated behavior:
    # user message remains in conversation.
    assert len(agent.conversation) == 1
    assert agent.conversation[0]["role"] == "user"
    assert agent.conversation[0]["content"] == "Test message"


# ============================================================
# Tool Definition Tests
# ============================================================

def test_tool_has_required_attributes():
    """Verify tool classes have name, description, and input_schema."""

    tool = ReadFile()

    assert tool.name == "read_file"
    assert tool.description is not None
    assert tool.input_schema is not None


def test_write_file_has_valid_input_schema():
    """Verify WriteFile has a valid JSON-schema-style input definition."""

    tool = WriteFile()

    assert tool.input_schema["type"] == "object"
    assert "path" in tool.input_schema["properties"]
    assert "content" in tool.input_schema["properties"]

    # required must be a list, not a Python set
    assert tool.input_schema["required"] == [
        "path",
        "content"
    ]


# ============================================================
# Gemini Tool Conversion Tests
# ============================================================

def test_gemini_converts_tools_to_function_declarations():
    """
    Verify provider-neutral tools are converted into
    Gemini's functionDeclarations format.
    """

    brain = Gemini(
        tools=tool_definitions(tools)
    )

    converted_tools = brain._convert_tools()

    assert converted_tools is not None
    assert len(converted_tools) == 1

    function_declarations = converted_tools[0][
        "functionDeclarations"
    ]

    assert len(function_declarations) == 2

    read_file = next(
        tool
        for tool in function_declarations
        if tool["name"] == "read_file"
    )

    write_file = next(
        tool
        for tool in function_declarations
        if tool["name"] == "write_file"
    )

    assert read_file["description"] == ReadFile.description
    assert read_file["parameters"] == ReadFile.input_schema

    assert write_file["description"] == WriteFile.description
    assert write_file["parameters"] == WriteFile.input_schema


# ============================================================
# Tool Class Tests
# ============================================================

def test_read_file_adds_line_numbers():
    """Verify ReadFile prefixes each line with line numbers."""

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False
    ) as f:

        f.write(
            "line one\n"
            "line two\n"
            "line three\n"
        )

        temp_path = f.name

    try:

        tool = ReadFile()

        result = tool.execute(temp_path)

        assert "1 | line one" in result
        assert "2 | line two" in result
        assert "3 | line three" in result

    finally:
        os.unlink(temp_path)


def test_write_file_creates_file():
    """Verify WriteFile creates a file with content."""

    with tempfile.TemporaryDirectory() as tmpdir:

        path = os.path.join(
            tmpdir,
            "test.txt"
        )

        tool = WriteFile()

        result = tool.execute(
            path,
            "hello world"
        )

        assert os.path.exists(path)
        assert "Successfully wrote" in result
        assert "11 characters" in result

        with open(
            path,
            encoding="utf-8"
        ) as f:

            assert f.read() == "hello world"


# ============================================================
# Agentic Loop Tests
# ============================================================

def test_agentic_loop_executes_tool_calls():
    """
    Verify the agentic loop:

    1. Receives a tool call from the brain.
    2. Executes the requested tool.
    3. Preserves the model response.
    4. Adds the tool result.
    5. Calls the brain again.
    6. Returns the final response.
    """

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False
    ) as f:

        f.write("test content\n")
        temp_path = f.name

    try:

        brain = FakeBrain(
            responses=[
                Thought(
                    text="Let me read that file.",

                    tool_calls=[
                        ToolCall(
                            id="1",
                            name="read_file",
                            args={
                                "path": temp_path
                            }
                        )
                    ],

                    raw_content={
                        "role": "model",
                        "parts": [
                            {
                                "text": "Let me read that file."
                            },
                            {
                                "functionCall": {
                                    "name": "read_file",
                                    "args": {
                                        "path": temp_path
                                    },
                                    "id": "1"
                                }
                            }
                        ]
                    }
                ),

                Thought(
                    text="The file contains test content.",
                    tool_calls=[],

                    raw_content={
                        "role": "model",
                        "parts": [
                            {
                                "text": (
                                    "The file contains "
                                    "test content."
                                )
                            }
                        ]
                    }
                )
            ]
        )

        agent = Agent(
            brain=brain,
            tools=tools
        )

        result = agent.handle_input(
            "Read the file"
        )

        # Both model responses should be returned.
        assert "Let me read that file." in result
        assert "The file contains test content." in result

        # Brain called twice:
        # 1. Initial request → tool call
        # 2. Tool result → final response
        assert brain.call_count == 2

        # --------------------------------------------------------
        # Verify conversation structure
        # --------------------------------------------------------

        assert len(agent.conversation) == 3

        # 1. Original user message
        assert agent.conversation[0] == {
            "role": "user",
            "content": "Read the file"
        }

        # 2. Gemini model response
        model_message = agent.conversation[1]

        assert model_message["role"] == "assistant"

        assert (
            model_message["content"]["role"]
            == "model"
        )

        function_call = (
            model_message["content"]["parts"][1][
                "functionCall"
            ]
        )

        assert function_call["name"] == "read_file"

        assert function_call["id"] == "1"

        # 3. Provider-neutral tool result
        tool_result_message = agent.conversation[2]

        assert tool_result_message["role"] == "user"

        tool_results = (
            tool_result_message["content"]
        )

        assert len(tool_results) == 1

        assert tool_results[0]["type"] == "tool_result"

        assert (
            tool_results[0]["tool_call_id"]
            == "1"
        )

        assert (
            tool_results[0]["name"]
            == "read_file"
        )

        assert (
            "test content"
            in tool_results[0]["content"]
        )

    finally:
        os.unlink(temp_path)


# ============================================================
# Gemini Conversation Conversion Tests
# ============================================================

def test_gemini_converts_tool_result_to_function_response():
    """
    Verify provider-neutral tool results are converted
    into Gemini's functionResponse format.
    """

    brain = Gemini(
        tools=tool_definitions(tools)
    )

    conversation = [
        {
            "role": "user",
            "content": "Read the file"
        },

        {
            "role": "assistant",
            "content": {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "path": "test.txt"
                            },
                            "id": "call-123"
                        }
                    }
                ]
            }
        },

        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_call_id": "call-123",
                    "name": "read_file",
                    "content": "1 | hello world"
                }
            ]
        }
    ]

    converted = (
        brain._convert_conversation(
            conversation
        )
    )

    # User message
    assert converted[0]["role"] == "user"

    assert (
        converted[0]["parts"][0]["text"]
        == "Read the file"
    )

    # Original model function call
    assert converted[1]["role"] == "model"

    function_call = (
        converted[1]["parts"][0]["functionCall"]
    )

    assert function_call["name"] == "read_file"
    assert function_call["id"] == "call-123"

    # Gemini function response
    assert converted[2]["role"] == "user"

    function_response = (
        converted[2]["parts"][0]["functionResponse"]
    )

    assert function_response["name"] == "read_file"
    assert function_response["id"] == "call-123"

    assert (
        function_response["response"]["result"]
        == "1 | hello world"
    )


# ============================================================
# Gemini Response Parsing Tests
# ============================================================

def test_gemini_parses_function_call():
    """Verify Gemini functionCall is converted into ToolCall."""

    brain = Gemini(
        tools=tool_definitions(tools)
    )

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {
                                    "path": "test.txt"
                                },
                                "id": "call-123"
                            }
                        }
                    ]
                }
            }
        ]
    }

    thought = brain._parse_response(
        response
    )

    assert thought.text is None

    assert len(
        thought.tool_calls
    ) == 1

    tool_call = thought.tool_calls[0]

    assert tool_call.id == "call-123"
    assert tool_call.name == "read_file"

    assert tool_call.args == {
        "path": "test.txt"
    }

    # Original Gemini content should be preserved.
    assert (
        thought.raw_content
        == response["candidates"][0]["content"]
    )


def test_gemini_parses_text_response():
    """Verify Gemini text response is converted into Thought."""

    brain = Gemini(
        tools=tool_definitions(tools)
    )

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "The file contains "
                                "hello world."
                            )
                        }
                    ]
                }
            }
        ]
    }

    thought = brain._parse_response(
        response
    )

    assert (
        thought.text
        == "The file contains hello world."
    )

    assert thought.tool_calls == []

    assert (
        thought.raw_content
        == response["candidates"][0]["content"]
    )


def test_gemini_parses_thinking_response():
    """Verify Gemini thought parts are separated from normal text."""

    brain = Gemini(
        tools=tool_definitions(tools)
    )

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "I need to inspect "
                                "the file."
                            ),
                            "thought": True
                        },
                        {
                            "text": (
                                "I'll read the file now."
                            )
                        }
                    ]
                }
            }
        ]
    }

    thought = brain._parse_response(
        response
    )

    assert (
        thought.thinking
        == "I need to inspect the file."
    )

    assert (
        thought.text
        == "I'll read the file now."
    )

    assert thought.tool_calls == []