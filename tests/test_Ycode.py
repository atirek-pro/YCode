import os
import tempfile
from unittest.mock import Mock, patch

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
    SaveMemory,
    Gemini,
    Memory,
    ToolContext,
    tool_definitions,
    tools,
    get_tool,
    ListFiles,
    RunCommand,
    EditFile,
    SearchCodebase,
    SearchWeb,
)


# ============================================================
# Fake Brain
# ============================================================

class FakeBrain(Brain):
    """Fake brain for testing predictable agent behavior."""

    def __init__(self, responses=None, memory=None, tools=None):
        self.responses = responses or [
            Thought(text="Fake response")
        ]
        self.call_count = 0
        self.last_conversation = None
        self.memory = memory
        self.tools = tools or []
        self.last_input_tokens = 0

    def think(self, conversation):
        self.last_conversation = list(conversation)

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response

        return Thought(text="No more responses")

    def _parse_response(self, response):
        raise NotImplementedError


# ============================================================
# Helpers
# ============================================================

def make_memory():
    """Create an isolated temporary Memory instance."""
    tmpdir = tempfile.TemporaryDirectory()
    path = os.path.join(tmpdir.name, "memory.md")
    memory = Memory(path=path)
    return tmpdir, memory


def make_agent(brain=None, memory=None, brain_name="gemini"):
    """Create an Agent using the current Ycode constructor."""
    if memory is None:
        _, memory = make_memory()

    if brain is None:
        brain = FakeBrain(memory=memory, tools=tool_definitions(tools))

    return Agent(
        brain=brain,
        tools=tools,
        memory=memory,
        brain_name=brain_name,
    )


# ============================================================
# Basic Agent Tests
# ============================================================

def test_quit_command_raises_agent_stop():
    """Verify /q raises AgentStop exception."""

    _, memory = make_memory()

    try:
        agent = Agent(
            brain=FakeBrain(),
            tools=tools,
            memory=memory,
        )

        with pytest.raises(AgentStop):
            agent.handle_input("/q")

    finally:
        memory_path = memory.path
        memory_dir = os.path.dirname(memory_path)

        # TemporaryDirectory owns cleanup.
        # Nothing else is required here.


def test_quit_command_with_whitespace():
    """Verify /q works with surrounding whitespace."""

    _, memory = make_memory()

    agent = Agent(
        brain=FakeBrain(),
        tools=tools,
        memory=memory,
    )

    with pytest.raises(AgentStop):
        agent.handle_input("  /q  ")


def test_empty_input_returns_empty_string():
    """Verify empty/whitespace input returns empty string."""

    _, memory = make_memory()

    agent = Agent(
        brain=FakeBrain(),
        tools=tools,
        memory=memory,
    )

    assert agent.handle_input("") == ""
    assert agent.handle_input("   ") == ""
    assert agent.handle_input("\n") == ""


def test_handle_input_returns_brain_response():
    """Verify handle_input returns the brain's response text."""

    _, memory = make_memory()

    brain = FakeBrain(
        responses=[
            Thought(text="Hello from brain")
        ]
    )

    agent = Agent(
        brain=brain,
        tools=tools,
        memory=memory,
    )

    result = agent.handle_input("hi")

    assert result == "Hello from brain"


def test_conversation_accumulates():
    """Verify conversation list grows with each user interaction."""

    _, memory = make_memory()

    brain = FakeBrain(
        responses=[
            Thought(text="Response 1"),
            Thought(text="Response 2"),
        ]
    )

    agent = Agent(
        brain=brain,
        tools=tools,
        memory=memory,
    )

    agent.handle_input("First Message")

    assert len(agent.conversation) == 2

    agent.handle_input("Second Message")

    assert len(agent.conversation) == 4


# ============================================================
# Brain Tests
# ============================================================

def test_agent_stores_brain_name():
    """Verify agent stores the brain name."""

    _, memory = make_memory()

    agent = Agent(
        brain=FakeBrain(),
        tools=tools,
        memory=memory,
        brain_name="claude",
    )

    assert agent.brain_name == "claude"

    agent = Agent(
        brain=FakeBrain(),
        tools=tools,
        memory=memory,
        brain_name="deepseek",
    )

    assert agent.brain_name == "deepseek"


def test_brains_registry_has_expected_providers():
    """Verify BRAINS registry contains Gemini."""

    assert "gemini" in BRAINS


def test_switch_command_reinitializes_brain():
    """
    Verify /switch reinitializes the selected brain.

    Currently BRAINS only contains Gemini, so /switch
    switches from Gemini back to Gemini.
    """

    _, memory = make_memory()

    agent = Agent(
        brain=FakeBrain(memory=memory),
        tools=tools,
        memory=memory,
        brain_name="gemini",
    )

    original_brains = BRAINS.copy()

    try:
        BRAINS["gemini"] = FakeBrain

        result = agent.handle_input("/switch")

        assert result == "Switched to: gemini"
        assert agent.brain_name == "gemini"
        assert isinstance(agent.brain, FakeBrain)

        # Most importantly, the same Memory instance is preserved.
        assert agent.brain.memory is memory

    finally:
        BRAINS.clear()
        BRAINS.update(original_brains)


# ============================================================
# Conversation Tests
# ============================================================

def test_conversation_contains_correct_roles():
    """Verify user input is stored correctly."""

    _, memory = make_memory()

    brain = FakeBrain(
        responses=[
            Thought(text="AI response")
        ]
    )

    agent = Agent(
        brain=brain,
        tools=tools,
        memory=memory,
    )

    agent.handle_input("User message")

    assert agent.conversation[0]["role"] == "user"
    assert agent.conversation[0]["content"] == "User message"


def test_brain_receives_conversation():
    """Verify brain.think receives the conversation list."""

    _, memory = make_memory()

    brain = FakeBrain()

    agent = Agent(
        brain=brain,
        tools=tools,
        memory=memory,
    )

    agent.handle_input("Test message")

    assert brain.last_conversation is not None
    assert len(brain.last_conversation) == 1
    assert brain.last_conversation[0]["role"] == "user"
    assert brain.last_conversation[0]["content"] == "Test message"


def test_failed_brain_call_preserves_user_message():
    """
    Verify failed brain calls return an error and the
    user message remains in conversation history.
    """

    _, memory = make_memory()

    class FailingBrain(Brain):

        def __init__(self):
            self.memory = memory

        def think(self, conversation):
            raise Exception("API Error")

        def _parse_response(self, response):
            raise NotImplementedError

    agent = Agent(
        brain=FailingBrain(),
        tools=tools,
        memory=memory,
    )

    result = agent.handle_input("Test message")

    assert "Error" in result
    assert len(agent.conversation) == 1
    assert agent.conversation[0]["role"] == "user"
    assert agent.conversation[0]["content"] == "Test message"


# ============================================================
# Tool Definition Tests
# ============================================================

def test_tool_has_required_attributes():
    """Verify tool classes have required attributes."""

    tool = ReadFile()

    assert tool.name == "read_file"
    assert tool.description is not None
    assert tool.input_schema is not None


def test_write_file_has_valid_input_schema():
    """Verify WriteFile has a valid JSON-schema-style definition."""

    tool = WriteFile()

    assert tool.input_schema["type"] == "object"
    assert "path" in tool.input_schema["properties"]
    assert "content" in tool.input_schema["properties"]

    assert tool.input_schema["required"] == [
        "path",
        "content",
    ]


def test_save_memory_has_valid_input_schema():
    """Verify SaveMemory has the expected schema."""

    tool = SaveMemory()

    assert tool.name == "save_memory"
    assert tool.input_schema["type"] == "object"
    assert "content" in tool.input_schema["properties"]
    assert tool.input_schema["required"] == ["content"]


def test_tool_definitions_are_provider_neutral():
    """Verify tool_definitions returns the expected internal format."""

    definitions = tool_definitions(tools)

    assert len(definitions) == len(tools)

    names = {tool["name"] for tool in definitions}

    assert "read_file" in names
    assert "write_file" in names
    assert "save_memory" in names
    assert "search_web" in names

    for definition in definitions:
        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition


# ============================================================
# Gemini Tool Conversion Tests
# ============================================================

def test_gemini_converts_tools_to_function_declarations():
    """
    Verify provider-neutral tools are converted into
    Gemini's functionDeclarations format.
    """

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        brain = Gemini(
            tools=tool_definitions(tools)
        )

    converted_tools = brain._convert_tools()

    assert converted_tools is not None
    assert len(converted_tools) == 1

    function_declarations = converted_tools[0][
        "functionDeclarations"
    ]

    assert len(function_declarations) == len(tools)

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

    save_memory = next(
        tool
        for tool in function_declarations
        if tool["name"] == "save_memory"
    )

    search_web = next(
        tool
        for tool in function_declarations
        if tool["name"] == "search_web"
    )

    assert read_file["description"] == ReadFile.description
    assert read_file["parameters"] == ReadFile.input_schema

    assert write_file["description"] == WriteFile.description
    assert write_file["parameters"] == WriteFile.input_schema

    assert save_memory["description"] == SaveMemory.description
    assert save_memory["parameters"] == SaveMemory.input_schema

    assert search_web["description"] == SearchWeb.description
    assert search_web["parameters"] == SearchWeb.input_schema


# ============================================================
# Tool Class Tests
# ============================================================

def test_read_file_adds_line_numbers():
    """Verify ReadFile prefixes each line with line numbers."""

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as f:

        f.write(
            "line one\n"
            "line two\n"
            "line three\n"
        )

        temp_path = f.name

    try:
        tool = ReadFile()

        context = ToolContext()

        result = tool.execute(
            context,
            temp_path,
        )

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
            "test.txt",
        )

        tool = WriteFile()

        context = ToolContext()

        result = tool.execute(
            context,
            path,
            "hello world",
        )

        assert os.path.exists(path)
        assert "Successfully wrote" in result
        assert "11 characters" in result

        with open(
            path,
            encoding="utf-8",
        ) as f:

            assert f.read() == "hello world"


# ============================================================
# Memory Class Tests
# ============================================================

def test_memory_creates_default_file():
    """Verify Memory creates file with default content."""

    with tempfile.TemporaryDirectory() as tmpdir:

        path = os.path.join(
            tmpdir,
            "memory.md",
        )

        memory = Memory(path=path)

        assert os.path.exists(path)
        assert "I am Ycode" in memory.content


def test_memory_save_updates_content_and_file():
    """Verify Memory.save updates memory and persists to disk."""

    with tempfile.TemporaryDirectory() as tmpdir:

        path = os.path.join(
            tmpdir,
            "memory.md",
        )

        memory = Memory(path=path)

        memory.save("New content")

        assert memory.content == "New content"

        with open(path, encoding="utf-8") as f:
            assert f.read() == "New content"


def test_memory_reloads_persisted_content():
    """Verify saved memory survives creating a new Memory instance."""

    with tempfile.TemporaryDirectory() as tmpdir:

        path = os.path.join(
            tmpdir,
            "memory.md",
        )

        memory = Memory(path=path)

        memory.save("Persistent memory")

        new_memory = Memory(path=path)

        assert new_memory.content == "Persistent memory"


# ============================================================
# ToolContext Tests
# ============================================================

def test_tool_context_contains_memory():
    """Verify ToolContext stores the Memory instance."""

    _, memory = make_memory()

    context = ToolContext(memory=memory)

    assert context.memory is memory


# ============================================================
# SaveMemory Tool Tests
# ============================================================

def test_save_memory_updates_memory():
    """Verify SaveMemory updates the Memory object."""

    with tempfile.TemporaryDirectory() as tmpdir:

        memory = Memory(
            path=os.path.join(
                tmpdir,
                "memory.md",
            )
        )

        tool = SaveMemory()
        context = ToolContext(memory=memory)

        result = tool.execute(
            context,
            "Updated preferences",
        )

        assert "successfully" in result.lower()
        assert memory.content == "Updated preferences"

        with open(memory.path, encoding="utf-8") as f:
            assert f.read() == "Updated preferences"


def test_save_memory_without_memory_returns_error():
    """Verify SaveMemory handles missing memory context."""

    tool = SaveMemory()
    context = ToolContext(memory=None)

    result = tool.execute(
        context,
        "Some memory",
    )

    assert result == "Error: Memory not available"


# ============================================================
# Agent Tool Execution Tests
# ============================================================

def test_agent_executes_save_memory_with_context():
    """
    Verify Agent._execute_tool passes ToolContext correctly.

    This specifically protects against the bug where context was
    created but not passed to tool.execute().
    """

    with tempfile.TemporaryDirectory() as tmpdir:

        memory = Memory(
            path=os.path.join(
                tmpdir,
                "memory.md",
            )
        )

        agent = Agent(
            brain=FakeBrain(),
            tools=tools,
            memory=memory,
        )

        result = agent._execute_tool(
            "save_memory",
            {
                "content": "User prefers Python"
            },
        )

        assert result == "Memory updated successfully"
        assert memory.content == "User prefers Python"

        with open(memory.path, encoding="utf-8") as f:
            assert f.read() == "User prefers Python"


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
        delete=False,
        encoding="utf-8",
    ) as f:

        f.write("test content\n")
        temp_path = f.name

    try:

        _, memory = make_memory()

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
                            },
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
                                    "id": "1",
                                },
                            },
                        ],
                    },
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
                        ],
                    },
                ),
            ]
        )

        agent = Agent(
            brain=brain,
            tools=tools,
            memory=memory,
        )

        result = agent.handle_input(
            "Read the file"
        )

        assert "Let me read that file." in result
        assert "The file contains test content." in result

        assert brain.call_count == 2

        assert len(agent.conversation) == 4

        # Original user message
        assert agent.conversation[0] == {
            "role": "user",
            "content": "Read the file",
        }

        # Gemini model response
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

        # Provider-neutral tool result
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
# Memory Agentic Loop Test
# ============================================================

def test_agentic_loop_executes_save_memory():
    """
    Verify the full memory flow:

    Gemini requests save_memory
        ↓
    Agent executes tool
        ↓
    ToolContext provides Memory
        ↓
    Memory.save() writes to disk
        ↓
    Agent continues the loop
    """

    with tempfile.TemporaryDirectory() as tmpdir:

        memory_path = os.path.join(
            tmpdir,
            "memory.md",
        )

        memory = Memory(path=memory_path)

        brain = FakeBrain(
            responses=[
                Thought(
                    text="I'll remember that.",
                    tool_calls=[
                        ToolCall(
                            id="memory-1",
                            name="save_memory",
                            args={
                                "content": (
                                    "User prefers Python "
                                    "for programming."
                                )
                            },
                        )
                    ],
                    raw_content={
                        "role": "model",
                        "parts": [
                            {
                                "text": "I'll remember that."
                            },
                            {
                                "functionCall": {
                                    "name": "save_memory",
                                    "args": {
                                        "content": (
                                            "User prefers Python "
                                            "for programming."
                                        )
                                    },
                                    "id": "memory-1",
                                },
                            },
                        ],
                    },
                ),

                Thought(
                    text="I've saved that preference.",
                    tool_calls=[],
                ),
            ]
        )

        agent = Agent(
            brain=brain,
            tools=tools,
            memory=memory,
        )

        result = agent.handle_input(
            "Remember that I prefer Python."
        )

        assert "I'll remember that." in result
        assert "I've saved that preference." in result

        assert brain.call_count == 2

        assert (
            memory.content
            == "User prefers Python for programming."
        )

        with open(memory_path, encoding="utf-8") as f:
            assert (
                f.read()
                == "User prefers Python for programming."
            )


# ============================================================
# Gemini Conversation Conversion Tests
# ============================================================

def test_gemini_converts_tool_result_to_function_response():
    """
    Verify provider-neutral tool results are converted
    into Gemini functionResponse format.
    """

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        brain = Gemini(
            tools=tool_definitions(tools)
        )

    conversation = [
        {
            "role": "user",
            "content": "Read the file",
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
                            "id": "call-123",
                        }
                    }
                ],
            },
        },

        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_call_id": "call-123",
                    "name": "read_file",
                    "content": "1 | hello world",
                }
            ],
        },
    ]

    converted = brain._convert_conversation(
        conversation
    )

    assert converted[0]["role"] == "user"

    assert (
        converted[0]["parts"][0]["text"]
        == "Read the file"
    )

    assert converted[1]["role"] == "model"

    function_call = (
        converted[1]["parts"][0]["functionCall"]
    )

    assert function_call["name"] == "read_file"
    assert function_call["id"] == "call-123"

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
    """Verify Gemini functionCall becomes ToolCall."""

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
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
                                "id": "call-123",
                            }
                        }
                    ],
                }
            }
        ]
    }

    thought = brain._parse_response(response)

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

    assert (
        thought.raw_content
        == response["candidates"][0]["content"]
    )


def test_gemini_parses_text_response():
    """Verify Gemini text response becomes Thought."""

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
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
                    ],
                }
            }
        ]
    }

    thought = brain._parse_response(response)

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

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
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
                            "thought": True,
                        },
                        {
                            "text": (
                                "I'll read the file now."
                            )
                        },
                    ],
                }
            }
        ]
    }

    thought = brain._parse_response(response)

    assert (
        thought.thinking
        == "I need to inspect the file."
    )

    assert (
        thought.text
        == "I'll read the file now."
    )

    assert thought.tool_calls == []


# ============================================================
# Gemini Memory Payload Tests
# ============================================================

def test_gemini_adds_memory_to_system_instruction():
    """
    Verify persistent memory is included in Gemini's
    systemInstruction payload.
    """

    with tempfile.TemporaryDirectory() as tmpdir:

        memory = Memory(
            path=os.path.join(
                tmpdir,
                "memory.md",
            )
        )

        memory.save(
            "User prefers Python.\n"
            "User uses Windows."
        )

        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key"},
        ):
            brain = Gemini(
                memory=memory,
                tools=tool_definitions(tools),
            )

            mock_response = Mock()
            mock_response.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": "Hello"
                                }
                            ],
                        }
                    }
                ]
            }

            with patch(
                "Ycode.request_with_retry",
                return_value=mock_response,
            ) as mock_request:

                brain.think(
                    [
                        {
                            "role": "user",
                            "content": "Hello",
                        }
                    ]
                )

    payload = mock_request.call_args.kwargs["payload"]

    assert "systemInstruction" in payload

    system_instruction = payload[
        "systemInstruction"
    ]

    assert "parts" in system_instruction
    assert len(system_instruction["parts"]) == 1

    system_text = (
        system_instruction["parts"][0]["text"]
    )

    assert "=== MEMORY ===" in system_text
    assert "=== END MEMORY ===" in system_text

    assert "User prefers Python." in system_text
    assert "User uses Windows." in system_text


def test_gemini_without_memory_does_not_add_system_instruction():
    """Verify Gemini does not add memory when none is provided."""

    with patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "test-key"},
    ):
        brain = Gemini(
            memory=None,
            tools=tool_definitions(tools),
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "text": "Hello"
                            }
                        ],
                    }
                }
            ]
        }

        with patch(
            "Ycode.request_with_retry",
            return_value=mock_response,
        ) as mock_request:

            brain.think(
                [
                    {
                        "role": "user",
                        "content": "Hello",
                    }
                ]
            )

    payload = mock_request.call_args.kwargs["payload"]

    assert "systemInstruction" not in payload

# --- Mode tests ---

def test_agent_defaults_to_plan_mode():
    """Verify agent starts in plan mode by default."""
    agent = Agent(brain=FakeBrain(), tools=tools)
    assert agent.mode == "plan"

def test_plan_mode_hides_write_file():
    """Verify plan mode does not expose write_file to the brain."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")
    tool_names = [t["name"] for t in agent.brain.tools]

    assert "write_file" not in tool_names
    assert "write_plan" in tool_names
    assert "read_file" in tool_names

def test_act_mode_shows_all_tools():
    """Verify act mode exposes all tools to the brain."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="act")
    tool_names = [t["name"] for t in agent.brain.tools]

    assert "write_file" in tool_names
    assert "write_plan" in tool_names
    assert "read_file" in tool_names

def test_mode_command_switches_to_plan():
    """Verify /mode plan switches to plan mode."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="act")
    result = agent.handle_input("/mode plan")

    assert agent.mode == "plan"
    assert "PLAN" in result

def test_mode_command_switches_to_act():
    """Verify /mode act switches to act mode."""
    agent = Agent(brain=FakeBrain(), tools=tools, mode="plan")
    result = agent.handle_input("/mode act")

    assert agent.mode == "act"
    assert "ACT" in result

# --- New tests for Chapter 8: Awareness tools ---

def test_list_files_returns_file_tree():
    """Verify ListFiles returns a tree structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "README.md"), 'w') as f:
            f.write("# Test")
        with open(os.path.join(tmpdir, "src", "main.py"), 'w') as f:
            f.write("print('hello')")

        tool = ListFiles()
        context = ToolContext()
        result = tool.execute(context, path=tmpdir)

        assert "README.md" in result
        assert "src/" in result
        assert "main.py" in result

def test_list_files_skips_git_and_pycache():
    """Verify ListFiles skips .git and __pycache__ directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directories that should be skipped
        os.makedirs(os.path.join(tmpdir, ".git"))
        os.makedirs(os.path.join(tmpdir, "__pycache__"))
        os.makedirs(os.path.join(tmpdir, "src"))

        with open(os.path.join(tmpdir, ".git", "config"), 'w') as f:
            f.write("git config")
        with open(os.path.join(tmpdir, "__pycache__", "cache.pyc"), 'w') as f:
            f.write("cache")
        with open(os.path.join(tmpdir, "src", "main.py"), 'w') as f:
            f.write("print('hello')")

        tool = ListFiles()
        context = ToolContext()
        result = tool.execute(context, path=tmpdir)

        assert "config" not in result
        assert "cache.pyc" not in result
        assert "main.py" in result

def test_search_codebase_finds_matches():
    """Verify SearchCodebase finds text in files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.py"), 'w') as f:
            f.write("def hello_world():\n    print('hello')\n")

        tool = SearchCodebase()
        context = ToolContext()
        result = tool.execute(context, query="hello_world", path=tmpdir)

        assert "test.py" in result
        assert "hello_world" in result
        assert ":1:" in result  # Line number

def test_search_codebase_case_insensitive():
    """Verify SearchCodebase is case-insensitive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.py"), 'w') as f:
            f.write("def HelloWorld():\n    pass\n")

        tool = SearchCodebase()
        context = ToolContext()
        result = tool.execute(context, query="helloworld", path=tmpdir)
        assert "HelloWorld" in result

# --- New tests for Chapter 9: RunCommand tool ---

def test_run_command_executes():
    """Verify run_command executes a command."""
    tool = RunCommand()
    context = ToolContext()
    result = tool.execute(context, command="echo hello")

    assert "STDOUT" in result
    assert "hello" in result

def test_run_command_captures_stderr():
    """Verify run_command captures error output."""
    tool = RunCommand()
    context = ToolContext()
    result = tool.execute(context, command="python -c \"import sys; sys.stderr.write('error!')\"")

    assert "STDERR" in result
    assert "error!" in result

def test_run_coomand_timeout(monkeypatch):
    "Verify run_command times out on long-running commands"
    monkeypatch.setenv("YCODE_TIMEOUT", "1")
    tool = RunCommand()
    context = ToolContext()
    result = tool.execute(context, command='python -c "import time; time.sleep(10)"')

    assert "timed out" in result.lower()

# --- EditFile Tests ---

def test_edit_file_replaces_text():
    """Verify EditFile replaces text in a file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("x = 1\ny = 2\nz = 3\n")
        temp_path = f.name

    try:
        tool = EditFile()
        context = ToolContext()
        result = tool.execute(context, temp_path, "y = 2", "y = 42")

        assert "Successfully" in result
        with open(temp_path) as f:
            content = f.read()
        assert "y = 42" in content
        assert "y = 2" not in content
    finally:
        os.unlink(temp_path)

# --- Chapter 11: SearchWeb Tool Tests ---

def test_search_web_tool_exists():
    """Verify SearchWeb class exists with required attributes."""
    tool = SearchWeb()
    assert tool.name == "search_web"
    assert tool.description is not None
    assert tool.input_schema is not None
    assert "query" in tool.input_schema["properties"]


def test_search_web_in_tools_list():
    """Verify SearchWeb is registered in the tools list."""
    tool_names = [t.name for t in tools]
    assert "search_web" in tool_names


def test_search_web_can_be_found():
    """Verify get_tool can find search_web."""
    tool = get_tool(tools, "search_web")
    assert tool is not None
    assert tool.name == "search_web"


def test_search_web_in_tool_definitions():
    """Verify search_web appears in tool definitions for API."""
    definitions = tool_definitions(tools)
    names = [d["name"] for d in definitions]
    assert "search_web" in names


def test_search_web_execute_success(monkeypatch):
    """Verify SearchWeb.execute() returns formatted results."""
    fake_results = [
        {"title": "Python 3.13", "href": "https://python.org", "body": "Latest release"},
    ]
    monkeypatch.setattr("nanocode.DDGS", lambda: type("FakeDDGS", (), {"text": lambda self, q, max_results=3: fake_results})())
    tool = SearchWeb()
    context = ToolContext()
    result = tool.execute(context, "latest python version")
    assert "Python 3.13" in result
    assert "https://python.org" in result


def test_search_web_execute_no_results(monkeypatch):
    """Verify SearchWeb.execute() handles empty results."""
    monkeypatch.setattr("nanocode.DDGS", lambda: type("FakeDDGS", (), {"text": lambda self, q, max_results=3: []})())
    tool = SearchWeb()
    context = ToolContext()
    result = tool.execute(context, "impossible query xyz")
    assert "No results found" in result


def test_search_web_execute_error(monkeypatch):
    """Verify SearchWeb.execute() handles errors gracefully."""
    def raise_error():
        raise RuntimeError("Network down")
    monkeypatch.setattr("nanocode.DDGS", lambda: type("FakeDDGS", (), {"text": lambda self, q, max_results=3: raise_error()})())
    tool = SearchWeb()
    context = ToolContext()
    result = tool.execute(context, "test query")
    assert "Error" in result