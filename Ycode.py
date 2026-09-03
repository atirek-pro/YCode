import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# --- HTTP Helpers ---

def request_with_retry(url, headers, payload, max_retries=10):
    """Make HTTP POST with retry on rate limit (429), server errors (5xx), and network failures."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.exceptions.RequestException as e:
            wait_time = 2 ** attempt
            print(f"Network error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            retry_after = response.headers.get("retry-after")
            try:
                wait_time = int(retry_after) if retry_after else 2 ** attempt
            except (ValueError, TypeError):
                wait_time = 2 ** attempt
            print(f"Error {response.status_code}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue

        if response.status_code >= 400:
            try:
                error_msg = response.json()["error"]["message"]
            except (KeyError, ValueError):
                error_msg = response.text
            raise Exception(f"API error ({response.status_code}): {error_msg}")
        return response
    raise Exception(f"Request Failed after {max_retries} retries")

# --- Exceptions ---

class AgentStop(Exception):
    """Raised when the agent should stop processing."""
    pass


# --- Brain Response Types ---

class ToolCall:
    """A tool invocation request from the brain."""

    def __init__(self, id, name, args):
        self.id = id
        self.name = name
        self.args = args  # dict


class Thought:
    """Standardized response from any Brain."""

    def __init__(self, text=None, tool_calls=None, raw_content=None, thinking=None):
        self.text = text  # str or None
        self.tool_calls = tool_calls or []  # list of ToolCall
        self.raw_content = raw_content # original API response from message history
        self.thinking = thinking  # str or None

# --- Brain Interface ---

class Brain:
    """Base class for LLM providers."""

    def think(self, conversation):
        """Process conversation and return a Thought."""
        raise NotImplementedError

    def _parse_response(self, response):
        """Convert provider-specific API response into Thought."""
        raise NotImplementedError

# --- Gemini (The Brain) ---

class Gemini(Brain):
    """Gemini API - the brain of our agent."""

    def __init__(self, tools=None):
        self.tools = tools or []
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.7-flash"
        )

        self.base_url = (
            "https://generativelanguage.googleapis.com/v1beta"
        )

        self.url = (
            f"{self.base_url}/models/"
            f"{self.model}:generateContent"
        )

    def think(self, conversation):
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "contents": self._convert_conversation(conversation),
            "generationConfig": {
                "maxOutputTokens": 16000,
                "thinkingConfig": {
                    "thinkingLevel": "medium"
                }
            }
        }

        gemini_tools = self._convert_tools()

        if gemini_tools:
            payload["tools"] = gemini_tools

        response = request_with_retry(
            self.url,
            headers=headers,
            payload=payload
        )

        return self._parse_response(response.json())

    def _convert_conversation(self, conversation):
        """
        Convert our internal conversation format
        into Gemini's contents format.
        """

        contents = []

        for message in conversation:

            role = message["role"]
            content = message["content"]

            # -------------------------------------------------
            # Normal user / assistant text
            # -------------------------------------------------

            if isinstance(content, str):

                if role == "assistant":
                    role = "model"

                contents.append({
                    "role": role,
                    "parts": [
                        {
                            "text": content
                        }
                    ]
                })

            # -------------------------------------------------
            # Gemini model response containing function calls
            # -------------------------------------------------

            elif isinstance(content, dict):

                contents.append(content)

            # -------------------------------------------------
            # Tool results
            # -------------------------------------------------

            elif isinstance(content, list):

                parts = []

                for item in content:

                    if item.get("type") == "tool_result":

                        parts.append({
                            "functionResponse": {
                                "name": item["name"],
                                "response": {
                                    "result": item["content"]
                                },
                                "id": item.get("tool_call_id")
                            }
                        })

                if parts:

                    contents.append({
                        "role": "user",
                        "parts": parts
                    })

        return contents

    def _parse_response(self, response):
        """Convert Gemini's response format to Thought."""

        text_parts = []
        tool_calls = []
        thinking_parts = []

        candidates = response.get("candidates", [])

        if not candidates:
            return Thought(
                text=None,
                tool_calls=[],
                raw_content=None,
                thinking=None
            )

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        for part in parts:

            # ---------------------------------------------
            # Text / thinking
            # ---------------------------------------------

            if "text" in part:

                if part.get("thought", False):
                    thinking_parts.append(part["text"])
                else:
                    text_parts.append(part["text"])

            # ---------------------------------------------
            # Function call
            # ---------------------------------------------

            elif "functionCall" in part:

                function_call = part["functionCall"]

                tool_calls.append(
                    ToolCall(
                        id=function_call.get("id"),
                        name=function_call["name"],
                        args=function_call.get("args", {})
                    )
                )

        return Thought(
            text="\n".join(text_parts)
            if text_parts else None,

            tool_calls=tool_calls,

            # Preserve Gemini's original Content object
            raw_content=content,

            thinking="\n".join(thinking_parts)
            if thinking_parts else None
        )

    def _convert_tools(self):
        """
        Convert our provider-neutral tool definitions
        into Gemini's function declaration format.
        """
        if not self.tools:
            return None

        return [
            {
                "functionDeclarations": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                    for tool in self.tools
                ]
            }
        ]


# Available brains
BRAINS = {
    "gemini": Gemini
}

# Tool Classes
class ReadFile:
    "Reads a file from the filesystem"
    name = "read_file"
    description = "Reads a file from the filesystem. Use the to examine code."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The path to the file"}
        },
        "required": ["path"]
    }

    def execute(self, path):
        print(f"--> Reading {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                numbered_lines = [f"{i+1} | {line}" for i, line in enumerate(lines)]
                return "".join(numbered_lines)
        except Exception as e:
            return f"Error Reading File: {e}"

class WriteFile:
    "Write content to a file"
    name = "write_file"
    description = "Writes content to a file OVERWRITES existing content"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file"
            },
            "content": {
                "type": "string",
                "description": "The full content to write"
            }
        },
        "required": ["path", "content"]
    }

    def execute(self, path, content):
        print(f"--> Writting{path}")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
                return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writting file: {e}"

# Tool helpers
def get_tool(tools, name):
    "Find a tool by name, or None if not found"
    return next((t for t in tools if t.name == name), None)

def tool_definitions(tools):
    """Return provider-neutral tool definitions."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema
        }
        for t in tools
    ]

tools = [ReadFile(), WriteFile()]

# --- Agent Class ---

class Agent:
    """A coding agent with conversation memory."""

    def __init__(self, brain, tools, brain_name="gemini"):
        self.brain = brain
        self.tools = list(tools)
        self.brain_name = brain_name
        self.conversation = []

    def handle_input(self, user_input):
        """Handle user input. Returns output string, raises AgentStop to quit."""

        if user_input.strip() == "/q":
            raise AgentStop()

        if user_input.strip() == "/switch":
            return self._switch_brain()

        if not user_input.strip():
            return ""

        self.conversation.append({
            "role": "user",
            "content": user_input
        })

        try:
            return self._agentic_loop()

        except Exception as e:
            return f"Error: {e}"

    def _switch_brain(self):
        "Toggle to the next brain"
        names = list(BRAINS.keys())
        idx = names.index(self.brain_name)
        new_name = names[(idx + 1)%len(names)]

        try:
            self.brain = BRAINS[new_name](tools=tool_definitions(self.tools))
            self.brain_name = new_name
            return f"Switched to: {new_name}"
        except ValueError as e:
            return f"Cannot switch to {new_name}: {e}"

    def _agentic_loop(self):
        """Process brain responses, executing tools until done."""

        output_parts = []

        while True:

            thought = self.brain.think(self.conversation)

            # ---------------------------------------------
            # Display thinking
            # ---------------------------------------------

            if thought.thinking:

                lines = thought.thinking.strip().split("\n")[:5]

                for i, line in enumerate(lines):

                    prefix = "  💭 " if i == 0 else "     "

                    print(
                        f"\033[2m{prefix}{line}\033[0m"
                    )

            # ---------------------------------------------
            # Collect text output
            # ---------------------------------------------

            if thought.text:
                output_parts.append(thought.text)

            # ---------------------------------------------
            # No tool calls → agent is finished
            # ---------------------------------------------

            if not thought.tool_calls:
                break

            # ---------------------------------------------
            # Preserve Gemini model response
            # ---------------------------------------------

            self.conversation.append({
                "role": "assistant",
                "content": thought.raw_content
            })

            # ---------------------------------------------
            # Execute tools
            # ---------------------------------------------

            tool_results = []

            for tool_call in thought.tool_calls:

                result = self._execute_tool(
                    tool_call.name,
                    tool_call.args
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": result
                })

            # ---------------------------------------------
            # Add tool results to conversation
            # ---------------------------------------------

            self.conversation.append({
                "role": "user",
                "content": tool_results
            })

        return "\n".join(output_parts)

    def _execute_tool(self, name, args):
        """Execute a tool by name with given arguments."""
        tool = get_tool(self.tools, name)
        if tool is None:
            return f"Error: Tool '{name}' not found"
        try:
            return tool.execute(**args)
        except TypeError as e:
            return f"Error: Invalid arguments - {e}"

# --- Main Loop ---

def main():
    brain_name = os.getenv("YCode_BRAIN", "gemini")
    brain = BRAINS[brain_name](tools=tool_definitions(tools))
    agent = Agent(brain=brain, tools=tools, brain_name=brain_name)
    print("⚡ Nanocode v0.3")
    print(f"Commands: /q quit, /switch toggle brain")
    print(f"Brain: {brain_name}\n")

    while True:
        try:
            user_input = input(f"[{agent.brain_name}]>>")
            output = agent.handle_input(user_input)
            if output:
                print(f"\n{output}\n")

        except (AgentStop, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()