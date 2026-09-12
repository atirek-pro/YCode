import os
import re
import sys
import json
import time
import html
import requests
import subprocess
import urllib.parse

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

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

# --- Memory Class ---
class Memory:
    """Persistent scratchpad for the agent."""
    def __init__(self, path=".Ycode-Memory/memory.md"):
        self.path = path
        self._ensure_exists()
        self.content = self._load()

    def _ensure_exists(self):
        """Create memory file with default content if needed."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            default = "I am Ycode, a helpful coding assistant.\n"
            with open(self.path, "w") as f:
                f.write(default)

    def _load(self):
        "Load Content from disk"
        with open(self.path, "r") as f:
            return f.read()

    def save(self, content):
        "Update memory content and persists to disk"
        self.content = content
        with open(self.path, "w") as f:
            f.write(content)

# --- Tool Context ---
class ToolContext:
    "What tools need to know about the agent's state"
    def __init__(self, memory=None):
        self.memory = memory # Memory object or None


# --- Brain Interface ---

class Brain:
    """Base class for LLM providers."""

    context_limit = 200_000
    last_input_tokens = 0

    def __init__(self, streaming=False):
        self.streaming = streaming

    def think(self, conversation):
        """Process conversation and return a Thought."""
        raise NotImplementedError

    def _parse_response(self, response):
        """Convert provider-specific API response into Thought."""
        raise NotImplementedError

# --- Gemini (The Brain) ---

class Gemini(Brain):
    """Gemini API - the brain of our agent."""

    context_limit = 1_048_576

    def __init__(self, memory=None, tools=None, streaming=False):
        super().__init__(streaming=streaming)

        self.memory = memory
        self.tools = tools or []
        self.system = None

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

        self.stream_url = (
            f"{self.base_url}/models/"
            f"{self.model}:streamGenerateContent?alt=sse"
        )
    # =====================================================
    # THINK
    # =====================================================

    def think(self, conversation):
        """Send conversation to Gemini and return a Thought."""

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

        # -------------------------------------------------
        # System instruction
        # -------------------------------------------------

        system_parts = []

        if self.system:
            system_parts.append({
                "text": self.system
            })

        # -------------------------------------------------
        # Persistent memory
        # -------------------------------------------------

        if self.memory and self.memory.content.strip():
            system_parts.append({
                "text": (
                    "You are Ycode, a helpful coding assistant.\n\n"
                    "Here is your persistent memory. "
                    "Use it to maintain context about the user "
                    "and previous interactions. "
                    "Do not mention or expose the memory system "
                    "unless relevant.\n\n"
                    "=== MEMORY ===\n"
                    f"{self.memory.content}\n"
                    "=== END MEMORY ==="
                )
            })

        if system_parts:
            payload["systemInstruction"] = {
                "parts": system_parts
            }

        # -------------------------------------------------
        # Tools
        # -------------------------------------------------

        gemini_tools = self._convert_tools()

        if gemini_tools:
            payload["tools"] = gemini_tools

        # -------------------------------------------------
        # Generate response
        # -------------------------------------------------

        if self.streaming:
            return self._think_streaming(
                headers=headers,
                payload=payload
            )

        return self._think_normal(
            headers=headers,
            payload=payload
        )

    # =====================================================
    # NORMAL GENERATION
    # =====================================================

    def _think_normal(self, headers, payload):
        """Generate a normal non-streaming Gemini response."""

        response = request_with_retry(
            self.url,
            headers=headers,
            payload=payload
        )

        data = response.json()

        # -------------------------------------------------
        # Token usage
        # -------------------------------------------------

        usage = data.get("usageMetadata", {})

        self.last_input_tokens = usage.get(
            "promptTokenCount",
            0
        )

        return self._parse_response(data)

    # =====================================================
    # STREAMING GENERATION
    # =====================================================

    def _think_streaming(self, headers, payload):
        """Generate and display a streamed Gemini response."""

        response = requests.post(
            self.stream_url,
            headers=headers,
            json=payload,
            timeout=120,
            stream=True
        )

        # -------------------------------------------------
        # Handle API errors
        # -------------------------------------------------

        if response.status_code >= 400:

            try:
                error_msg = response.json()["error"]["message"]

            except (KeyError, ValueError):
                error_msg = response.text

            raise Exception(
                f"API error ({response.status_code}): {error_msg}"
            )

        responses = []

        # -------------------------------------------------
        # Process SSE stream
        # -------------------------------------------------

        for line in response.iter_lines(
            chunk_size=1,
            decode_unicode=True
        ):

            if not line:
                continue

            # Gemini streaming responses are delivered
            # through Server-Sent Events.
            if line.startswith("data:"):
                line = line[5:].strip()

            if not line:
                continue

            try:
                chunk = json.loads(line)

            except json.JSONDecodeError:
                continue

            responses.append(chunk)

            # -------------------------------------------------
            # Token usage
            # -------------------------------------------------

            usage = chunk.get(
                "usageMetadata",
                {}
            )

            if usage:
                self.last_input_tokens = usage.get(
                    "promptTokenCount",
                    self.last_input_tokens
                )

            # -------------------------------------------------
            # Display streamed content
            # -------------------------------------------------

            self._display_stream_chunk(chunk)

        # Move to a new line after streaming finishes.
        print()

        # -------------------------------------------------
        # Reconstruct complete Gemini response
        # -------------------------------------------------

        return self._merge_stream_responses(
            responses
        )

    # =====================================================
    # STREAM DISPLAY
    # =====================================================

    def _display_stream_chunk(self, chunk):
        """Display text/thinking from a streamed Gemini chunk."""

        candidates = chunk.get(
            "candidates",
            []
        )

        if not candidates:
            return

        content = candidates[0].get(
            "content",
            {}
        )

        for part in content.get(
            "parts",
            []
        ):

            if "text" not in part:
                continue

            text = part["text"]

            # Thinking output
            if part.get("thought", False):

                print(
                    f"\033[2m{text}\033[0m",
                    end="",
                    flush=True
                )

            # Normal response
            else:

                print(
                    text,
                    end="",
                    flush=True
                )

    # =====================================================
    # MERGE STREAMED RESPONSES
    # =====================================================

    def _merge_stream_responses(self, responses):
        """Merge streamed Gemini chunks into one response."""

        merged_parts = []

        usage_metadata = {}

        finish_reason = None

        for response in responses:

            candidates = response.get(
                "candidates",
                []
            )

            if not candidates:
                continue

            candidate = candidates[0]

            content = candidate.get(
                "content",
                {}
            )

            merged_parts.extend(
                content.get("parts", [])
            )

            if candidate.get("finishReason"):
                finish_reason = candidate[
                    "finishReason"
                ]

            # Keep the latest usage metadata.
            if response.get("usageMetadata"):
                usage_metadata = response[
                    "usageMetadata"
                ]

        merged_candidate = {
            "content": {
                "parts": merged_parts
            }
        }

        if finish_reason:
            merged_candidate[
                "finishReason"
            ] = finish_reason

        merged_response = {
            "candidates": [
                merged_candidate
            ]
        }

        if usage_metadata:
            merged_response[
                "usageMetadata"
            ] = usage_metadata

        return self._parse_response(
            merged_response
        )

    # =====================================================
    # CONVERSATION CONVERSION
    # =====================================================

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

                    if item.get(
                        "type"
                    ) == "tool_result":

                        parts.append({
                            "functionResponse": {
                                "name": item["name"],
                                "response": {
                                    "result": item["content"]
                                },
                                "id": item.get(
                                    "tool_call_id"
                                )
                            }
                        })

                if parts:

                    contents.append({
                        "role": "user",
                        "parts": parts
                    })

        return contents

    # =====================================================
    # RESPONSE PARSING
    # =====================================================

    def _parse_response(self, response):
        """Convert Gemini's response format to Thought."""

        text_parts = []
        tool_calls = []
        thinking_parts = []

        candidates = response.get(
            "candidates",
            []
        )

        if not candidates:
            return Thought(
                text=None,
                tool_calls=[],
                raw_content=None,
                thinking=None
            )

        content = candidates[0].get(
            "content",
            {}
        )

        parts = content.get(
            "parts",
            []
        )

        for part in parts:

            # -------------------------------------------------
            # Text / thinking
            # -------------------------------------------------

            if "text" in part:

                if part.get(
                    "thought",
                    False
                ):
                    thinking_parts.append(
                        part["text"]
                    )

                else:
                    text_parts.append(
                        part["text"]
                    )

            # -------------------------------------------------
            # Function call
            # -------------------------------------------------

            elif "functionCall" in part:

                function_call = part[
                    "functionCall"
                ]

                tool_calls.append(
                    ToolCall(
                        id=function_call.get(
                            "id"
                        ),
                        name=function_call[
                            "name"
                        ],
                        args=function_call.get(
                            "args",
                            {}
                        )
                    )
                )

        return Thought(
            text=(
                "\n".join(text_parts)
                if text_parts
                else None
            ),

            tool_calls=tool_calls,

            # Preserve Gemini's original
            # Content object.
            raw_content=content,

            thinking=(
                "\n".join(thinking_parts)
                if thinking_parts
                else None
            )
        )

    # =====================================================
    # TOOL CONVERSION
    # =====================================================

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
    plan_safe = True
    description = "Reads a file from the filesystem. Use the to examine code."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The path to the file"}
        },
        "required": ["path"]
    }

    def execute(self, context, path):
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
    plan_safe = False
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

    def execute(self, context, path, content):
        print(f"--> Writting{path}")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
                return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writting file: {e}"

class WritePlan:
    "Save a plan to Plan.md"
    name = "write_plan"
    plan_safe = True
    description = "Saves a plan to PLAN.md. Use this to outline your approach before making changes."
    input_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The plan content in markdown"}
        },
        "required": ["content"]
    }

    def execute(self, context, content):
        print("--> Writing PLAN.md")
        try:
            with open("PLAN.md", "w", encoding="utf-8") as f:
                f.write(content)
            return "Plan saved to PLAN.md"
        except Exception as e:
            return f"Error saving Plan: {e}"

class EditFile:
    """Replaces text in a file (surgical edit)."""
    name = "edit_file"
    plan_safe = False
    description = "Replaces specific text in a file. Use for surgical edits instead of rewriting entire files."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "old_text": {"type": "string", "description": "Exact text to find and replace"},
            "new_text": {"type": "string", "description": "Text to replace it with"}
        },
        "required": ["path", "old_text", "new_text"]
    }    

    def execute(self, context, path, old_text, new_text):
        print(f"--> Editing {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if old_text not in content:
                return f"Error: Could not find the specified text in {path}"
            new_content = content.replace(old_text, new_text, 1)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing file: {e}"

class ListFiles:
    """Lists files in the project structure."""
    name = "list_files"
    plan_safe = True
    description = "Lists all files in the project structure. Useful to understand the project layout."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The root path (default '.')"}
        }
    }

    def execute(self, context, path="."):
        print(f"-> Listing {path}")
        try:
            file_list = []
            for root, dirs, files in os.walk(path):

                dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "venv", ".Ycode", ".venv", "myenv"}]

                level = root.replace(path, '').count(os.sep)
                indent = ' ' * 4 * (level)
                file_list.append(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 4 * (level + 1)

                for f in files:
                    file_list.append(f"{subindent}{f}")
            return "\n".join(file_list)
        except Exception as e:
            return f"Error Listing Files: {e}"

class SearchCodebase:
    "Searches for a string in all files"
    name = "search_codebase"
    plan_safe = True
    description = "Searches the entire codebase for a text string. Useful to find where functions or variables are defined."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The string to search for"},
            "path": {"type": "string", "description": "The root path (default '.')"}
        },
        "required": ["query"]
    }

    def execute(self, context, query, path="."):
        print(f"-> Searching for '{query}'")
        results = []
        try:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in {".get", "__pycache__", "venv", ".Ycode", ".venv", "myenv"}]
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for i, line in enumerate(f):
                                if query.lower() in line.lower():
                                    results.append(f"{file_path}:{i+1}: {line.strip()}")
                    except Exception:
                        continue
            return "\n".join(results) if results else "No Matches found."
        except Exception as e:
            return f"Error Searching: {e}"

class SearchWeb:
    """Searches the internet using DuckDuckGo."""
    name = "search_web"
    plan_safe = True
    description = "Searches the internet for current information. Use when you need knowledge beyond your training data."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }

    def execute(self, context, query):
        print(f"  → Searching web for '{query}'")
        if DDGS is None:
            return "Error: ddgs package not installed. Run: pip install ddgs"
        try:
            results = DDGS().text(query, max_results=3)
            if not results:
                return "No results found."

            formatted = []
            for r in results:
                formatted.append(f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}\n")

            return "\n".join(formatted)
        except Exception as e:
            return f"Error searching web: {e}"


class SaveMemory:
    """Updates the agent's internal memory/scratchpad."""
    name = "save_memory"
    plan_safe = True
    description = "Updates your internal memory/scratchpad. Use this to remember user preferences."
    input_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The full text to save."}
        },
        "required": ["content"]
    }

    def execute(self, context, content):
        print(f"--> Saving Memory")
        if context.memory is None:
            return "Error: Memory not available"
        context.memory.save(content)
        return "Memory updated successfully"

class RunCommand:
    """Executes shell commands."""
    name = "run_command"
    plan_safe = False
    description = "Executes a terminal command. Use this to run scripts, tests, or install packages."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run (e.g., 'python test.py')"}
        },
        "required": ["command"]
    }

    def execute(self, context, command):
        print(f"-> Running: {command[:50]}...")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=int(os.environ.get("YCODE_TIMEOUT", "30")),
                cwd=os.getcwd()
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            if not output:
                output = "(NO OUTPUT)"

            return output.strip() 

        except subprocess.TimeoutExpired:
            return "Error: Command Timed out."
        except Exception as e:
            return f"Error executing command: {e}"

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

tools = [ReadFile(), WritePlan(), SaveMemory(), ListFiles(), SearchCodebase(), SearchWeb(), WriteFile(), RunCommand(), EditFile()]

# --- Agent Class ---

class Agent:
    """A coding agent with conversation memory."""

    def __init__(self, brain, tools, memory=None, mode="plan", brain_name="gemini"):
        self.brain = brain
        self.tools = list(tools)
        self.memory = memory
        self.mode = mode
        self.brain_name = brain_name
        self.conversation = []
        self.brain.tools = self._tools_for_mode()
        self.brain.system = self._build_system_prompt()

    def _build_system_prompt(self):
        "Build system prompt from memory and current mode."
        parts = [self.memory.content] if self.memory else []

        if self.mode == "plan":
            parts.append(
                "You are in PLAN mode. You cannot write code files. "
                "Use write_plan to save your plans to PLAN.md."
            )

        return "\n".join(parts)

    def _tools_for_mode(self):
        "Return tool definations based on current mode."
        if self.mode == "act":
            return tool_definitions(self.tools)
        return tool_definitions([t for t in self.tools if t.plan_safe])

    def handle_input(self, user_input):
        """Handle user input. Returns output string, raises AgentStop to quit."""

        if user_input.strip() == "/q":
            raise AgentStop()

        if user_input.strip() == "/switch":
            return self._switch_brain()

        if not user_input.strip():
            return ""

        # handle mode switching
        if user_input.strip().startswith("/mode"):
            return self._handle_mode_command(user_input)

        self.conversation.append({
            "role": "user",
            "content": user_input
        })

        try:
            return self._agentic_loop()

        except Exception as e:
            return f"Error: {e}"

    def _handle_mode_command(self, user_input):
        "Handle /mode command to switch between plan and act"
        parts = user_input.strip().split()
        if len(parts) > 1 and parts[1] == "act":
            self.mode = "act"
            self.brain.tools = self._tools_for_mode()
            self.brain.system = self._build_system_prompt()
            return "⚠️  Switched to ACT MODE (Writing Enabled)"
        else:
            self.mode = "plan"
            self.brain.tools = self._tools_for_mode()
            self.brain.system = self._build_system_prompt()
            return "🛡️  Switched to PLAN MODE (Code Read-Only)"

    def _switch_brain(self):
        "Toggle to the next brain"
        names = list(BRAINS.keys())
        idx = names.index(self.brain_name)
        new_name = names[(idx + 1)%len(names)]

        try:
            self.brain = BRAINS[new_name](memory=self.memory, tools=tool_definitions(self.tools))
            self.brain_name = new_name
            return f"Switched to: {new_name}"
        except ValueError as e:
            return f"Cannot switch to {new_name}: {e}"

    def _agentic_loop(self):
        """Process brain responses, executing tools until done."""

        output_parts = []
        max_iterations = 50

        for _iteration in range(max_iterations):
            # Ask brain
            thought = self.brain.think(self.conversation)

            # ---------------------------------------------
            # Display thinking
            # ---------------------------------------------

            if thought.thinking and not self.brain.streaming:

                lines = thought.thinking.strip().split("\n")[:5]

                for i, line in enumerate(lines):

                    prefix = "  💭 " if i == 0 else "     "

                    print(
                        f"\033[2m{prefix}{line}\033[0m"
                    )
            
            # ---------------------------------------------
            # Compact if approaching context limit
            # ---------------------------------------------

            if self.brain.last_input_tokens > self.brain.context_limit * 0.75:
                self._compact_conversation()

            # Store raw content for message history
            self.conversation.append({"role": "assistant", "content": thought.raw_content})

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

        else:
            output_parts.append("(STOPPED: too many iterations)")

        return "\n".join(output_parts)

    def _compact_conversation(self):
        """Summarize old messages to stay within context limits."""

        print("(Compacting Conversation...)")

        history = "\n".join(
            f"{m['role']}: {str(m['content'])[:500]}"
            for m in self.conversation
        )

        prompt = [{
            "role": "user",
            "content": (
                "Summarize this conversation for continuity. "
                "Focus on what was accomplished, what's in progress, "
                "and key decisions:\n\n"
                f"{history}"
            )
        }]

        saved_tools = self.brain.tools
        self.brain.tools = []

        try:
            thought = self.brain.think(prompt)
        finally:
            self.brain.tools = saved_tools

        self.conversation = [{
            "role": "user",
            "content": (
                "Previous conversation summary:\n\n"
                f"{thought.text}"
            )
        }]

    def _execute_tool(self, name, args):
        """Execute a tool by name with given arguments."""
        tool = get_tool(self.tools, name)
        if tool is None:
            return f"Error: Tool '{name}' not found"
        try:
            context = ToolContext(memory=self.memory)
            return tool.execute(context=context, **args)
        except TypeError as e:
            return f"Error: Invalid arguments - {e}"

# --- Main Loop ---

def main():
    # parse mode from CLI
    mode = "act" if len(sys.argv) > 1 and sys.argv[1] == "--act" else "plan"
    brain_name = os.getenv("YCode_BRAIN", "gemini")
    memory = Memory()
    brain = BRAINS[brain_name](memory=memory, tools=tool_definitions(tools), streaming=True)
    agent = Agent(brain=brain, tools=tools, memory=memory, brain_name=brain_name, mode=mode)
    print("⚡ Nanocode v0.6")
    print(f"Commands: /q quit, /switch toggle brain, mode[plan | act]")
    print(f"Brain: {brain_name}\n")
    if mode == "act":
        print("MODE: ACT (WRITING ENABLED)")
    else:
        print("Mode: PLAN (Code Read-Only)")

    while True:
        try:
            user_input = input(f"[{agent.brain_name}]>>")
            output = agent.handle_input(user_input)
            if output and not agent.brain.streaming:
                print(f"\n{output}\n")

        except (AgentStop, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()