import os
import json
import time
import requests

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
