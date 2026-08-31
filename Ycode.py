import os
import requests
from dotenv import load_dotenv

load_dotenv()


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

    def __init__(self, text=None, tool_calls=None, thinking=None):
        self.text = text  # str or None
        self.tool_calls = tool_calls or []  # list of ToolCall
        self.thinking = thinking  # str or None


# --- Gemini (The Brain) ---

class Gemini:
    """Gemini API - the brain of our agent."""

    def __init__(self):
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

        response = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=120
        )

        # Useful while developing
        if not response.ok:
            print("Gemini API Error:")
            print(response.text)

        response.raise_for_status()

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

            # Gemini calls the assistant role "model"
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
                thinking=None
            )

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        for part in parts:

            # Normal text
            if "text" in part:

                if part.get("thought", False):
                    thinking_parts.append(part["text"])
                else:
                    text_parts.append(part["text"])

            # Function/tool call
            elif "functionCall" in part:

                function_call = part["functionCall"]

                tool_calls.append(
                    ToolCall(
                        id=None,
                        name=function_call["name"],
                        args=function_call.get("args", {})
                    )
                )

        return Thought(
            text="\n".join(text_parts)
            if text_parts else None,

            tool_calls=tool_calls,

            thinking="\n".join(thinking_parts)
            if thinking_parts else None
        )

# --- Agent Class ---

class Agent:
    """A coding agent with conversation memory."""

    def __init__(self, brain):
        self.brain = brain
        self.conversation = []

    def handle_input(self, user_input):
        """Handle user input. Returns output string, raises AgentStop to quit."""
        if user_input.strip() == "/q":
            raise AgentStop()

        if not user_input.strip():
            return ""

        self.conversation.append({"role": "user", "content": user_input})

        try:
            thought = self.brain.think(self.conversation)
            if thought.thinking:
                lines = thought.thinking.strip().split("\n")[:5]
                for i, line in enumerate(lines):
                    prefix = "  💭 " if i == 0 else "     "
                    print(f"\033[2m{prefix}{line}\033[0m")
            text = thought.text or ""
            self.conversation.append({"role": "assistant", "content": text})
            return text
        except Exception as e:
            self.conversation.pop()  # Remove failed user message
            return f"Error: {e}"


# --- Main Loop ---

def main():
    brain = Gemini()
    agent = Agent(brain)
    print("⚡ Nanocode v0.2 (Conversation Memory)")
    print("Type '/q' to quit.\n")

    while True:
        try:
            user_input = input(">>")
            output = agent.handle_input(user_input)
            if output:
                print(f"\n{output}\n")

        except (AgentStop, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()