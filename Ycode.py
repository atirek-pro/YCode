import os
import sys

from dotenv import load_dotenv

from brain import request_with_retry, ToolCall, Thought, Brain, Gemini, BRAINS
from tools import (
    ToolContext, ReadFile, WriteFile, WritePlan, EditFile, ListFiles, SearchCodebase,
    SearchWeb, SaveMemory, RunCommand, GitHubCLI, get_tool, tool_definitions, tools,
)

load_dotenv()

# --- Exceptions ---

class AgentStop(Exception):
    """Raised when the agent should stop processing."""
    pass


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