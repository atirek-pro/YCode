import os
import shlex
import subprocess

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

# --- Tool Context ---
class ToolContext:
    "What tools need to know about the agent's state"
    def __init__(self, memory=None):
        self.memory = memory # Memory object or None


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

class GitHubCLI:
    """Runs GitHub CLI (gh) and git commands for the current codebase."""
    name = "github"
    plan_safe = False
    description = (
        "Use this tool for ANY GitHub or git related task in the current codebase: creating or "
        "switching branches, committing, pushing, creating/viewing/merging pull requests, issues, "
        "releases, checking repo or CI status, etc. Runs a 'gh' (GitHub CLI) or 'git' command. "
        "Examples: 'git checkout -b feature-x', 'git push -u origin feature-x', "
        "'gh pr create --title \"Title\" --body \"Description\"', 'gh pr list', 'gh issue list'. "
        "Only works when the user is logged in to the GitHub CLI."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The full command, starting with 'gh' or 'git' (e.g., 'gh pr create --fill')"
            }
        },
        "required": ["command"]
    }

    def _check_auth(self):
        "Return an error message if gh is not usable, or None if it is installed and logged in."
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=15
            )
        except FileNotFoundError:
            return ("Error: GitHub CLI (gh) is not installed on this system, so I can't use it. "
                    "Install it from https://cli.github.com and run 'gh auth login'.")
        except subprocess.TimeoutExpired:
            return "Error: Timed out while checking GitHub CLI login status."
        if result.returncode != 0:
            return ("Error: I can't use the GitHub CLI because you are not logged in. "
                    "Please run 'gh auth login' in your terminal and try again.")
        return None

    def execute(self, context, command):
        print(f"--> GitHub: {command[:60]}")
        try:
            args = shlex.split(command, posix=True)
        except ValueError as e:
            return f"Error: Could not parse command - {e}"

        if not args or args[0] not in ("gh", "git"):
            return "Error: Command must start with 'gh' or 'git'."

        auth_error = self._check_auth()
        if auth_error:
            return auth_error

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=int(os.environ.get("YCODE_TIMEOUT", "30")),
                cwd=os.getcwd()
            )
        except FileNotFoundError:
            return f"Error: '{args[0]}' is not installed or not on PATH."
        except subprocess.TimeoutExpired:
            return "Error: Command Timed out."
        except Exception as e:
            return f"Error executing command: {e}"

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        if result.returncode != 0:
            output += f"EXIT CODE: {result.returncode}\n"
        if not output:
            output = "(NO OUTPUT)"
        return output.strip()

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

tools = [ReadFile(), WritePlan(), SaveMemory(), ListFiles(), SearchCodebase(), SearchWeb(), WriteFile(), RunCommand(), EditFile(), GitHubCLI()]
