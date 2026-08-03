import datetime
import random
import os
import math
from typing import Dict, Any

class ToolRegistry:
    """
    Tool & Plugin Registry.
    Allows LLM to execute local functions for calculation, time, file reading, and searching.
    """
    def __init__(self):
        self.tools = {
            "calculator": self.calculator,
            "clock": self.clock,
            "file_reader": self.file_reader,
            "random": self.random_generator
        }

    def calculator(self, expression: str) -> str:
        """Evaluates basic mathematical expression safely."""
        try:
            # Clean expression
            cleaned = re.sub(r'[^0-9+\-*/().]', '', expression)
            result = eval(cleaned, {"__builtins__": None, "math": math})
            return f"Calculator Result: {result}"
        except Exception as e:
            return f"Calculator Error: {str(e)}"

    def clock(self, dummy: str = "") -> str:
        """Returns current system date and time."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Current Time: {now}"

    def file_reader(self, filepath: str) -> str:
        """Reads contents of a local file."""
        filepath = filepath.strip()
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f"File Content:\n{f.read()[:300]}"
        return f"File not found: {filepath}"

    def random_generator(self, arg: str = "1-100") -> str:
        """Generates random integer in range."""
        try:
            parts = arg.split("-")
            low, high = int(parts[0]), int(parts[1])
            val = random.randint(low, high)
            return f"Random Value ({low}-{high}): {val}"
        except Exception:
            return f"Random Value: {random.randint(1, 100)}"

    def execute_tool(self, tool_name: str, argument: str) -> str:
        """Executes requested tool by name."""
        tool_name = tool_name.lower().strip()
        if tool_name in self.tools:
            return self.tools[tool_name](argument)
        return f"Unknown tool: {tool_name}"