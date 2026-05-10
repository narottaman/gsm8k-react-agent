"""
src/agent/tools.py
------------------
Two tools only for Phase 1:
  1. CodeExecutor — sandboxed Python subprocess
  2. Calculator   — safe eval for single expressions
"""

from __future__ import annotations
import subprocess, sys, tempfile, os, ast, operator, logging
from src.agent.loop import BaseTool

logger = logging.getLogger(__name__)

BLOCKED = ["import os", "import sys", "import subprocess", "open(", "__import__"]


class CodeExecutor(BaseTool):
    name        = "code_executor"
    description = "Executes Python code and returns stdout. Use for multi-step math and calculations."
    TIMEOUT     = 10

    def run(self, code: str) -> str:
        for b in BLOCKED:
            if b in code:
                return f"[code_executor] Blocked: '{b}' not allowed."
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            r = subprocess.run([sys.executable, tmp], capture_output=True,
                               text=True, timeout=self.TIMEOUT)
            out = r.stdout.strip()
            err = r.stderr.strip()
            if r.returncode != 0:
                return f"[code_executor] Error:\n{err}"
            return out or "[code_executor] No output."
        except subprocess.TimeoutExpired:
            return f"[code_executor] Timeout after {self.TIMEOUT}s."
        except Exception as e:
            return f"[code_executor] Failed: {e}"
        finally:
            try: os.unlink(tmp)
            except: pass

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"code": {"type": "string", "description": "Python code to run."}}
        }


class Calculator(BaseTool):
    """
    Safe arithmetic evaluator — no exec, no eval on arbitrary code.
    Supports: +, -, *, /, **, //, %, parentheses, and numeric literals.
    """
    name        = "calculator"
    description = "Evaluates a single math expression safely. E.g. '(34 * 17) / 2 + 5'."

    _SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def run(self, expression: str) -> str:
        try:
            result = self._safe_eval(ast.parse(expression, mode="eval").body)
            # Round floats that are effectively integers
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return str(result)
        except Exception as e:
            return f"[calculator] Error: {e}"

    def _safe_eval(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")
        elif isinstance(node, ast.BinOp):
            op = self._SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {node.op}")
            return op(self._safe_eval(node.left), self._safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = self._SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary op: {node.op}")
            return op(self._safe_eval(node.operand))
        else:
            raise ValueError(f"Unsupported node: {type(node)}")

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"expression": {"type": "string", "description": "Math expression to evaluate."}}
        }


def build_tool_registry():
    from src.agent.loop import ToolRegistry
    return ToolRegistry([CodeExecutor(), Calculator()])