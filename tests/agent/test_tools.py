"""
tests/agent/test_tools.py
Run: pytest tests/agent/test_tools.py -v
"""
import pytest
from src.agent.tools import CodeExecutor, Calculator, build_tool_registry
from src.agent.loop import Action, ToolRegistry


class TestCodeExecutor:
    def setup_method(self): self.tool = CodeExecutor()

    def test_basic_print(self):
        assert "hello" in self.tool.run(code='print("hello")')

    def test_arithmetic(self):
        assert "42" in self.tool.run(code="print(6 * 7)")

    def test_multiline(self):
        assert "30" in self.tool.run(code="x=[i**2 for i in range(5)]\nprint(sum(x))")

    def test_syntax_error_returns_error(self):
        result = self.tool.run(code="def bad(:")
        assert "Error" in result or "error" in result

    def test_blocked_import_os(self):
        assert "Blocked" in self.tool.run(code="import os")

    def test_blocked_import_subprocess(self):
        assert "Blocked" in self.tool.run(code="import subprocess")

    def test_no_output(self):
        assert "No output" in self.tool.run(code="x = 1 + 1")

    def test_schema_valid(self):
        s = self.tool.schema()
        assert s["name"] == "code_executor"
        assert "code" in s["parameters"]


class TestCalculator:
    def setup_method(self): self.tool = Calculator()

    def test_addition(self):       assert self.tool.run(expression="2 + 2") == "4"
    def test_multiplication(self): assert self.tool.run(expression="6 * 7") == "42"
    def test_division(self):       assert self.tool.run(expression="10 / 2") == "5"
    def test_power(self):          assert self.tool.run(expression="2 ** 8") == "256"
    def test_floor_div(self):      assert self.tool.run(expression="17 // 3") == "5"
    def test_modulo(self):         assert self.tool.run(expression="17 % 3") == "2"
    def test_parentheses(self):    assert self.tool.run(expression="(3 + 4) * 6") == "42"
    def test_float_result(self):   assert self.tool.run(expression="1 / 3").startswith("0.333")
    def test_float_to_int(self):   assert self.tool.run(expression="10.0 / 2") == "5"
    def test_negative(self):       assert self.tool.run(expression="-5 + 3") == "-2"
    def test_complex_expression(self):
        assert self.tool.run(expression="(34 * 17) / 2") == "289"

    def test_invalid_expression(self):
        result = self.tool.run(expression="2 + * 3")
        assert "Error" in result

    def test_string_blocked(self):
        result = self.tool.run(expression='"hello"')
        assert "Error" in result

    def test_schema_valid(self):
        s = self.tool.schema()
        assert s["name"] == "calculator"
        assert "expression" in s["parameters"]


class TestBuildToolRegistry:
    def test_returns_two_tools(self):
        r     = build_tool_registry()
        names = [t.name for t in r._tools.values()]
        assert "code_executor" in names
        assert "calculator" in names

    def test_execute_via_registry(self):
        r      = build_tool_registry()
        action = Action.tool_call("calculator", {"expression": "100 / 4"})
        assert "25" in r.execute(action)

    def test_code_exec_via_registry(self):
        r      = build_tool_registry()
        action = Action.tool_call("code_executor", {"code": "print(2**10)"})
        assert "1024" in r.execute(action)