"""
tests/agent/test_loop.py
Run: pytest tests/agent/test_loop.py -v
"""
import json, pytest
from src.agent.loop import (
    Action, ActionType, AgentStep, AgentTrajectory,
    MockLLM, ReActAgent, ToolRegistry, BaseTool,
    parse_llm_response, build_messages
)


class EchoTool(BaseTool):
    name        = "echo"
    description = "Returns input as output."
    def run(self, text: str = "") -> str:
        return f"echo: {text}"


def make_agent(responses=None, max_steps=10):
    llm   = MockLLM(responses=responses)
    tools = ToolRegistry([EchoTool()])
    return ReActAgent(llm=llm, tools=tools, max_steps=max_steps, )


class TestParseLLMResponse:
    def test_tool_call(self):
        raw = json.dumps({"thought": "searching", "action": {
            "type": "tool_call", "tool_name": "echo", "tool_args": {"text": "hi"}}})
        thought, action = parse_llm_response(raw)
        assert action.type == ActionType.TOOL_CALL
        assert action.tool_name == "echo"
        assert action.tool_args == {"text": "hi"}

    def test_final_answer(self):
        raw = json.dumps({"thought": "done", "action": {"type": "final_answer", "content": "42"}})
        _, action = parse_llm_response(raw)
        assert action.type == ActionType.FINAL_ANSWER
        assert action.content == "42"

    def test_malformed_json_becomes_final_answer(self):
        _, action = parse_llm_response("not json at all")
        assert action.type == ActionType.FINAL_ANSWER

    def test_strips_markdown_fences(self):
        raw = '```json\n{"thought":"ok","action":{"type":"final_answer","content":"7"}}\n```'
        _, action = parse_llm_response(raw)
        assert action.type == ActionType.FINAL_ANSWER
        assert action.content == "7"

    def test_unknown_action_type_is_error(self):
        raw = json.dumps({"thought": "?", "action": {"type": "explode"}})
        _, action = parse_llm_response(raw)
        assert action.type == ActionType.ERROR


class TestReActAgentLoop:
    def test_runs_to_final_answer(self):
        agent = make_agent()
        traj  = agent.run("t1", "What is 2+2?", ground_truth="4")
        assert traj.success is True
        assert traj.final_answer == "4"
        assert traj.total_steps == 2

    def test_trajectory_steps_captured(self):
        agent = make_agent()
        traj  = agent.run("t2", "Compute something.")
        assert len(traj.steps) == 2
        assert traj.steps[0].action.type == ActionType.TOOL_CALL
        assert traj.steps[0].observation is not None
        assert traj.steps[1].action.type == ActionType.FINAL_ANSWER

    def test_stops_at_max_steps(self):
        only_tools = [json.dumps({"thought": "still thinking", "action": {
            "type": "tool_call", "tool_name": "echo", "tool_args": {"text": "x"}}})] * 20
        agent = make_agent(responses=only_tools, max_steps=3)
        traj  = agent.run("t3", "Infinite loop test")
        assert traj.success is False
        assert traj.total_steps == 3
        assert "[max steps]" in traj.final_answer

    def test_ground_truth_stored(self):
        agent = make_agent()
        traj  = agent.run("t4", "query", ground_truth="99")
        assert traj.ground_truth == "99"

    def test_latency_recorded(self):
        agent = make_agent()
        traj  = agent.run("t5", "query")
        assert all(s.latency_ms >= 0 for s in traj.steps)
        assert traj.total_time_ms > 0

    def test_error_action_stops_loop(self):
        err = [json.dumps({"thought": "oops", "action": {"type": "bad_type"}})]
        agent = make_agent(responses=err, max_steps=5)
        traj  = agent.run("t6", "error test")
        assert traj.success is False
        assert traj.total_steps == 1


class TestToolRegistry:
    def test_execute_known_tool(self):
        registry = ToolRegistry([EchoTool()])
        action   = Action.tool_call("echo", {"text": "hello"})
        result   = registry.execute(action)
        assert "hello" in result

    def test_unknown_tool_error(self):
        registry = ToolRegistry()
        action   = Action.tool_call("ghost_tool", {})
        result   = registry.execute(action)
        assert "Unknown tool" in result

    def test_schemas_returned(self):
        registry = ToolRegistry([EchoTool()])
        schemas  = registry.schemas()
        assert any(s["name"] == "echo" for s in schemas)


class TestBuildMessages:
    def test_system_first(self):
        msgs = build_messages("test", [], [])
        assert msgs[0]["role"] == "system"
        assert "ARIA" in msgs[0]["content"] or "math" in msgs[0]["content"].lower()

    def test_user_query_present(self):
        msgs = build_messages("What is 7*6?", [], [])
        assert any("7*6" in m["content"] for m in msgs if m["role"] == "user")

    def test_previous_steps_replayed(self):
        step = AgentStep(0, "searching", Action.tool_call("echo", {"text": "hi"}),
                         observation="echo: hi")
        msgs = build_messages("q", [step], [])
        roles = [m["role"] for m in msgs]
        assert "assistant" in roles
        assert roles.count("user") >= 2   # original query + tool result