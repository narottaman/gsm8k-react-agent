"""
tests/agent/test_reward.py
Run: pytest tests/agent/test_reward.py -v
"""
import pytest
from src.agent.loop import AgentTrajectory, AgentStep, Action, ActionType
from src.agent.reward import (
    normalize_answer, answer_correct_reward,
    tool_efficiency_reward, format_valid_reward, compute_reward
)


def make_traj(final_answer, ground_truth, success=True, steps=None):
    t = AgentTrajectory(task_id="test", query="q",
                        final_answer=final_answer, ground_truth=ground_truth,
                        success=success, steps=steps or [])
    return t


def tool_step(idx=0):
    return AgentStep(idx, "thinking", Action.tool_call("calculator", {"expression": "1+1"}),
                     observation="2")

def answer_step(idx=1):
    return AgentStep(idx, "done", Action.final_answer("42"))

def error_step(idx=0):
    return AgentStep(idx, "oops", Action.error("bad"))


class TestNormalizeAnswer:
    def test_strips_dollar(self):   assert normalize_answer("$42") == "42"
    def test_strips_comma(self):    assert normalize_answer("1,000") == "1000"
    def test_float_to_int(self):    assert normalize_answer("42.0") == "42"
    def test_strips_whitespace(self): assert normalize_answer("  42  ") == "42"
    def test_empty_string(self):    assert normalize_answer("") == ""
    def test_none(self):            assert normalize_answer(None) == ""
    def test_keeps_float(self):     assert normalize_answer("3.14") == "3.14"


class TestAnswerCorrectReward:
    def test_exact_match(self):
        t = make_traj("42", "42")
        assert answer_correct_reward(t) == 1.0

    def test_dollar_sign_normalized(self):
        t = make_traj("$42", "42")
        assert answer_correct_reward(t) == 1.0

    def test_float_normalized(self):
        t = make_traj("42.0", "42")
        assert answer_correct_reward(t) == 1.0

    def test_wrong_answer(self):
        t = make_traj("43", "42")
        assert answer_correct_reward(t) == 0.0

    def test_no_ground_truth(self):
        t = make_traj("42", None)
        assert answer_correct_reward(t) == 0.0

    def test_no_prediction(self):
        t = make_traj(None, "42")
        assert answer_correct_reward(t) == 0.0


class TestToolEfficiencyReward:
    def test_one_tool_call(self):
        t = make_traj("42", "42", steps=[tool_step(), answer_step()])
        assert tool_efficiency_reward(t) == 1.0

    def test_two_tool_calls(self):
        t = make_traj("42", "42", steps=[tool_step(0), tool_step(1), answer_step(2)])
        assert tool_efficiency_reward(t) == 1.0

    def test_no_tools_success(self):
        t = make_traj("42", "42", success=True, steps=[answer_step(0)])
        assert tool_efficiency_reward(t) == 0.8

    def test_no_tools_failure(self):
        t = make_traj("43", "42", success=False, steps=[answer_step(0)])
        assert tool_efficiency_reward(t) == 0.2

    def test_many_tool_calls_penalized(self):
        many_steps = [tool_step(i) for i in range(10)] + [answer_step(10)]
        t = make_traj("42", "42", steps=many_steps)
        r = tool_efficiency_reward(t)
        assert r < 1.0
        assert r >= 0.0


class TestFormatValidReward:
    def test_all_valid(self):
        t = make_traj("42", "42", steps=[tool_step(), answer_step()])
        assert format_valid_reward(t) == 1.0

    def test_one_error_step(self):
        t = make_traj("42", "42", steps=[error_step(), answer_step()])
        assert format_valid_reward(t) == 0.5

    def test_empty_steps(self):
        t = make_traj("42", "42", steps=[])
        assert format_valid_reward(t) == 0.0


class TestComputeReward:
    def test_perfect_trajectory(self):
        # Correct answer, 1 tool call, no errors
        t = make_traj("42", "42", success=True,
                      steps=[tool_step(), answer_step()])
        r = compute_reward(t)
        assert r == pytest.approx(1.0)
        assert t.reward == r
        assert "total" in t.reward_breakdown

    def test_wrong_answer_still_scores_partial(self):
        # Wrong answer but efficient tool use and valid format
        t = make_traj("43", "42", success=False,
                      steps=[tool_step(), answer_step()])
        r = compute_reward(t)
        assert 0.0 < r < 1.0

    def test_reward_in_zero_one_range(self):
        t = make_traj("42", "42", steps=[tool_step(), answer_step()])
        r = compute_reward(t)
        assert 0.0 <= r <= 1.0

    def test_breakdown_keys_present(self):
        t = make_traj("42", "42", steps=[tool_step(), answer_step()])
        compute_reward(t)
        assert set(t.reward_breakdown.keys()) == {
            "answer_correct", "tool_efficiency", "format_valid", "total"}