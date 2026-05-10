"""
tests/rl/test_grpo.py
Run: pytest tests/rl/test_grpo.py -v
Tests GRPO logic + trajectory buffer — no GPU needed.
"""
import json, os, tempfile, pytest
from src.agent.loop import AgentTrajectory, AgentStep, Action, ActionType
from src.rl.trajectory import TrajectoryBuffer, traj_to_dict, dict_to_traj
from src.rl.trajectory import trajectory_to_text


def make_traj(task_id="t1", reward=0.8, success=True):
    t = AgentTrajectory(task_id=task_id, query="What is 2+2?",
                        final_answer="4", ground_truth="4",
                        reward=reward, success=success,
                        reward_breakdown={"answer_correct": 1.0, "tool_efficiency": 0.8,
                                          "format_valid": 1.0, "total": reward})
    step = AgentStep(0, "I'll compute", Action.tool_call("calculator", {"expression": "2+2"}),
                     observation="4")
    ans  = AgentStep(1, "Answer is 4", Action.final_answer("4"))
    t.steps      = [step, ans]
    t.total_steps = 2
    return t


class TestTrajectoryBuffer:
    def test_add_and_len(self):
        buf = TrajectoryBuffer("data/trajectories/_test.jsonl")
        buf.add(make_traj("t1"))
        buf.add(make_traj("t2"))
        assert len(buf) == 2

    def test_mean_reward(self):
        buf = TrajectoryBuffer("data/trajectories/_test.jsonl")
        buf.add(make_traj("t1", reward=0.5))
        buf.add(make_traj("t2", reward=1.0))
        assert buf.mean_reward() == pytest.approx(0.75)

    def test_success_rate(self):
        buf = TrajectoryBuffer("data/trajectories/_test.jsonl")
        buf.add(make_traj("t1", success=True))
        buf.add(make_traj("t2", success=False))
        assert buf.success_rate() == pytest.approx(0.5)

    def test_clear(self):
        buf = TrajectoryBuffer("data/trajectories/_test.jsonl")
        buf.add(make_traj())
        buf.clear()
        assert len(buf) == 0

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            buf = TrajectoryBuffer(path)
            buf.add(make_traj("t1", reward=0.9))
            buf.add(make_traj("t2", reward=0.6))
            buf.save()

            loaded = buf.load(path)
            assert len(loaded) == 2
            assert loaded[0].task_id == "t1"
            assert loaded[0].reward  == pytest.approx(0.9)
            assert loaded[1].task_id == "t2"
        finally:
            os.unlink(path)


class TestSerialization:
    def test_roundtrip(self):
        t = make_traj("roundtrip_test", reward=0.75)
        d = traj_to_dict(t)
        r = dict_to_traj(d)
        assert r.task_id       == t.task_id
        assert r.query         == t.query
        assert r.final_answer  == t.final_answer
        assert r.ground_truth  == t.ground_truth
        assert r.reward        == pytest.approx(t.reward)
        assert r.success       == t.success
        assert len(r.steps)    == len(t.steps)

    def test_steps_preserved(self):
        t = make_traj()
        d = traj_to_dict(t)
        r = dict_to_traj(d)
        assert r.steps[0].action.type     == ActionType.TOOL_CALL
        assert r.steps[0].action.tool_name == "calculator"
        assert r.steps[0].observation      == "4"
        assert r.steps[1].action.type      == ActionType.FINAL_ANSWER
        assert r.steps[1].action.content   == "4"


class TestTrajectoryToText:
    def test_contains_problem(self):
        t    = make_traj()
        text = trajectory_to_text(t)
        assert "[Problem]" in text
        assert "2+2" in text

    def test_contains_answer(self):
        t    = make_traj()
        text = trajectory_to_text(t)
        assert "[Answer]" in text
        assert "4" in text

    def test_contains_steps(self):
        t    = make_traj()
        text = trajectory_to_text(t)
        assert "calculator" in text