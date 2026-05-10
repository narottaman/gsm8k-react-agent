"""
src/rl/trajectory.py — Buffer + JSONL serialization for trajectories.
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from src.agent.loop import AgentTrajectory, AgentStep, Action, ActionType

logger = logging.getLogger(__name__)


def traj_to_dict(t: AgentTrajectory) -> dict:
    return {
        "task_id": t.task_id, "query": t.query, "final_answer": t.final_answer,
        "ground_truth": t.ground_truth, "reward": t.reward,
        "reward_breakdown": t.reward_breakdown, "success": t.success,
        "total_steps": t.total_steps, "total_time_ms": t.total_time_ms,
        "steps": [{"step_idx": s.step_idx, "thought": s.thought,
                   "action_type": s.action.type, "tool_name": s.action.tool_name,
                   "tool_args": s.action.tool_args, "content": s.action.content,
                   "observation": s.observation, "latency_ms": s.latency_ms}
                  for s in t.steps],
    }


def dict_to_traj(d: dict) -> AgentTrajectory:
    steps = [AgentStep(step_idx=s["step_idx"], thought=s["thought"],
                       action=Action(type=ActionType(s["action_type"]),
                                     tool_name=s.get("tool_name"),
                                     tool_args=s.get("tool_args"),
                                     content=s.get("content")),
                       observation=s.get("observation"), latency_ms=s.get("latency_ms", 0.0))
             for s in d.get("steps", [])]
    t = AgentTrajectory(task_id=d["task_id"], query=d["query"], steps=steps)
    t.final_answer = d.get("final_answer"); t.ground_truth = d.get("ground_truth")
    t.reward = d.get("reward"); t.reward_breakdown = d.get("reward_breakdown", {})
    t.success = d.get("success", False); t.total_steps = d.get("total_steps", 0)
    t.total_time_ms = d.get("total_time_ms", 0.0)
    return t


class TrajectoryBuffer:
    def __init__(self, save_path: str = "data/trajectories/buffer.jsonl"):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self._buf: list[AgentTrajectory] = []

    def add(self, t: AgentTrajectory) -> None: self._buf.append(t)
    def clear(self) -> None:                   self._buf.clear()
    def __len__(self) -> int:                  return len(self._buf)

    def save(self) -> None:
        with open(self.save_path, "w") as f:
            for t in self._buf:
                f.write(json.dumps(traj_to_dict(t)) + "\n")
        logger.info("Saved %d trajectories → %s", len(self._buf), self.save_path)

    def load(self, path: str = None) -> list[AgentTrajectory]:
        p = Path(path or self.save_path)
        trajs = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line: trajs.append(dict_to_traj(json.loads(line)))
        logger.info("Loaded %d trajectories from %s", len(trajs), p)
        return trajs

    def mean_reward(self) -> float:
        r = [t.reward for t in self._buf if t.reward is not None]
        return sum(r) / len(r) if r else 0.0

    def success_rate(self) -> float:
        if not self._buf: return 0.0
        return sum(1 for t in self._buf if t.success) / len(self._buf)


def trajectory_to_text(traj: AgentTrajectory) -> str:
    """Convert trajectory to flat text for GRPO training tokenization."""
    import json as _json
    parts = [f"[Problem] {traj.query}"]
    for step in traj.steps:
        parts.append(_json.dumps({
            "thought": step.thought,
            "action": {"type": step.action.type, "tool_name": step.action.tool_name,
                       "tool_args": step.action.tool_args, "content": step.action.content}
        }))
        if step.observation:
            parts.append(f"[Observation] {step.observation}")
    parts.append(f"[Answer] {traj.final_answer or ''}")
    return "\n".join(parts)