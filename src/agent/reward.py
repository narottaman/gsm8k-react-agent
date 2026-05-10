"""
src/agent/reward.py — Reward function for GRPO.
  answer_correct  (0.5) — exact match after normalization
  tool_efficiency (0.3) — fewer unnecessary calls = better
  format_valid    (0.2) — all steps produced valid JSON actions
"""
from __future__ import annotations
import re, logging
from src.agent.loop import AgentTrajectory, ActionType

logger  = logging.getLogger(__name__)
MAX_TOOL_CALLS = 6


def normalize_answer(ans: str) -> str:
    if not ans: return ""
    ans = ans.strip().lower()
    ans = re.sub(r"\$([\d,\.]+)", r"\1", ans)
    ans = re.sub(r"[^\d\.\-]", "", ans)
    try:
        val = float(ans)
        return str(int(val)) if val.is_integer() else str(val)
    except ValueError:
        return ans


def answer_correct_reward(traj: AgentTrajectory) -> float:
    if not traj.ground_truth or not traj.final_answer: return 0.0
    pred = normalize_answer(traj.final_answer)
    gold = normalize_answer(traj.ground_truth)
    if pred == gold:    return 1.0
    if gold and gold in pred: return 0.5
    return 0.0


def tool_efficiency_reward(traj: AgentTrajectory) -> float:
    calls = sum(1 for s in traj.steps if s.action.type == ActionType.TOOL_CALL)
    if calls == 0:           return 0.8 if traj.success else 0.2
    if calls <= 2:           return 1.0
    if calls <= MAX_TOOL_CALLS:
        return 1.0 - 0.5 * (calls - 2) / (MAX_TOOL_CALLS - 2)
    return max(0.0, 0.5 - 0.1 * (calls - MAX_TOOL_CALLS))


def format_valid_reward(traj: AgentTrajectory) -> float:
    if not traj.steps: return 0.0
    valid = sum(1 for s in traj.steps if s.action.type != ActionType.ERROR)
    return valid / len(traj.steps)


def compute_reward(traj: AgentTrajectory) -> float:
    r_correct    = answer_correct_reward(traj)
    r_efficiency = tool_efficiency_reward(traj)
    r_format     = format_valid_reward(traj)
    total = 0.5 * r_correct + 0.3 * r_efficiency + 0.2 * r_format
    traj.reward           = total
    traj.reward_breakdown = {"answer_correct": r_correct, "tool_efficiency": r_efficiency,
                             "format_valid": r_format, "total": total}
    # success = answer was actually correct (not just "agent answered")
    traj.success = (r_correct == 1.0)
    return total


def set_success(traj) -> None:
    """Set traj.success=True only if answer was correct."""
    from src.agent.reward import answer_correct_reward
    traj.success = answer_correct_reward(traj) == 1.0