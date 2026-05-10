"""
src/agent/reward.py
-------------------
Reward function for GRPO training on GSM8K.

Components:
  1. answer_correct  (0.5) — exact match after normalization
  2. tool_efficiency (0.3) — fewer unnecessary tool calls = better
  3. format_valid    (0.2) — agent responded in correct JSON format

Final reward: float in [0.0, 1.0]
"""

from __future__ import annotations
import re, logging
from src.agent.loop import AgentTrajectory, ActionType

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 6   # more than this = penalty kicks in


def normalize_answer(ans: str) -> str:
    """Strip units, commas, whitespace. Extract last number if embedded in text."""
    if ans is None:
        return ""
    ans = ans.strip().lower()
    # Remove common units
    ans = re.sub(r"\$([\d,\.]+)", r"\1", ans)   # $42 → 42
    ans = re.sub(r"[^\d\.\-]", "", ans)         # keep digits, dot, minus
    ans = ans.replace(",", "")
    # Try to parse as number and back to string to normalize 4.0 → 4
    try:
        val = float(ans)
        return str(int(val)) if val.is_integer() else str(val)
    except ValueError:
        return ans


def answer_correct_reward(traj: AgentTrajectory) -> float:
    """1.0 if exact match after normalization, 0.0 otherwise."""
    if traj.ground_truth is None or traj.final_answer is None:
        return 0.0
    pred = normalize_answer(traj.final_answer)
    gold = normalize_answer(traj.ground_truth)
    if pred == gold:
        return 1.0
    # Partial credit: answer is contained in a longer string
    if gold and gold in pred:
        return 0.5
    return 0.0


def tool_efficiency_reward(traj: AgentTrajectory) -> float:
    """
    Reward for using tools efficiently.
    0 tool calls when a tool was needed → 0.0
    Optimal (1-2 calls) → 1.0
    Too many calls → decreasing reward
    """
    tool_calls = sum(1 for s in traj.steps if s.action.type == ActionType.TOOL_CALL)

    if tool_calls == 0:
        # No tools used — fine only if the agent solved it correctly by reasoning
        return 0.8 if traj.success else 0.2

    if tool_calls <= 2:
        return 1.0
    if tool_calls <= MAX_TOOL_CALLS:
        # Linear decay from 1.0 at 2 calls to 0.5 at MAX_TOOL_CALLS
        return 1.0 - 0.5 * (tool_calls - 2) / (MAX_TOOL_CALLS - 2)

    # Over the limit — strong penalty
    return max(0.0, 0.5 - 0.1 * (tool_calls - MAX_TOOL_CALLS))


def format_valid_reward(traj: AgentTrajectory) -> float:
    """
    1.0 if agent always produced valid JSON actions.
    Penalizes malformed steps.
    """
    if not traj.steps:
        return 0.0
    valid = sum(1 for s in traj.steps if s.action.type != ActionType.ERROR)
    return valid / len(traj.steps)


def compute_reward(traj: AgentTrajectory) -> float:
    """
    Main reward function. Weights:
      answer_correct  0.5
      tool_efficiency 0.3
      format_valid    0.2
    """
    r_correct    = answer_correct_reward(traj)
    r_efficiency = tool_efficiency_reward(traj)
    r_format     = format_valid_reward(traj)

    total = (0.5 * r_correct) + (0.3 * r_efficiency) + (0.2 * r_format)

    traj.reward = total
    traj.reward_breakdown = {
        "answer_correct":    r_correct,
        "tool_efficiency":   r_efficiency,
        "format_valid":      r_format,
        "total":             total,
    }

    logger.debug(
        "task=%s | correct=%.2f | efficiency=%.2f | format=%.2f | total=%.3f",
        traj.task_id, r_correct, r_efficiency, r_format, total
    )
    return total