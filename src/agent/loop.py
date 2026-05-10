"""
src/agent/loop.py
-----------------
ReAct loop for GSM8K math agent.
Think → Act → Observe → repeat → FinalAnswer

Each run returns AgentTrajectory used by GRPO trainer.
No APIs. Open models only (Qwen3-8B via vLLM).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums + Dataclasses
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    TOOL_CALL    = "tool_call"
    FINAL_ANSWER = "final_answer"
    ERROR        = "error"


@dataclass
class Action:
    type:      ActionType
    tool_name: Optional[str]  = None
    tool_args: Optional[dict] = None
    content:   Optional[str]  = None

    @classmethod
    def tool_call(cls, tool_name: str, tool_args: dict) -> "Action":
        return cls(type=ActionType.TOOL_CALL, tool_name=tool_name, tool_args=tool_args)

    @classmethod
    def final_answer(cls, content: str) -> "Action":
        return cls(type=ActionType.FINAL_ANSWER, content=content)

    @classmethod
    def error(cls, msg: str) -> "Action":
        return cls(type=ActionType.ERROR, content=msg)


@dataclass
class AgentStep:
    step_idx:    int
    thought:     str
    action:      Action
    observation: Optional[str] = None
    latency_ms:  float = 0.0


@dataclass
class AgentTrajectory:
    """One full episode. GRPO trains on these."""
    task_id:          str
    query:            str
    steps:            list[AgentStep] = field(default_factory=list)
    final_answer:     Optional[str]   = None
    ground_truth:     Optional[str]   = None
    reward:           Optional[float] = None
    reward_breakdown: dict            = field(default_factory=dict)
    success:          bool            = False
    total_steps:      int             = 0
    total_time_ms:    float           = 0.0


# ---------------------------------------------------------------------------
# LLM Backend
# ---------------------------------------------------------------------------

class BaseLLM:
    def generate(self, messages: list[dict]) -> str:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """For tests — no GPU needed."""
    def __init__(self, responses: Optional[list[str]] = None):
        self._responses = responses or [
            json.dumps({
                "thought": "I should execute this as code to be precise.",
                "action": {"type": "tool_call", "tool_name": "code_executor",
                           "tool_args": {"code": "print(2 + 2)"}}
            }),
            json.dumps({
                "thought": "The code returned 4. That is the answer.",
                "action": {"type": "final_answer", "content": "4"}
            }),
        ]
        self._idx = 0

    def generate(self, messages: list[dict]) -> str:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp


class VLLMBackend(BaseLLM):
    """
    Real backend — Qwen3-8B via vLLM on Sol A100.
    Loaded once in train.py, passed in here.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-8B-Instruct",
                 temperature: float = 0.7, max_tokens: int = 512):
        from vllm import LLM, SamplingParams
        self.llm    = LLM(model=model_name, dtype="bfloat16")
        self.params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def generate(self, messages: list[dict]) -> str:
        prompt  = self._apply_chat_template(messages)
        outputs = self.llm.generate([prompt], self.params)
        return outputs[0].outputs[0].text.strip()

    def _apply_chat_template(self, messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class BaseTool:
    name: str        = "base"
    description: str = ""

    def run(self, **kwargs) -> str:
        raise NotImplementedError

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description}


class ToolRegistry:
    def __init__(self, tools: Optional[list[BaseTool]] = None):
        self._tools: dict[str, BaseTool] = {}
        for t in (tools or []):
            self._tools[t.name] = t

    def execute(self, action: Action) -> str:
        if action.tool_name not in self._tools:
            return f"[error] Unknown tool: {action.tool_name}"
        try:
            return self._tools[action.tool_name].run(**(action.tool_args or {}))
        except Exception as e:
            return f"[error] {action.tool_name} failed: {e}"

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_llm_response(raw: str) -> tuple[str, Action]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])
    try:
        parsed  = json.loads(raw)
        thought = parsed.get("thought", "")
        act     = parsed.get("action", {})
        atype   = act.get("type", "error")
        if atype == "tool_call":
            return thought, Action.tool_call(act.get("tool_name", ""), act.get("tool_args", {}))
        elif atype == "final_answer":
            return thought, Action.final_answer(act.get("content", ""))
        else:
            return thought, Action.error(f"Unknown type: {atype}")
    except json.JSONDecodeError:
        return "Parse failed.", Action.final_answer(raw)


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a math reasoning agent. Solve problems step by step.
Always respond with valid JSON only:

For tool use:
{"thought": "...", "action": {"type": "tool_call", "tool_name": "code_executor", "tool_args": {"code": "print(...)"}}}

For final answer:
{"thought": "...", "action": {"type": "final_answer", "content": "42"}}

Available tools:
TOOLS_PLACEHOLDER

Rules:
- Use code_executor for arithmetic or multi-step math
- Use calculator for simple single operations  
- Final answer must be the numeric value only
- Never guess — always compute
"""

def build_messages(query: str, steps: list[AgentStep], tools: list[dict]) -> list[dict]:
    system = SYSTEM_PROMPT.replace("TOOLS_PLACEHOLDER", json.dumps(tools, indent=2))
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": query},
    ]
    for step in steps:
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "thought": step.thought,
                "action": {
                    "type":      step.action.type,
                    "tool_name": step.action.tool_name,
                    "tool_args": step.action.tool_args,
                    "content":   step.action.content,
                }
            })
        })
        if step.observation is not None:
            messages.append({"role": "user", "content": f"[Tool Result]\n{step.observation}"})
    return messages


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

class ReActAgent:
    def __init__(self, llm: BaseLLM, tools: Optional[ToolRegistry] = None, max_steps: int = 8):
        self.llm       = llm
        self.tools     = tools or ToolRegistry()
        self.max_steps = max_steps

    def run(self, task_id: str, query: str, ground_truth: Optional[str] = None) -> AgentTrajectory:
        traj  = AgentTrajectory(task_id=task_id, query=query, ground_truth=ground_truth)
        start = time.time()

        for step_idx in range(self.max_steps):
            t0               = time.time()
            messages         = build_messages(query, traj.steps, self.tools.schemas())
            raw              = self.llm.generate(messages)
            thought, action  = parse_llm_response(raw)

            if action.type == ActionType.FINAL_ANSWER:
                traj.steps.append(AgentStep(step_idx, thought, action, latency_ms=(time.time()-t0)*1000))
                traj.final_answer = action.content
                traj.success      = True
                break
            elif action.type == ActionType.TOOL_CALL:
                obs = self.tools.execute(action)
                traj.steps.append(AgentStep(step_idx, thought, action, obs, latency_ms=(time.time()-t0)*1000))
            else:
                traj.steps.append(AgentStep(step_idx, thought, action, f"[error] {action.content}", latency_ms=(time.time()-t0)*1000))
                break
        else:
            traj.final_answer = "[max steps]"
            traj.success      = False

        traj.total_steps   = len(traj.steps)
        traj.total_time_ms = (time.time() - start) * 1000
        return traj