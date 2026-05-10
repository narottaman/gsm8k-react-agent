"""
src/agent/loop.py — ReAct loop for GSM8K agent.
Think → Act → Observe → repeat → FinalAnswer
No APIs. Open models only (Qwen3-8B via vLLM on Sol).
"""
from __future__ import annotations
import re
import json, logging, time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


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
    """One full episode. GRPO trains on a batch of these."""
    task_id:          str
    query:            str
    steps:            list[AgentStep] = field(default_factory=list)
    final_answer:     Optional[str]   = None
    ground_truth:     Optional[str]   = None
    reward:           Optional[float] = None
    reward_breakdown: dict            = field(default_factory=dict)
    answered:         bool            = False  # agent produced final_answer
    success:          bool            = False  # answer was correct
    total_steps:      int             = 0
    total_time_ms:    float           = 0.0


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------

class BaseLLM:
    def generate(self, messages: list[dict]) -> str:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """Deterministic mock — no GPU needed, used in tests."""
    def __init__(self, responses: Optional[list[str]] = None):
        self._responses = responses or [
            json.dumps({"thought": "I'll compute this precisely with code.",
                        "action": {"type": "tool_call", "tool_name": "code_executor",
                                   "tool_args": {"code": "print(2 + 2)"}}}),
            json.dumps({"thought": "Output was 4. That is the answer.",
                        "action": {"type": "final_answer", "content": "4"}}),
        ]
        self._idx = 0

    def generate(self, messages: list[dict]) -> str:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp


class VLLMBackend(BaseLLM):
    """Qwen3-8B via vLLM on Sol A100. Loaded once, reused across episodes."""
    def __init__(self, model_name: str = "Qwen/Qwen3-8B",
                 temperature: float = 0.7, max_tokens: int = 512):
        import os
        import torch._dynamo
        # Sol /etc/python/sitecustomize.py breaks torch.compile subprocess
        # Must disable compilation BEFORE vLLM spawns its engine subprocess
        os.environ["VLLM_COMPILE_LEVEL"] = "0"
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        torch._dynamo.config.suppress_errors = True
        from vllm import LLM, SamplingParams
        logger.info("Loading vLLM model: %s", model_name)
        self.llm    = LLM(model=model_name, dtype="bfloat16",
                          gpu_memory_utilization=0.85,
                          enforce_eager=True,
                          compilation_config={"level": 0})
        self.params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                     stop=["<|im_end|>"])

    def generate(self, messages: list[dict]) -> str:
        prompt  = self._chat_template(messages)
        outputs = self.llm.generate([prompt], self.params)
        return outputs[0].outputs[0].text.strip()

    def _chat_template(self, messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role, content = m["role"], m["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
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
# Parser + Context Builder
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> str:
    """Extract last number from plain text. '42 dollars' -> '42'."""
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return text.strip()


def parse_llm_response(raw: str) -> tuple[str, Action]:
    """
    Robust parser handling:
      1. Qwen3 <think>...</think> prefix (thinking mode)
      2. markdown code fences
      3. JSON object anywhere in text
      4. Plain text fallback with number extraction
    """
    raw = raw.strip()

    # Step 1: Strip Qwen3 <think>...</think> block
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Step 2: Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]).strip()

    # Step 3: Extract first complete JSON object
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = brace_match.group(0) if brace_match else raw

    # Step 4: Try JSON parse
    try:
        parsed  = json.loads(json_str)
        thought = parsed.get("thought", "")
        act     = parsed.get("action", {})
        atype   = act.get("type", "error")
        if atype == "tool_call":
            return thought, Action.tool_call(act.get("tool_name", ""), act.get("tool_args", {}))
        elif atype == "final_answer":
            return thought, Action.final_answer(act.get("content", ""))
        else:
            return thought, Action.error(f"Unknown type: {atype}")
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 5: Fallback - extract last number from raw text
    number = _extract_number(raw)
    logger.debug("Parser fallback: extracted %r from: %s", number, raw[:100])
    return "Extracted from text.", Action.final_answer(number)


SYSTEM_PROMPT = """You are a math reasoning agent. You MUST use tools to solve every problem.

STRICT RULES:
1. ALWAYS use code_executor or calculator before giving a final answer
2. NEVER guess or compute mentally — always verify with a tool
3. Respond ONLY with valid JSON — no extra text, no markdown

Tool call format:
{"thought": "I need to compute X", "action": {"type": "tool_call", "tool_name": "code_executor", "tool_args": {"code": "print(...)"}}}

After seeing tool result, give final answer:
{"thought": "The result is X", "action": {"type": "final_answer", "content": "42"}}

Available tools:
TOOLS_PLACEHOLDER

IMPORTANT: You must call a tool on your FIRST response. Never answer directly without computing.
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
                "action": {"type": step.action.type, "tool_name": step.action.tool_name,
                           "tool_args": step.action.tool_args, "content": step.action.content}
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
            t0              = time.time()
            messages        = build_messages(query, traj.steps, self.tools.schemas())
            raw             = self.llm.generate(messages)
            thought, action = parse_llm_response(raw)
            latency         = (time.time() - t0) * 1000

            if action.type == ActionType.FINAL_ANSWER:
                traj.steps.append(AgentStep(step_idx, thought, action, latency_ms=latency))
                traj.final_answer = action.content
                traj.answered     = True   # agent produced an answer
                traj.success      = False  # set to True by reward.py if correct
                break
            elif action.type == ActionType.TOOL_CALL:
                obs = self.tools.execute(action)
                traj.steps.append(AgentStep(step_idx, thought, action, obs, latency_ms=latency))
            else:
                traj.steps.append(AgentStep(step_idx, thought, action,
                                            f"[error] {action.content}", latency_ms=latency))
                break
        else:
            traj.final_answer = "[max steps]"
            traj.success      = False

        traj.total_steps   = len(traj.steps)
        traj.total_time_ms = (time.time() - start) * 1000
        return traj