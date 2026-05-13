"""
scripts/eval_agent_behavior.py
-------------------------------
Agent-specific behavioral evaluation — goes beyond accuracy.

Tests what standard benchmarks don't:
  1. Tool Decision Quality  — did the agent call a tool when it should?
  2. Tool Selection         — code_executor vs calculator, right choice?
  3. Error Recovery         — does the agent retry after bad tool output?
  4. Step Efficiency        — does it converge in fewer steps over training?
  5. Problem Difficulty     — does tool use help more on hard vs easy problems?

This is the eval that shows AGENT behavior improvement, not just LLM accuracy.
Run on all checkpoints to compare behavioral differences across training phases.

Usage:
    python scripts/eval_agent_behavior.py --model Qwen/Qwen3-8B --label baseline
    python scripts/eval_agent_behavior.py --model checkpoints/sft_full --label full_sft
    python scripts/eval_agent_behavior.py --model checkpoints/rl/final --lora_base Qwen/Qwen3-8B --label lora_grpo
"""
import argparse, json, logging, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb
from src.agent.loop import ReActAgent, VLLMBackend, MockLLM, ActionType
from src.agent.tools import build_tool_registry
from src.agent.reward import compute_reward, normalize_answer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Problem difficulty classifier
# Simple heuristic: count operations mentioned in the problem text
# ---------------------------------------------------------------------------

def classify_difficulty(query: str) -> str:
    """Classify GSM8K problem as easy/medium/hard by operation count."""
    ops = ["how many", "total", "left", "remaining", "each", "per",
           "times", "twice", "half", "percent", "ratio", "difference"]
    count = sum(1 for op in ops if op in query.lower())
    if count <= 2:  return "easy"
    if count <= 4:  return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Merge LoRA if needed
# ---------------------------------------------------------------------------

def resolve_model(model_path: str, lora_base: str = None) -> str:
    from pathlib import Path as P
    p = P(model_path)
    if (p / "adapter_config.json").exists() and not (p / "config.json").exists():
        if not lora_base:
            raise ValueError(f"{model_path} is a LoRA adapter. Pass --lora_base")
        merged = model_path.rstrip("/") + "_merged"
        if not (P(merged) / "config.json").exists():
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            logger.info("Merging LoRA %s + %s → %s", model_path, lora_base, merged)
            P(merged).mkdir(parents=True, exist_ok=True)
            tok   = AutoTokenizer.from_pretrained(lora_base)
            model = AutoModelForCausalLM.from_pretrained(lora_base, torch_dtype=torch.bfloat16, device_map="auto")
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            model.save_pretrained(merged)
            tok.save_pretrained(merged)
        return merged
    return model_path


# ---------------------------------------------------------------------------
# Main behavioral eval
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--lora_base",   default="Qwen/Qwen3-8B",
                        help="Base model for LoRA merging")
    parser.add_argument("--label",       default="model")
    parser.add_argument("--split",       default="test")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_steps",   type=int, default=8)
    parser.add_argument("--mock",        action="store_true")
    parser.add_argument("--wandb",       action="store_true")
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="gsm8k-react-agent",
                   name=f"agent-behavior-{args.label}",
                   tags=["behavior", args.label])

    # Load tasks
    tasks = []
    with open(f"data/gsm8k/{args.split}.jsonl") as f:
        for line in f:
            line = line.strip()
            if line: tasks.append(json.loads(line))
    tasks = random.sample(tasks, min(args.max_samples, len(tasks)))

    # Classify by difficulty
    for t in tasks:
        t["difficulty"] = classify_difficulty(t["query"])

    # Load model
    model_path = args.model
    if not args.mock:
        model_path = resolve_model(args.model, args.lora_base)
        llm = VLLMBackend(model_name=model_path, temperature=0.0, max_tokens=512)
    else:
        llm = MockLLM()

    tools = build_tool_registry()
    agent = ReActAgent(llm=llm, tools=tools, max_steps=args.max_steps)

    # Run eval
    results = []
    for i, task in enumerate(tasks):
        traj = agent.run(task_id=task["task_id"], query=task["query"],
                         ground_truth=task["ground_truth"])
        compute_reward(traj)

        tool_calls  = [s for s in traj.steps if s.action.type == ActionType.TOOL_CALL]
        code_calls  = [s for s in tool_calls if s.action.tool_name == "code_executor"]
        calc_calls  = [s for s in tool_calls if s.action.tool_name == "calculator"]

        # Error recovery: did tool return an error and agent continued?
        error_obs   = [s for s in tool_calls if s.observation and "[error]" in s.observation.lower()]
        recovered   = len(error_obs) > 0 and traj.success

        results.append({
            "task_id":       traj.task_id,
            "difficulty":    task["difficulty"],
            "accurate":      traj.success,
            "used_tool":     len(tool_calls) > 0,
            "n_tool_calls":  len(tool_calls),
            "n_code":        len(code_calls),
            "n_calc":        len(calc_calls),
            "n_steps":       traj.total_steps,
            "error_recovery":recovered,
            "reward":        traj.reward,
            "reward_breakdown": traj.reward_breakdown,
            "predicted":     traj.final_answer,
            "ground_truth":  traj.ground_truth,
        })

        if (i + 1) % 20 == 0:
            acc  = sum(r["accurate"]  for r in results) / len(results)
            tool = sum(r["used_tool"] for r in results) / len(results)
            logger.info("[%d/%d] accuracy=%.1f%% | tool_use=%.1f%%",
                        i+1, len(tasks), 100*acc, 100*tool)

    # Compute behavioral metrics
    n = len(results)
    by_diff = {"easy": [], "medium": [], "hard": []}
    for r in results:
        by_diff[r["difficulty"]].append(r)

    def metrics(subset):
        if not subset: return {}
        sn = len(subset)
        return {
            "n":             sn,
            "accuracy":      sum(r["accurate"]      for r in subset) / sn,
            "tool_use_rate": sum(r["used_tool"]      for r in subset) / sn,
            "mean_steps":    sum(r["n_steps"]         for r in subset) / sn,
            "mean_tools":    sum(r["n_tool_calls"]    for r in subset) / sn,
            "code_rate":     sum(r["n_code"] > 0      for r in subset) / sn,
            "calc_rate":     sum(r["n_calc"] > 0      for r in subset) / sn,
            "error_recover": sum(r["error_recovery"]  for r in subset) / sn,
            "mean_reward":   sum(r["reward"] or 0     for r in subset) / sn,
        }

    summary = {
        "label":    args.label,
        "model":    args.model,
        "overall":  metrics(results),
        "by_difficulty": {
            "easy":   metrics(by_diff["easy"]),
            "medium": metrics(by_diff["medium"]),
            "hard":   metrics(by_diff["hard"]),
        }
    }

    # Print behavioral report
    print(f"\n{'='*65}")
    print(f"AGENT BEHAVIOR REPORT — {args.label}")
    print(f"{'='*65}")
    o = summary["overall"]
    print(f"  Accuracy:          {o['accuracy']:.1%}")
    print(f"  Tool use rate:     {o['tool_use_rate']:.1%}")
    print(f"  Mean steps:        {o['mean_steps']:.2f}")
    print(f"  Mean tool calls:   {o['mean_tools']:.2f}")
    print(f"  Code executor use: {o['code_rate']:.1%}")
    print(f"  Calculator use:    {o['calc_rate']:.1%}")
    print(f"  Error recovery:    {o['error_recover']:.1%}")
    print(f"  Mean reward:       {o['mean_reward']:.3f}")
    print(f"\n  By difficulty:")
    for diff in ["easy", "medium", "hard"]:
        d = summary["by_difficulty"][diff]
        if d:
            print(f"    {diff:6s}: acc={d['accuracy']:.1%} | "
                  f"tool={d['tool_use_rate']:.1%} | "
                  f"steps={d['mean_steps']:.1f} | "
                  f"n={d['n']}")
    print(f"{'='*65}\n")

    # Save
    out = Path("data/results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"behavior_{args.label}.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logger.info("Saved → data/results/behavior_%s.json", args.label)

    if args.wandb:
        flat = {f"behavior/{k}": v for k, v in o.items() if isinstance(v, (int, float))}
        for diff, dm in summary["by_difficulty"].items():
            for k, v in dm.items():
                if isinstance(v, (int, float)):
                    flat[f"behavior_{diff}/{k}"] = v
        wandb.log(flat)
        wandb.finish()


if __name__ == "__main__":
    main()
