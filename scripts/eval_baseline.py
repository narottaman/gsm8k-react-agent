"""
scripts/eval_baseline.py
------------------------
Phase 1: Zero-shot baseline evaluation.
Metrics tracked:
  - accuracy       : answer exactly correct (primary metric)
  - success_rate   : agent produced a final answer (should always be ~100%)
  - mean_reward    : weighted reward (answer + efficiency + format)
  - tool_use_rate  : fraction of episodes where agent used any tool
  - mean_steps     : avg steps per episode
"""
import argparse, json, logging, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml, wandb
from src.agent.loop import ReActAgent, VLLMBackend, MockLLM, ActionType
from src.agent.tools import build_tool_registry
from src.agent.reward import compute_reward, answer_correct_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("logs/eval_baseline.log")]
)
logger = logging.getLogger(__name__)


def load_tasks(path: str, max_samples: int = None) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: tasks.append(json.loads(line))
    if max_samples:
        tasks = random.sample(tasks, min(max_samples, len(tasks)))
    return tasks


def run_eval(agent: ReActAgent, tasks: list[dict]) -> list[dict]:
    results = []
    for i, task in enumerate(tasks):
        try:
            traj = agent.run(
                task_id=task["task_id"],
                query=task["query"],
                ground_truth=task["ground_truth"],
            )
            compute_reward(traj)

            # Accuracy = answer was actually correct
            accuracy = answer_correct_reward(traj) == 1.0

            # Tool use = did agent call any tool
            tool_calls = sum(1 for s in traj.steps
                             if s.action.type == ActionType.TOOL_CALL)

            results.append({
                "task_id":        traj.task_id,
                "query":          traj.query[:100],
                "ground_truth":   traj.ground_truth,
                "predicted":      traj.final_answer,
                "accurate":       accuracy,           # ← correct answer
                "answered":       traj.final_answer is not None,
                "reward":         traj.reward,
                "reward_breakdown": traj.reward_breakdown,
                "total_steps":    traj.total_steps,
                "tool_calls":     tool_calls,
                "used_tools":     tool_calls > 0,
            })

            if (i + 1) % 10 == 0:
                done     = results
                acc      = sum(r["accurate"] for r in done) / len(done)
                tool_pct = sum(r["used_tools"] for r in done) / len(done)
                logger.info("[%d/%d] accuracy=%.3f | tool_use=%.1f%%",
                            i+1, len(tasks), acc, 100*tool_pct)

        except Exception as e:
            logger.warning("Task %s failed: %s", task["task_id"], e)

    return results


def summarize(results: list[dict]) -> dict:
    if not results: return {}
    n = len(results)
    return {
        "phase":          "baseline",
        "n":              n,
        "accuracy":       sum(r["accurate"]    for r in results) / n,
        "answered_rate":  sum(r["answered"]    for r in results) / n,
        "mean_reward":    sum(r["reward"]      for r in results if r["reward"]) / n,
        "tool_use_rate":  sum(r["used_tools"]  for r in results) / n,
        "mean_steps":     sum(r["total_steps"] for r in results) / n,
        "mean_tool_calls":sum(r["tool_calls"]  for r in results) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/eval_config.yaml")
    parser.add_argument("--mock",        action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    wandb.init(project=cfg.get("wandb_project", "gsm8k-react-agent"),
               name="eval-baseline", config=cfg, tags=["eval", "baseline"])

    tasks = load_tasks("data/gsm8k/test.jsonl",
                       max_samples=args.max_samples or cfg.get("max_eval_samples", 200))
    logger.info("Evaluating on %d tasks (baseline — no fine-tuning)", len(tasks))

    llm   = MockLLM() if args.mock else VLLMBackend(
                model_name=cfg.get("model_name"),
                temperature=cfg.get("temperature", 0.0),
                max_tokens=cfg.get("max_tokens", 512))
    tools = build_tool_registry()
    agent = ReActAgent(llm=llm, tools=tools, max_steps=cfg.get("max_steps", 8))

    results = run_eval(agent, tasks)
    summary = summarize(results)

    logger.info("=== BASELINE RESULTS ===")
    logger.info("  accuracy (correct answers): %.1f%%", 100 * summary["accuracy"])
    logger.info("  answered_rate:              %.1f%%", 100 * summary["answered_rate"])
    logger.info("  mean_reward:                %.4f",   summary["mean_reward"])
    logger.info("  tool_use_rate:              %.1f%%", 100 * summary["tool_use_rate"])
    logger.info("  mean_steps:                 %.2f",   summary["mean_steps"])
    logger.info("  mean_tool_calls:            %.2f",   summary["mean_tool_calls"])

    out = Path(cfg.get("results_dir", "data/results"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "baseline.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    wandb.log({
        "baseline/accuracy":        summary["accuracy"],
        "baseline/answered_rate":   summary["answered_rate"],
        "baseline/mean_reward":     summary["mean_reward"],
        "baseline/tool_use_rate":   summary["tool_use_rate"],
        "baseline/mean_steps":      summary["mean_steps"],
        "baseline/mean_tool_calls": summary["mean_tool_calls"],
    })
    wandb.finish()
    logger.info("Saved → data/results/baseline.json")


if __name__ == "__main__":
    main()