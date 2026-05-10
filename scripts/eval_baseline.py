"""
scripts/eval_baseline.py
------------------------
Phase 1: Zero-shot baseline evaluation.
Runs Qwen3-8B (no fine-tuning) on GSM8K test set via ReAct agent.
Saves results to data/results/baseline.json + logs to W&B.

This establishes the floor — everything else is measured against this.

Usage:
    python scripts/eval_baseline.py --config configs/eval_config.yaml
"""
import argparse, json, logging, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml, wandb
from src.agent.loop import ReActAgent, VLLMBackend, MockLLM
from src.agent.tools import build_tool_registry
from src.agent.reward import compute_reward

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
            traj = agent.run(task_id=task["task_id"], query=task["query"],
                             ground_truth=task["ground_truth"])
            compute_reward(traj)
            results.append({
                "task_id":       traj.task_id,
                "query":         traj.query,
                "ground_truth":  traj.ground_truth,
                "predicted":     traj.final_answer,
                "success":       traj.success,
                "reward":        traj.reward,
                "reward_breakdown": traj.reward_breakdown,
                "total_steps":   traj.total_steps,
            })
            if (i + 1) % 10 == 0:
                done    = results
                success = sum(r["success"] for r in done) / len(done)
                logger.info("[%d/%d] running success=%.3f", i+1, len(tasks), success)
        except Exception as e:
            logger.warning("Task %s failed: %s", task["task_id"], e)
    return results


def summarize(results: list[dict]) -> dict:
    if not results: return {}
    n         = len(results)
    success   = sum(r["success"] for r in results) / n
    avg_reward = sum(r["reward"] for r in results if r["reward"]) / n
    avg_steps  = sum(r["total_steps"] for r in results) / n
    return {"n": n, "success_rate": success, "mean_reward": avg_reward,
            "mean_steps": avg_steps, "phase": "baseline"}


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
    for k, v in summary.items():
        logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    out = Path(cfg.get("results_dir", "data/results"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "baseline.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    wandb.log({"baseline/success_rate": summary["success_rate"],
               "baseline/mean_reward":  summary["mean_reward"],
               "baseline/mean_steps":   summary["mean_steps"]})
    wandb.finish()
    logger.info("Saved → data/results/baseline.json")


if __name__ == "__main__":
    main()
