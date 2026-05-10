"""
scripts/eval_compare.py
-----------------------
Phase 4: Compare Baseline vs SFT vs RL on GSM8K test set.
Loads saved results + optionally re-runs each model.
Produces a clean comparison table + W&B summary.

Usage:
    python scripts/eval_compare.py --config configs/eval_config.yaml
    python scripts/eval_compare.py --config configs/eval_config.yaml --run_all
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
              logging.FileHandler("logs/eval_compare.log")]
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


def eval_model(model_name: str, tasks: list[dict], cfg: dict,
               mock: bool = False, label: str = "model") -> dict:
    logger.info("Evaluating: %s (%s)", label, model_name)
    llm   = MockLLM() if mock else VLLMBackend(
                model_name=model_name,
                temperature=cfg.get("temperature", 0.0),
                max_tokens=cfg.get("max_tokens", 512))
    tools = build_tool_registry()
    agent = ReActAgent(llm=llm, tools=tools, max_steps=cfg.get("max_steps", 8))

    results = []
    for i, task in enumerate(tasks):
        try:
            traj = agent.run(task_id=task["task_id"], query=task["query"],
                             ground_truth=task["ground_truth"])
            compute_reward(traj)
            results.append({"task_id": traj.task_id, "success": traj.success,
                             "reward": traj.reward, "total_steps": traj.total_steps,
                             "predicted": traj.final_answer,
                             "ground_truth": traj.ground_truth})
            if (i + 1) % 20 == 0:
                sr = sum(r["success"] for r in results) / len(results)
                logger.info("  [%s] %d/%d | success=%.3f", label, i+1, len(tasks), sr)
        except Exception as e:
            logger.warning("  Task %s failed: %s", task["task_id"], e)

    n          = len(results)
    success    = sum(r["success"] for r in results) / n if n else 0
    avg_reward = sum(r["reward"] for r in results if r["reward"]) / n if n else 0
    avg_steps  = sum(r["total_steps"] for r in results) / n if n else 0

    summary = {"label": label, "model": model_name, "n": n,
                "success_rate": success, "mean_reward": avg_reward, "mean_steps": avg_steps}
    logger.info("[%s] success=%.3f | reward=%.3f | steps=%.1f",
                label, success, avg_reward, avg_steps)
    return {"summary": summary, "results": results}


def print_table(comparisons: list[dict]) -> None:
    print("\n" + "="*70)
    print(f"{'Phase':<15} {'Model':<30} {'Success':>8} {'Reward':>8} {'Steps':>7}")
    print("-"*70)
    for c in comparisons:
        s = c["summary"]
        print(f"{s['label']:<15} {s['model']:<30} "
              f"{s['success_rate']:>7.1%} {s['mean_reward']:>8.3f} {s['mean_steps']:>7.1f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/eval_config.yaml")
    parser.add_argument("--mock",        action="store_true")
    parser.add_argument("--run_all",     action="store_true",
                        help="Re-run all models instead of loading saved results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    wandb.init(project=cfg.get("wandb_project", "gsm8k-react-agent"),
               name="eval-comparison", tags=["eval", "comparison"])

    results_dir = Path(cfg.get("results_dir", "data/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks("data/gsm8k/test.jsonl",
                       args.max_samples or cfg.get("max_eval_samples", 200))
    logger.info("Comparing on %d test tasks", len(tasks))

    comparisons = []

    # ── Load existing baseline if available ────────────────────────────
    baseline_path = results_dir / "baseline.json"
    if baseline_path.exists() and not args.run_all:
        logger.info("Loading saved baseline results...")
        with open(baseline_path) as f:
            saved = json.load(f)
        comparisons.append(saved)
    else:
        r = eval_model(cfg["model_name"], tasks, cfg, args.mock, label="baseline")
        with open(results_dir / "baseline.json", "w") as f:
            json.dump(r, f, indent=2)
        comparisons.append(r)

    # ── SFT model ──────────────────────────────────────────────────────
    sft_ckpt = Path(cfg.get("sft_checkpoint", "checkpoints/sft"))
    if sft_ckpt.exists():
        r = eval_model(str(sft_ckpt), tasks, cfg, args.mock, label="sft")
        with open(results_dir / "sft.json", "w") as f:
            json.dump(r, f, indent=2)
        comparisons.append(r)
    else:
        logger.warning("SFT checkpoint not found: %s — skipping", sft_ckpt)

    # ── RL model ───────────────────────────────────────────────────────
    rl_ckpt = Path(cfg.get("rl_checkpoint", "checkpoints/rl/final"))
    if not rl_ckpt.exists():
        rl_ckpt = Path("checkpoints/rl/final")
    if rl_ckpt.exists():
        r = eval_model(str(rl_ckpt), tasks, cfg, args.mock, label="rl_grpo")
        with open(results_dir / "rl.json", "w") as f:
            json.dump(r, f, indent=2)
        comparisons.append(r)
    else:
        logger.warning("RL checkpoint not found: %s — skipping", rl_ckpt)

    # ── Print comparison table ─────────────────────────────────────────
    print_table(comparisons)

    # ── Log to W&B ─────────────────────────────────────────────────────
    for c in comparisons:
        s = c["summary"]
        wandb.log({f"{s['label']}/success_rate": s["success_rate"],
                   f"{s['label']}/mean_reward":  s["mean_reward"],
                   f"{s['label']}/mean_steps":   s["mean_steps"]})

    # Save full comparison
    with open(results_dir / "comparison.json", "w") as f:
        json.dump([c["summary"] for c in comparisons], f, indent=2)

    wandb.finish()
    logger.info("Comparison saved → data/results/comparison.json")


if __name__ == "__main__":
    main()
