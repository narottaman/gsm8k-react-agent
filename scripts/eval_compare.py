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


def is_lora_checkpoint(path: str) -> bool:
    """Check if a path is a LoRA adapter (has adapter_config.json but no config.json)."""
    from pathlib import Path as P
    p = P(path)
    return (p / "adapter_config.json").exists() and not (p / "config.json").exists()


def merge_lora_for_inference(adapter_path: str, base_model: str, merged_path: str) -> str:
    """Merge LoRA adapter into base model weights for vLLM inference."""
    from pathlib import Path as P
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    merged = P(merged_path)
    if (merged / "config.json").exists():
        logger.info("Merged model already exists at %s", merged_path)
        return merged_path

    logger.info("Merging LoRA adapter %s + base %s → %s", adapter_path, base_model, merged_path)
    merged.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()   # fuses adapter into base weights

    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    logger.info("Merge complete → %s", merged_path)
    return merged_path


def eval_model(model_name: str, tasks: list[dict], cfg: dict,
               mock: bool = False, label: str = "model") -> dict:
    logger.info("Evaluating: %s (%s)", label, model_name)

    # LoRA checkpoints need merging before vLLM can load them
    if not mock and is_lora_checkpoint(model_name):
        base_model  = cfg.get("model_name", "Qwen/Qwen3-8B")
        merged_path = model_name.rstrip("/") + "_merged"
        model_name  = merge_lora_for_inference(model_name, base_model, merged_path)

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
    print("\\n" + "="*70)
    print(f"{'Phase':<15} {'Model':<30} {'Accuracy':>9} {'Reward':>8} {'Steps':>7}")
    print("-"*70)
    for entry in comparisons:
        # handle both {summary: ...} and flat dict formats
        s = entry.get("summary", entry)
        label    = s.get("label", s.get("phase", "unknown"))
        model    = s.get("model", "—")[:30]
        accuracy = s.get("accuracy", s.get("success_rate", 0))
        reward   = s.get("mean_reward", 0) or 0
        steps    = s.get("mean_steps", 0) or 0
        print(f"{label:<15} {model:<30} "
              f"{accuracy:>8.1%} {reward:>8.3f} {steps:>7.1f}")
    print("="*70 + "\\n")


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

    # Helper to eval a checkpoint if it exists
    def try_eval(ckpt_key: str, default: str, label: str, out_file: str):
        ckpt = Path(cfg.get(ckpt_key, default))
        if ckpt.exists():
            r = eval_model(str(ckpt), tasks, cfg, args.mock, label=label)
            with open(results_dir / out_file, "w") as f:
                json.dump(r, f, indent=2)
            comparisons.append(r)
        else:
            logger.warning("Checkpoint not found: %s — skipping", ckpt)

    try_eval("sft_checkpoint",      "checkpoints/sft",           "lora_sft",   "sft_lora.json")
    try_eval("sft_full_checkpoint",  "checkpoints/sft_full",      "full_sft",   "sft_full.json")
    try_eval("rl_checkpoint",        "checkpoints/rl/final",      "lora_grpo",  "rl_lora.json")
    try_eval("rl_full_checkpoint",   "checkpoints/rl_full/final", "full_grpo",  "rl_full.json")

    # ── Print comparison table ─────────────────────────────────────────
    print_table(comparisons)

    # ── Log to W&B ─────────────────────────────────────────────────────
    for c in comparisons:
        s = c["summary"]
        s = entry.get("summary", entry)
        label = s.get("label", s.get("phase", "unknown"))
        wandb.log({f"{label}/accuracy":   s.get("accuracy", s.get("success_rate", 0)),
                   f"{label}/mean_reward": s.get("mean_reward", 0),
                   f"{label}/mean_steps":  s.get("mean_steps", 0)})

    # Save full comparison
    with open(results_dir / "comparison.json", "w") as f:
        json.dump([c["summary"] for c in comparisons], f, indent=2)

    wandb.finish()
    logger.info("Comparison saved → data/results/comparison.json")


if __name__ == "__main__":
    main()