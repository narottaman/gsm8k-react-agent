"""
scripts/train.py
----------------
Full GRPO training run on Sol A100.

What this does:
  1. Load GSM8K dataset (auto-download from HuggingFace)
  2. Load Qwen3-8B via vLLM for fast rollout generation
  3. For each iteration:
       a. Sample N problems from GSM8K
       b. Run ReActAgent on each → collect trajectories
       c. Compute rewards (answer_correct + efficiency + format)
       d. GRPO update on the batch
       e. Log to W&B
  4. Save checkpoints every K iterations

Run on Sol:
  sbatch configs/slurm.sh
  OR: python scripts/train.py --config configs/grpo_config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import wandb
import yaml

# make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.loop import ReActAgent, VLLMBackend, MockLLM
from src.agent.tools import build_tool_registry
from src.agent.reward import compute_reward
from src.rl.grpo_trainer import GRPOTrainer
from src.rl.trajectory import TrajectoryBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train.log"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GSM8K loader
# ---------------------------------------------------------------------------

def load_gsm8k(split: str = "train", max_samples: int = None):
    """
    Load GSM8K from HuggingFace datasets.
    Auto-downloads on first run. Cached after.
    Returns list of {"task_id", "query", "ground_truth"}
    """
    from datasets import load_dataset
    logger.info("Loading GSM8K %s split...", split)
    ds = load_dataset("openai/gsm8k", "main", split=split)

    tasks = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        # GSM8K ground truth is after #### in the answer field
        answer_text = row["answer"]
        if "####" in answer_text:
            gt = answer_text.split("####")[-1].strip()
        else:
            gt = answer_text.strip()
        tasks.append({
            "task_id":      f"gsm8k_{split}_{i:05d}",
            "query":        row["question"],
            "ground_truth": gt,
        })

    logger.info("Loaded %d tasks from GSM8K %s", len(tasks), split)
    return tasks


# ---------------------------------------------------------------------------
# Rollout — run agent on a batch of tasks
# ---------------------------------------------------------------------------

def run_rollout(agent: ReActAgent, tasks: list[dict]) -> list:
    trajectories = []
    for task in tasks:
        try:
            traj = agent.run(
                task_id=task["task_id"],
                query=task["query"],
                ground_truth=task["ground_truth"],
            )
            compute_reward(traj)
            trajectories.append(traj)
        except Exception as e:
            logger.warning("Task %s failed: %s", task["task_id"], e)
    return trajectories


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/grpo_config.yaml")
    parser.add_argument("--mock",         action="store_true", help="Use MockLLM (no GPU)")
    parser.add_argument("--max_samples",  type=int, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # W&B
    wandb.init(
        project=cfg.get("wandb_project", "gsm8k-react-agent"),
        name=cfg.get("run_name", "grpo-run-01"),
        config=cfg,
    )

    # Load data
    train_tasks = load_gsm8k("train", max_samples=args.max_samples or cfg.get("max_train_samples"))
    eval_tasks  = load_gsm8k("test",  max_samples=cfg.get("max_eval_samples", 100))

    # Build agent
    if args.mock:
        logger.warning("Using MockLLM — for testing only, no real training.")
        llm = MockLLM()
    else:
        llm = VLLMBackend(
            model_name=cfg.get("model_name", "Qwen/Qwen3-8B-Instruct"),
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 512),
        )

    tools = build_tool_registry()
    agent = ReActAgent(llm=llm, tools=tools, max_steps=cfg.get("max_steps", 8))

    # GRPO trainer (loads model separately for weight updates)
    if not args.mock:
        trainer = GRPOTrainer(
            model_name=cfg.get("model_name", "Qwen/Qwen3-8B-Instruct"),
            lr=cfg.get("lr", 1e-5),
            clip_ratio=cfg.get("clip_ratio", 0.2),
            kl_coeff=cfg.get("kl_coeff", 0.01),
            max_length=cfg.get("max_length", 1024),
            output_dir=cfg.get("output_dir", "checkpoints"),
            use_wandb=True,
        )

    buffer = TrajectoryBuffer(save_path="data/trajectories/buffer.jsonl")

    import random
    n_iterations  = cfg.get("n_iterations", 20)
    batch_size    = cfg.get("batch_size", 16)
    save_every    = cfg.get("save_every", 5)
    eval_every    = cfg.get("eval_every", 5)

    logger.info("Starting GRPO training: %d iterations, batch_size=%d", n_iterations, batch_size)

    for iteration in range(n_iterations):
        # ── ROLLOUT ────────────────────────────────────────────────────
        batch_tasks = random.sample(train_tasks, min(batch_size, len(train_tasks)))
        trajectories = run_rollout(agent, batch_tasks)
        buffer.clear()
        for t in trajectories:
            buffer.add(t)

        logger.info(
            "[iter %d] rollout done | mean_reward=%.3f | success=%.2f",
            iteration, buffer.mean_reward(), buffer.success_rate()
        )

        wandb.log({
            "rollout/mean_reward":  buffer.mean_reward(),
            "rollout/success_rate": buffer.success_rate(),
            "rollout/n_trajs":      len(trajectories),
        }, step=iteration)

        # ── GRPO UPDATE ────────────────────────────────────────────────
        if not args.mock:
            trainer.train_step(trajectories, iteration=iteration)

        # ── EVAL ───────────────────────────────────────────────────────
        if iteration % eval_every == 0:
            eval_sample  = random.sample(eval_tasks, min(20, len(eval_tasks)))
            eval_trajs   = run_rollout(agent, eval_sample)
            eval_success = sum(t.success for t in eval_trajs) / len(eval_trajs)
            eval_reward  = sum(t.reward for t in eval_trajs if t.reward) / len(eval_trajs)
            logger.info("[eval iter %d] success=%.2f | reward=%.3f", iteration, eval_success, eval_reward)
            wandb.log({
                "eval/success_rate": eval_success,
                "eval/mean_reward":  eval_reward,
            }, step=iteration)

        # ── SAVE ───────────────────────────────────────────────────────
        if not args.mock and iteration % save_every == 0 and iteration > 0:
            trainer.save(iteration)
            buffer.save()

    logger.info("Training complete.")
    wandb.finish()


if __name__ == "__main__":
    main()