"""
scripts/train_grpo_full.py
--------------------------
Phase 3b: GRPO with full weight updates on 2x A100 GPUs.

Architecture:
  - vLLM rollout:  1 GPU (tensor_parallel_size=1, pinned to cuda:0 via env)
  - Policy update: 2 GPUs via device_map="auto" (spreads 8B + optimizer)

Memory per GPU (~80GB each):
  GPU 0: vLLM(16GB) + half policy(8GB) + half optimizer(16GB) = ~40GB
  GPU 1: half policy(8GB) + half optimizer(16GB) + activations(8GB) = ~32GB

Usage:
    python scripts/train_grpo_full.py --config configs/grpo_full_config.yaml
"""
import argparse, gc, json, logging, os, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, yaml, wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

from src.agent.loop import ReActAgent, VLLMBackend, MockLLM, AgentTrajectory, ActionType
from src.agent.tools import build_tool_registry
from src.agent.reward import compute_reward
from src.rl.trajectory import TrajectoryBuffer, trajectory_to_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("logs/train_grpo_full.log")]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_tasks(path: str, max_samples: int = None) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: tasks.append(json.loads(line))
    return tasks[:max_samples] if max_samples else tasks


# ---------------------------------------------------------------------------
# GRPO Loss
# ---------------------------------------------------------------------------

def grpo_loss(logits, input_ids, attention_mask, rewards,
              clip_ratio=0.2, kl_coeff=0.01):
    log_probs  = torch.log_softmax(logits, dim=-1)
    token_lp   = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
    token_lp   = token_lp * attention_mask
    seq_lp     = token_lp.sum(dim=1)

    if rewards.std() > 1e-8:
        norm_r = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        norm_r = rewards - rewards.mean()

    ratio   = torch.exp(seq_lp - seq_lp.detach())
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    pg_loss = -torch.min(ratio * norm_r, clipped * norm_r).mean()

    probs   = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return pg_loss - kl_coeff * entropy


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout(agent: ReActAgent, tasks: list[dict]) -> list[AgentTrajectory]:
    trajs = []
    for task in tasks:
        try:
            traj = agent.run(task_id=task["task_id"], query=task["query"],
                             ground_truth=task["ground_truth"])
            compute_reward(traj)
            trajs.append(traj)
        except Exception as e:
            logger.warning("Rollout failed %s: %s", task["task_id"], e)
    return trajs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/grpo_full_config.yaml")
    parser.add_argument("--mock",        action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sft_ckpt    = Path(cfg.get("sft_checkpoint", "checkpoints/sft_full"))
    start_model = str(sft_ckpt) if sft_ckpt.exists() else cfg["model_name"]
    logger.info("Starting from: %s", start_model)

    wandb.init(project=cfg.get("wandb_project", "gsm8k-react-agent"),
               name=cfg.get("run_name", "grpo-full-run"),
               config=cfg, tags=["grpo", "full", "2gpu"])

    max_samples = args.max_samples or cfg.get("max_train_samples", 500)
    train_tasks = load_tasks("data/gsm8k/train.jsonl", max_samples)
    eval_tasks  = load_tasks("data/gsm8k/test.jsonl",  cfg.get("max_eval_samples", 100))
    logger.info("Train: %d | Eval: %d", len(train_tasks), len(eval_tasks))

    n_iter     = cfg.get("n_iterations", 20)
    batch_size = cfg.get("batch_size", 16)
    save_every = cfg.get("save_every", 5)
    eval_every = cfg.get("eval_every", 5)
    max_length = cfg.get("max_length", 512)
    rl_dir     = Path(cfg.get("output_dir", "checkpoints")) / "rl_full"
    rl_dir.mkdir(parents=True, exist_ok=True)

    # ── vLLM rollout model — pin to GPU 0 only ─────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tools = build_tool_registry()
    if not args.mock:
        rollout_llm = VLLMBackend(
            model_name=start_model,
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 512),
        )
    else:
        rollout_llm = MockLLM()
    agent = ReActAgent(llm=rollout_llm, tools=tools, max_steps=cfg.get("max_steps", 8))

    # ── Policy model — spread across both GPUs ─────────────────────────
    if not args.mock:
        # Restore both GPUs for policy
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        logger.info("Loading full policy model across 2 GPUs...")
        tokenizer = AutoTokenizer.from_pretrained(start_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        policy = AutoModelForCausalLM.from_pretrained(
            start_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",          # spreads layers across GPU 0 + GPU 1
        )
        policy.gradient_checkpointing_enable()
        policy.enable_input_require_grads()
        policy.config.use_cache = False
        policy.train()

        total = sum(p.numel() for p in policy.parameters())
        logger.info("Full policy: %.2fB parameters", total / 1e9)

        optimizer = AdamW(policy.parameters(), lr=cfg.get("lr", 1e-5))

    buffer = TrajectoryBuffer("data/trajectories/grpo_full_buffer.jsonl")
    logger.info("Starting full-weight GRPO: %d iterations", n_iter)

    for iteration in range(n_iter):

        # ── ROLLOUT ────────────────────────────────────────────────────
        batch = random.sample(train_tasks, min(batch_size, len(train_tasks)))
        trajs = rollout(agent, batch)
        buffer.clear()
        for t in trajs: buffer.add(t)

        mr = buffer.mean_reward()
        sr = buffer.success_rate()
        logger.info("[iter %02d] rollout | reward=%.3f | success=%.2f | n=%d",
                    iteration, mr, sr, len(trajs))
        wandb.log({"rollout/mean_reward": mr, "rollout/success_rate": sr}, step=iteration)

        # ── GRPO UPDATE ────────────────────────────────────────────────
        if not args.mock and trajs:
            # Free vLLM before update to reclaim GPU 0 memory
            del agent.llm.llm
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("[iter %02d] vLLM freed, running full GRPO update", iteration)

            texts   = [trajectory_to_text(t) for t in trajs]
            rewards = torch.tensor([t.reward or 0.0 for t in trajs], dtype=torch.float32)

            encoded = tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length)

            # Policy is spread across both GPUs — input goes to first device
            first_device = next(policy.parameters()).device
            ids     = encoded["input_ids"].to(first_device)
            mask    = encoded["attention_mask"].to(first_device)
            rewards = rewards.to(first_device)

            optimizer.zero_grad()
            out  = policy(input_ids=ids, attention_mask=mask)
            loss = grpo_loss(
                out.logits[:, :-1].contiguous(),
                ids[:, 1:].contiguous(),
                mask[:, 1:].contiguous(),
                rewards,
                clip_ratio=cfg.get("clip_ratio", 0.2),
                kl_coeff=cfg.get("kl_coeff", 0.01),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            logger.info("[iter %02d] loss=%.4f | reward=%.3f",
                        iteration, loss.item(), rewards.mean().item())
            wandb.log({"train/loss": loss.item(),
                       "train/mean_reward": rewards.mean().item()}, step=iteration)

            # Reload vLLM for next iteration
            if iteration < n_iter - 1:
                del out
                gc.collect()
                torch.cuda.empty_cache()
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                rollout_llm = VLLMBackend(
                    model_name=start_model,
                    temperature=cfg.get("temperature", 0.7),
                    max_tokens=cfg.get("max_tokens", 512))
                agent = ReActAgent(llm=rollout_llm, tools=tools,
                                   max_steps=cfg.get("max_steps", 8))
                logger.info("[iter %02d] vLLM reloaded", iteration)

        # ── EVAL ───────────────────────────────────────────────────────
        if iteration % eval_every == 0:
            sample   = random.sample(eval_tasks, min(20, len(eval_tasks)))
            eval_t   = rollout(agent, sample)
            e_success = sum(t.success for t in eval_t) / max(len(eval_t), 1)
            e_reward  = sum((t.reward or 0) for t in eval_t) / max(len(eval_t), 1)
            logger.info("[iter %02d] eval | success=%.3f | reward=%.3f",
                        iteration, e_success, e_reward)
            wandb.log({"eval/success_rate": e_success,
                       "eval/mean_reward":  e_reward}, step=iteration)

        # ── SAVE ───────────────────────────────────────────────────────
        if not args.mock and iteration % save_every == 0 and iteration > 0:
            ckpt = rl_dir / f"iter_{iteration:03d}"
            policy.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            buffer.save()
            logger.info("Checkpoint saved → %s", ckpt)

    if not args.mock:
        final = rl_dir / "final"
        policy.save_pretrained(final)
        tokenizer.save_pretrained(final)
        logger.info("Final full GRPO model saved → %s", final)

    wandb.finish()
    logger.info("Full GRPO training complete.")


if __name__ == "__main__":
    main()
