"""
scripts/train_grpo.py
---------------------
Phase 3: GRPO RL training on top of SFT model.
Teaches the agent WHEN and HOW to use tools via reward signals.

Reward = answer_correct(0.5) + tool_efficiency(0.3) + format_valid(0.2)

Flow per iteration:
  1. Sample N GSM8K problems
  2. Run SFT model as agent → collect trajectories
  3. Compute rewards per trajectory
  4. GRPO policy update
  5. Log to W&B

Usage:
    python scripts/train_grpo.py --config configs/grpo_config.yaml
"""
import argparse, json, logging, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, yaml, wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

from src.agent.loop import ReActAgent, VLLMBackend, MockLLM, AgentTrajectory, ActionType
from src.agent.tools import build_tool_registry
from src.agent.reward import compute_reward
from src.rl.trajectory import TrajectoryBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("logs/train_grpo.log")]
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

def trajectory_to_text(traj: AgentTrajectory) -> str:
    parts = [f"[Problem] {traj.query}"]
    for step in traj.steps:
        parts.append(json.dumps({"thought": step.thought,
                                  "action": {"type": step.action.type,
                                             "tool_name": step.action.tool_name,
                                             "tool_args": step.action.tool_args,
                                             "content": step.action.content}}))
        if step.observation:
            parts.append(f"[Observation] {step.observation}")
    parts.append(f"[Answer] {traj.final_answer or ''}")
    return "\n".join(parts)


def grpo_loss(logits, input_ids, attention_mask, rewards,
              clip_ratio=0.2, kl_coeff=0.01):
    log_probs       = torch.log_softmax(logits, dim=-1)
    token_lp        = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
    token_lp        = token_lp * attention_mask
    seq_lp          = token_lp.sum(dim=1)

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
            logger.warning("Rollout failed for %s: %s", task["task_id"], e)
    return trajs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/grpo_config.yaml")
    parser.add_argument("--mock",        action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Determine which model to start from
    sft_ckpt = Path(cfg.get("sft_checkpoint", "checkpoints/sft"))
    if sft_ckpt.exists():
        start_model = str(sft_ckpt)
        logger.info("Starting from SFT checkpoint: %s", start_model)
    else:
        start_model = cfg["model_name"]
        logger.warning("No SFT checkpoint found — starting from base model: %s", start_model)

    wandb.init(project=cfg.get("wandb_project", "gsm8k-react-agent"),
               name=cfg.get("run_name", "grpo-run"), config=cfg, tags=["grpo", "rl"])

    # Data
    max_samples = args.max_samples or cfg.get("max_train_samples", 500)
    train_tasks = load_tasks("data/gsm8k/train.jsonl", max_samples)
    eval_tasks  = load_tasks("data/gsm8k/test.jsonl",  cfg.get("max_eval_samples", 100))
    logger.info("Train: %d | Eval: %d", len(train_tasks), len(eval_tasks))

    # Agent (vLLM for fast rollouts)
    rollout_llm = MockLLM() if args.mock else VLLMBackend(
        model_name=start_model,
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens", 512))
    tools = build_tool_registry()
    agent = ReActAgent(llm=rollout_llm, tools=tools, max_steps=cfg.get("max_steps", 8))

    # Separate model for gradient updates — 4-bit to save memory
    # vLLM already holds ~16GB, so policy model must be quantized
    if not args.mock:
        logger.info("Loading policy model (4-bit) for GRPO updates...")
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        tokenizer = AutoTokenizer.from_pretrained(start_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        policy = AutoModelForCausalLM.from_pretrained(
            start_model,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # LoRA adapter on top of 4-bit base for gradient updates
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        policy = get_peft_model(policy, lora_cfg)
        policy.enable_input_require_grads()
        policy.train()
        policy.print_trainable_parameters()
        optimizer = AdamW(
            [p for p in policy.parameters() if p.requires_grad],
            lr=cfg.get("lr", 1e-5)
        )

    buffer       = TrajectoryBuffer("data/trajectories/grpo_buffer.jsonl")
    n_iter       = cfg.get("n_iterations", 20)
    batch_size   = cfg.get("batch_size", 16)
    save_every   = cfg.get("save_every", 5)
    eval_every   = cfg.get("eval_every", 5)
    max_length   = cfg.get("max_length", 1024)
    output_dir   = Path(cfg.get("output_dir", "checkpoints"))
    rl_dir       = output_dir / "rl"
    rl_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting GRPO: %d iterations, batch=%d", n_iter, batch_size)

    for iteration in range(n_iter):
        # ── ROLLOUT (vLLM) ─────────────────────────────────────────────
        batch  = random.sample(train_tasks, min(batch_size, len(train_tasks)))
        trajs  = rollout(agent, batch)
        buffer.clear()
        for t in trajs: buffer.add(t)

        mr = buffer.mean_reward()
        sr = buffer.success_rate()
        logger.info("[iter %02d] rollout | mean_reward=%.3f | success=%.2f | n=%d",
                    iteration, mr, sr, len(trajs))
        wandb.log({"rollout/mean_reward": mr, "rollout/success_rate": sr,
                   "rollout/n": len(trajs)}, step=iteration)

        # ── GRPO UPDATE ────────────────────────────────────────────────
        if not args.mock and trajs:
            # Free vLLM GPU memory before gradient update
            import gc
            del agent.llm.llm
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("[iter %02d] vLLM freed, running GRPO update", iteration)

            texts   = [trajectory_to_text(t) for t in trajs]
            rewards = torch.tensor([t.reward or 0.0 for t in trajs], dtype=torch.float32)

            encoded = tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length)
            device  = next(p for p in policy.parameters() if p.requires_grad).device
            ids     = encoded["input_ids"].to(device)
            mask    = encoded["attention_mask"].to(device)
            rewards = rewards.to(device)

            optimizer.zero_grad()
            out    = policy(input_ids=ids, attention_mask=mask)
            loss   = grpo_loss(out.logits[:, :-1], ids[:, 1:], mask[:, 1:], rewards,
                               clip_ratio=cfg.get("clip_ratio", 0.2),
                               kl_coeff=cfg.get("kl_coeff", 0.01))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            logger.info("[iter %02d] grpo loss=%.4f", iteration, loss.item())
            wandb.log({"train/loss": loss.item(), "train/mean_reward": rewards.mean().item()},
                      step=iteration)

            # Reload vLLM for next rollout (except last iteration)
            if iteration < n_iter - 1:
                del out
                torch.cuda.empty_cache()
                gc.collect()
                rollout_llm = VLLMBackend(
                    model_name=start_model,
                    temperature=cfg.get("temperature", 0.7),
                    max_tokens=cfg.get("max_tokens", 512))
                agent = ReActAgent(llm=rollout_llm, tools=tools,
                                   max_steps=cfg.get("max_steps", 8))
                logger.info("[iter %02d] vLLM reloaded for next rollout", iteration)

        # ── EVAL ───────────────────────────────────────────────────────
        if iteration % eval_every == 0:
            sample    = random.sample(eval_tasks, min(20, len(eval_tasks)))
            eval_t    = rollout(agent, sample)
            e_success = sum(t.success for t in eval_t) / max(len(eval_t), 1)
            e_reward  = sum((t.reward or 0) for t in eval_t) / max(len(eval_t), 1)
            logger.info("[iter %02d] eval | success=%.3f | reward=%.3f", iteration, e_success, e_reward)
            wandb.log({"eval/success_rate": e_success, "eval/mean_reward": e_reward}, step=iteration)

        # ── SAVE ───────────────────────────────────────────────────────
        if not args.mock and iteration % save_every == 0 and iteration > 0:
            ckpt = rl_dir / f"iter_{iteration:03d}"
            # Save LoRA adapter only (small, ~100MB)
            policy.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            buffer.save()
            logger.info("Checkpoint saved → %s", ckpt)

    # Final save
    if not args.mock:
        final = rl_dir / "final"
        policy.save_pretrained(final)
        tokenizer.save_pretrained(final)
        logger.info("Final RL adapter saved → %s", final)
        logger.info("To use: load base model + merge this adapter")

    wandb.finish()
    logger.info("GRPO training complete.")


if __name__ == "__main__":
    main()