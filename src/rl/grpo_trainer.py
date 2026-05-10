"""
src/rl/grpo_trainer.py
----------------------
GRPO trainer applied to agent trajectories.

GRPO (Group Relative Policy Optimization):
- Same as DeepSeek-R1 / Qwen3-Coder training
- No separate value model needed
- Uses group of trajectories for same prompt → relative rewards
- Updates the LLM policy to prefer higher-reward trajectories

Reference: DeepSeekMath / GRPO paper (Shao et al., 2024)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

from src.agent.loop import AgentTrajectory, ActionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory → Training Tokens
# ---------------------------------------------------------------------------

def trajectory_to_training_text(traj: AgentTrajectory) -> str:
    """
    Convert a trajectory into a single text string for language model training.
    Format: [Problem] ... [Step] thought + action ... [Answer] final_answer
    """
    parts = [f"[Problem] {traj.query}"]
    for step in traj.steps:
        action_str = json.dumps({
            "thought": step.thought,
            "action": {
                "type":      step.action.type,
                "tool_name": step.action.tool_name,
                "tool_args": step.action.tool_args,
                "content":   step.action.content,
            }
        })
        parts.append(f"[Step {step.step_idx}] {action_str}")
        if step.observation:
            parts.append(f"[Observation] {step.observation}")
    parts.append(f"[Answer] {traj.final_answer or ''}")
    return "\n".join(parts)


def build_grpo_batch(
    trajectories: list[AgentTrajectory],
    tokenizer,
    max_length: int = 1024,
) -> dict[str, torch.Tensor]:
    """
    Tokenize trajectories + attach rewards.
    Returns dict ready for GRPO loss computation.
    """
    texts   = [trajectory_to_training_text(t) for t in trajectories]
    rewards = torch.tensor(
        [t.reward if t.reward is not None else 0.0 for t in trajectories],
        dtype=torch.float32
    )

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    return {
        "input_ids":      encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "rewards":        rewards,
    }


# ---------------------------------------------------------------------------
# GRPO Loss
# ---------------------------------------------------------------------------

def grpo_loss(
    logits:         torch.Tensor,   # [B, T, V]
    input_ids:      torch.Tensor,   # [B, T]
    attention_mask: torch.Tensor,   # [B, T]
    rewards:        torch.Tensor,   # [B]
    clip_ratio:     float = 0.2,
    kl_coeff:       float = 0.01,
) -> torch.Tensor:
    """
    GRPO loss:
      1. Compute log probs of generated tokens under current policy
      2. Normalize rewards within the group (zero mean, unit std)
      3. Policy gradient loss clipped by clip_ratio
      4. KL penalty to stay near reference (approximated by entropy)
    """
    B, T, V = logits.shape

    # Log probs for each token
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, V]
    token_log_probs = log_probs.gather(
        dim=2,
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)  # [B, T]

    # Mask padding
    token_log_probs = token_log_probs * attention_mask  # [B, T]

    # Sequence-level log prob (sum over tokens)
    seq_log_probs = token_log_probs.sum(dim=1)  # [B]

    # Normalize rewards within group (GRPO key step)
    if rewards.std() > 1e-8:
        norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        norm_rewards = rewards - rewards.mean()

    # Policy gradient loss (negative because we maximize reward)
    # Clamp ratio to avoid extreme updates (PPO-style clip)
    ratio = torch.exp(seq_log_probs - seq_log_probs.detach())
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    pg_loss = -torch.min(ratio * norm_rewards, clipped * norm_rewards).mean()

    # KL approximation via entropy (keep policy from collapsing)
    probs   = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    kl_loss = -kl_coeff * entropy

    return pg_loss + kl_loss


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    def __init__(
        self,
        model_name:  str   = "Qwen/Qwen3-8B-Instruct",
        lr:          float = 1e-5,
        clip_ratio:  float = 0.2,
        kl_coeff:    float = 0.01,
        max_length:  int   = 1024,
        output_dir:  str   = "checkpoints",
        use_wandb:   bool  = True,
    ):
        self.lr         = lr
        self.clip_ratio = clip_ratio
        self.kl_coeff   = kl_coeff
        self.max_length = max_length
        self.output_dir = Path(output_dir)
        self.use_wandb  = use_wandb
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.train()
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train_step(
        self,
        trajectories: list[AgentTrajectory],
        iteration:    int,
    ) -> dict[str, float]:
        """One GRPO update step on a batch of trajectories."""
        if not trajectories:
            return {}

        batch = build_grpo_batch(trajectories, self.tokenizer, self.max_length)

        device = next(self.model.parameters()).device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        rewards        = batch["rewards"].to(device)

        self.optimizer.zero_grad()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits  # [B, T, V]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_ids    = input_ids[:, 1:].contiguous()
        shift_mask   = attention_mask[:, 1:].contiguous()

        loss = grpo_loss(
            logits=shift_logits,
            input_ids=shift_ids,
            attention_mask=shift_mask,
            rewards=rewards,
            clip_ratio=self.clip_ratio,
            kl_coeff=self.kl_coeff,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Metrics
        metrics = {
            "train/loss":          loss.item(),
            "train/mean_reward":   rewards.mean().item(),
            "train/max_reward":    rewards.max().item(),
            "train/min_reward":    rewards.min().item(),
            "train/success_rate":  sum(t.success for t in trajectories) / len(trajectories),
            "train/batch_size":    len(trajectories),
            "train/iteration":     iteration,
        }

        if self.use_wandb:
            wandb.log(metrics, step=iteration)

        logger.info(
            "[iter %d] loss=%.4f | mean_reward=%.3f | success=%.2f",
            iteration, metrics["train/loss"],
            metrics["train/mean_reward"], metrics["train/success_rate"]
        )

        return metrics

    def save(self, iteration: int) -> None:
        path = self.output_dir / f"checkpoint-iter{iteration}"
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Saved checkpoint: %s", path)