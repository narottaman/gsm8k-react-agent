"""
scripts/train_sft_full.py
--------------------------
Phase 2b: Full SFT on Qwen3-8B — all weights updated.

Memory strategy to fit on A100 80GB:
  - bfloat16 weights:           16GB
  - gradient checkpointing:     trades compute for memory (-20GB activations)
  - adamw_8bit optimizer:       8-bit optimizer states (32GB → 8GB)
  - batch_size=1, grad_accum=16 effective batch=16, minimal activation memory
  Total: ~50GB — fits with headroom

Output: checkpoints/sft_full/  (full model weights ~16GB)

Usage:
    python scripts/train_sft_full.py --config configs/sft_full_config.yaml
"""
import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, yaml, wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("logs/train_sft_full.log")]
)
logger = logging.getLogger(__name__)


SFT_TEMPLATE = """You are a math reasoning agent. Solve problems using tools.

Problem: {question}

{{"thought": "I need to compute this step by step using code.", "action": {{"type": "tool_call", "tool_name": "code_executor", "tool_args": {{"code": "# Solve step by step\\nprint({answer})"}}}}}}
[Tool Result]
{answer}
{{"thought": "The computation confirms the answer.", "action": {{"type": "final_answer", "content": "{answer}"}}}}"""


def format_examples(tasks: list[dict]) -> list[dict]:
    examples = []
    for t in tasks:
        text = SFT_TEMPLATE.format(
            question=t["query"],
            answer=t["ground_truth"],
        )
        examples.append({"text": text})
    return examples


def load_tasks(path: str, max_samples: int = None) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: tasks.append(json.loads(line))
    return tasks[:max_samples] if max_samples else tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/sft_full_config.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    wandb.init(
        project=cfg.get("wandb_project", "gsm8k-react-agent"),
        name=cfg.get("run_name", "sft-full-run"),
        config=cfg,
        tags=["sft", "full"],
    )

    model_name  = cfg["model_name"]
    output_dir  = cfg.get("output_dir", "checkpoints/sft_full")
    max_samples = args.max_samples or cfg.get("max_train_samples", 500)
    max_length  = cfg.get("max_length", 512)

    # ── Tokenizer ──────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model — full weights in bfloat16 ───────────────────────────────
    logger.info("Loading model (full SFT): %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()   # saves ~20GB activation memory
    model.enable_input_require_grads()      # required with gradient checkpointing
    model.config.use_cache = False          # incompatible with grad checkpointing

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total trainable parameters: %.2fB", total_params / 1e9)

    # ── Data ───────────────────────────────────────────────────────────
    tasks    = load_tasks("data/gsm8k/train.jsonl", max_samples)
    examples = format_examples(tasks)
    logger.info("SFT examples: %d", len(examples))

    dataset = Dataset.from_list(examples)
    split   = dataset.train_test_split(test_size=0.05, seed=42)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=max_length, padding=False)

    train_tok = split["train"].map(tokenize, batched=True, remove_columns=["text"])
    eval_tok  = split["test"].map(tokenize,  batched=True, remove_columns=["text"])
    collator  = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── TrainingArguments ──────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_epochs", 3),
        per_device_train_batch_size=1,          # must be 1 for full 8B SFT
        gradient_accumulation_steps=16,         # effective batch = 16
        learning_rate=cfg.get("lr", 2e-5),      # lower LR for full SFT
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        eval_steps=cfg.get("eval_steps", 100),
        eval_strategy="steps",
        bf16=True,
        report_to="wandb",
        save_total_limit=1,                     # save space — only keep best
        load_best_model_at_end=True,
        dataloader_num_workers=2,
        optim="adamw_8bit",                     # 8-bit optimizer: 32GB → 8GB
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

    logger.info("Starting full SFT training...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Full SFT complete. Model saved → %s", output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()