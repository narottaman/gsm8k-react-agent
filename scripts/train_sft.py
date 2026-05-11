"""
scripts/train_sft.py
--------------------
Phase 2: SFT with LoRA on Qwen3-8B.

Why LoRA:
  - Full fine-tune 8B needs ~60-80GB for gradients alone → OOM on A100 80GB
  - LoRA freezes base weights, trains only small adapter matrices
  - Memory: ~20GB total (weights + adapters + gradients) → fits easily
  - Quality: LoRA SFT → same accuracy as full fine-tune for format learning

Output: checkpoints/sft/  (adapter weights only, ~100MB vs 16GB full model)

Usage:
    python scripts/train_sft.py --config configs/sft_config.yaml
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
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("logs/train_sft.log")]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent-format SFT template
# Teaches model: JSON format + use tools + give numeric answer
# ---------------------------------------------------------------------------

SFT_TEMPLATE = """You are a math reasoning agent. Solve problems using tools.

Problem: {question}

{{"thought": "I need to compute this step by step using code.", "action": {{"type": "tool_call", "tool_name": "code_executor", "tool_args": {{"code": "# Solve: {question_short}\\nprint({answer})"}}}}}}
[Tool Result]
{answer}
{{"thought": "The computation confirms the answer.", "action": {{"type": "final_answer", "content": "{answer}"}}}}"""


def format_examples(tasks: list[dict]) -> list[dict]:
    examples = []
    for t in tasks:
        question_short = t["query"][:60].replace('"', "'")
        text = SFT_TEMPLATE.format(
            question=t["query"],
            question_short=question_short,
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
    parser.add_argument("--config",      default="configs/sft_config.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    wandb.init(
        project=cfg.get("wandb_project", "gsm8k-react-agent"),
        name=cfg.get("run_name", "sft-lora-run"),
        config=cfg,
        tags=["sft", "lora"],
    )

    model_name  = cfg["model_name"]
    output_dir  = cfg.get("output_dir", "checkpoints/sft")
    max_samples = args.max_samples or cfg.get("max_train_samples", 500)
    max_length  = cfg.get("max_length", 512)

    # ── Tokenizer ──────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model + LoRA ───────────────────────────────────────────────────
    logger.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    # Required for gradient_checkpointing + LoRA to work together
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── Data ───────────────────────────────────────────────────────────
    tasks    = load_tasks("data/gsm8k/train.jsonl", max_samples)
    examples = format_examples(tasks)
    logger.info("SFT examples: %d", len(examples))

    dataset = Dataset.from_list(examples)
    split   = dataset.train_test_split(test_size=0.05, seed=42)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    train_tok = split["train"].map(tokenize, batched=True, remove_columns=["text"])
    eval_tok  = split["test"].map(tokenize,  batched=True, remove_columns=["text"])
    collator  = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── TrainingArguments (fixed: eval_strategy not evaluation_strategy) ─
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("grad_accum_steps", 4),
        learning_rate=cfg.get("lr", 2e-4),        # higher LR for LoRA
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        eval_steps=cfg.get("eval_steps", 100),
        eval_strategy="steps",                     # ← fixed (was evaluation_strategy)
        bf16=True,
        report_to="wandb",
        save_total_limit=2,
        load_best_model_at_end=True,
        dataloader_num_workers=2,
        optim="adamw_torch_fused",                 # faster on A100
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

    logger.info("Starting LoRA SFT training...")
    trainer.train()

    # Save adapter weights
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("LoRA SFT complete. Adapter saved → %s", output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()