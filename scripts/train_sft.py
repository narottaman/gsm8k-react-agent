"""
scripts/train_sft.py
--------------------
Phase 2: Supervised Fine-Tuning on GSM8K chain-of-thought solutions.
Teaches the model the correct reasoning FORMAT before RL.

Input:  GSM8K train split — question + step-by-step solution
Output: checkpoints/sft/  — fine-tuned model weights

Why SFT before RL:
  Raw Qwen3-8B doesn't know the JSON agent format we need.
  SFT teaches it the format. GRPO then optimizes the behavior.
  SFT alone = good format, mediocre answers.
  SFT + GRPO = good format + RL-optimized tool use.

Usage:
    python scripts/train_sft.py --config configs/sft_config.yaml
"""
import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, yaml, wandb
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("logs/train_sft.log")]
)
logger = logging.getLogger(__name__)


AGENT_FORMAT_PROMPT = """You are a math reasoning agent. Solve problems step by step using tools.

Problem: {question}

Solve this by:
1. Breaking it into steps
2. Using code_executor or calculator for arithmetic
3. Giving the final numeric answer

Response format (JSON):
{{"thought": "...", "action": {{"type": "tool_call", "tool_name": "code_executor", "tool_args": {{"code": "print(...)"}}}}}}\n[Tool Result]\n<result>\n{{"thought": "I have the answer.", "action": {{"type": "final_answer", "content": "{answer}"}}}}"""


def load_gsm8k_train(path: str, max_samples: int = None) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: tasks.append(json.loads(line))
    return tasks[:max_samples] if max_samples else tasks


def format_for_sft(tasks: list[dict]) -> list[dict]:
    """Convert GSM8K tasks into agent-format training examples."""
    examples = []
    for t in tasks:
        text = AGENT_FORMAT_PROMPT.format(
            question=t["query"],
            answer=t["ground_truth"]
        )
        examples.append({"text": text})
    return examples


def tokenize(examples, tokenizer, max_length):
    return tokenizer(examples["text"], truncation=True,
                     max_length=max_length, padding=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/sft_config.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    wandb.init(project=cfg.get("wandb_project", "gsm8k-react-agent"),
               name=cfg.get("run_name", "sft-run"), config=cfg, tags=["sft"])

    model_name = cfg["model_name"]
    output_dir = cfg.get("output_dir", "checkpoints/sft")
    max_samples = args.max_samples or cfg.get("max_train_samples", 500)

    logger.info("Loading tokenizer + model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")

    # Data
    tasks    = load_gsm8k_train("data/gsm8k/train.jsonl", max_samples)
    examples = format_for_sft(tasks)
    logger.info("SFT examples: %d", len(examples))

    dataset = Dataset.from_list(examples)
    split   = dataset.train_test_split(test_size=0.05, seed=42)

    max_length = cfg.get("max_length", 1024)
    train_tok  = split["train"].map(lambda x: tokenize(x, tokenizer, max_length),
                                    batched=True, remove_columns=["text"])
    eval_tok   = split["test"].map(lambda x: tokenize(x, tokenizer, max_length),
                                   batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("grad_accum_steps", 4),
        learning_rate=cfg.get("lr", 2e-5),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        eval_steps=cfg.get("eval_steps", 100),
        evaluation_strategy="steps",
        bf16=True,
        report_to="wandb",
        save_total_limit=2,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

    logger.info("Starting SFT...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("SFT complete. Model saved → %s", output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
