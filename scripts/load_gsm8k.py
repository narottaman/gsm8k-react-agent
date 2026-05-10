"""
scripts/load_gsm8k.py
---------------------
Downloads + caches GSM8K from HuggingFace.
Run this FIRST on Sol to pre-download before training.

Usage:
    python scripts/load_gsm8k.py
"""
import sys, json, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_gsm8k(split: str, max_samples: int = None) -> list[dict]:
    from datasets import load_dataset
    logger.info("Loading GSM8K split=%s ...", split)
    ds = load_dataset("openai/gsm8k", "main", split=split)
    tasks = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        answer_text = row["answer"]
        gt = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text.strip()
        tasks.append({"task_id": f"gsm8k_{split}_{i:05d}",
                      "query":   row["question"],
                      "ground_truth": gt})
    logger.info("Loaded %d tasks", len(tasks))
    return tasks


if __name__ == "__main__":
    out = Path("data/gsm8k")
    out.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        tasks = load_gsm8k(split)
        path  = out / f"{split}.jsonl"
        with open(path, "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")
        logger.info("Saved %d %s tasks → %s", len(tasks), split, path)

    logger.info("Done. GSM8K ready in data/gsm8k/")
