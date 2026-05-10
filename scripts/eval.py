"""
scripts/eval.py
---------------
Quick single-model eval — run any checkpoint on GSM8K test set.
Useful for spot-checking during development without a full SLURM job.

Usage:
    python scripts/eval.py --model Qwen/Qwen3-8B-Instruct
    python scripts/eval.py --model checkpoints/sft --max_samples 50
    python scripts/eval.py --mock   # no GPU, uses MockLLM
"""
import argparse, json, logging, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.loop import ReActAgent, VLLMBackend, MockLLM
from src.agent.tools import build_tool_registry
from src.agent.reward import compute_reward

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--split",       default="test", choices=["train", "test"])
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--max_steps",   type=int, default=8)
    parser.add_argument("--mock",        action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(f"data/gsm8k/{args.split}.jsonl", args.max_samples)
    logger.info("Evaluating %s on %d tasks", args.model, len(tasks))

    llm   = MockLLM() if args.mock else VLLMBackend(model_name=args.model, temperature=0.0)
    tools = build_tool_registry()
    agent = ReActAgent(llm=llm, tools=tools, max_steps=args.max_steps)

    results, correct = [], 0
    for i, task in enumerate(tasks):
        traj = agent.run(task_id=task["task_id"], query=task["query"],
                         ground_truth=task["ground_truth"])
        compute_reward(traj)
        results.append(traj)
        if traj.success: correct += 1
        if (i + 1) % 10 == 0:
            logger.info("[%d/%d] success=%.1f%%", i+1, len(tasks), 100*correct/(i+1))

    n          = len(results)
    success    = correct / n
    avg_reward = sum(r.reward for r in results if r.reward) / n
    avg_steps  = sum(r.total_steps for r in results) / n

    print(f"\n{'='*50}")
    print(f"Model:        {args.model}")
    print(f"Tasks:        {n}")
    print(f"Success rate: {success:.1%}")
    print(f"Mean reward:  {avg_reward:.3f}")
    print(f"Mean steps:   {avg_steps:.1f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()