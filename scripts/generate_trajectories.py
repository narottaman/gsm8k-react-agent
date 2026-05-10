"""
scripts/generate_trajectories.py
---------------------------------
Generate and save agent trajectories to JSONL — without running GRPO.
Useful for:
  - Inspecting what the agent actually does step by step
  - Debugging tool use before training
  - Pre-generating rollouts to speed up training

Usage:
    python scripts/generate_trajectories.py --model Qwen/Qwen3-8B-Instruct --n 100
    python scripts/generate_trajectories.py --mock --n 20
"""
import argparse, json, logging, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.loop import ReActAgent, VLLMBackend, MockLLM
from src.agent.tools import build_tool_registry
from src.agent.reward import compute_reward
from src.rl.trajectory import TrajectoryBuffer, traj_to_dict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_tasks(path: str, n: int) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: tasks.append(json.loads(line))
    return random.sample(tasks, min(n, len(tasks)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--split",     default="train", choices=["train", "test"])
    parser.add_argument("--n",         type=int, default=100, help="Number of trajectories")
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--out",       default="data/trajectories/generated.jsonl")
    parser.add_argument("--mock",      action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(f"data/gsm8k/{args.split}.jsonl", args.n)
    logger.info("Generating %d trajectories with model: %s", len(tasks), args.model)

    llm    = MockLLM() if args.mock else VLLMBackend(model_name=args.model, temperature=0.7)
    tools  = build_tool_registry()
    agent  = ReActAgent(llm=llm, tools=tools, max_steps=args.max_steps)
    buffer = TrajectoryBuffer(save_path=args.out)

    stats = {"success": 0, "tool_calls": 0, "total": 0}

    for i, task in enumerate(tasks):
        traj = agent.run(task_id=task["task_id"], query=task["query"],
                         ground_truth=task["ground_truth"])
        compute_reward(traj)
        buffer.add(traj)

        stats["total"]      += 1
        stats["success"]    += int(traj.success)
        stats["tool_calls"] += sum(1 for s in traj.steps
                                   if s.action.type == "tool_call")

        if (i + 1) % 20 == 0:
            logger.info("[%d/%d] success=%.1f%% | avg_tool_calls=%.1f",
                        i+1, len(tasks),
                        100 * stats["success"] / stats["total"],
                        stats["tool_calls"] / stats["total"])

    buffer.save()

    print(f"\n{'='*50}")
    print(f"Saved:         {len(buffer)} trajectories → {args.out}")
    print(f"Success rate:  {stats['success']/stats['total']:.1%}")
    print(f"Avg tool calls:{stats['tool_calls']/stats['total']:.1f}")
    print(f"Mean reward:   {buffer.mean_reward():.3f}")
    print(f"{'='*50}\n")

    # Print 2 example trajectories for inspection
    print("=== Example Trajectory ===")
    sample = buffer._buf[0]
    print(f"Query:   {sample.query[:80]}")
    print(f"Answer:  {sample.final_answer} (GT: {sample.ground_truth})")
    print(f"Reward:  {sample.reward:.3f} | Steps: {sample.total_steps}")
    for step in sample.steps:
        print(f"  Step {step.step_idx}: [{step.action.type}]", end="")
        if step.action.tool_name:
            print(f" {step.action.tool_name}", end="")
        if step.observation:
            print(f" → {step.observation[:50]}", end="")
        print()


if __name__ == "__main__":
    main()