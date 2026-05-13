"""
scripts/eval_all_benchmarks.py
-------------------------------
Runs all model versions against all benchmarks.

Benchmarks:
  GSM8K       — grade school math, exact number match (1,319 test)
  MATH-500    — competition math, harder generalization test (500 problems)
  ARC-Easy    — science reasoning, multiple choice (easy tier)
  ARC-Challenge — science reasoning, multiple choice (hard tier)

Models evaluated:
  baseline    — Qwen3-8B zero-shot
  full_sft    — full weight SFT on GSM8K format
  lora_grpo   — LoRA GRPO (RL on top of full SFT)
  full_grpo   — full weight GRPO on 2x A100 (if checkpoint exists)

Output:
  data/results/all_benchmarks.json   — raw results
  data/results/summary_table.txt     — clean comparison table

Usage:
    python scripts/eval_all_benchmarks.py --config configs/eval_config.yaml
    python scripts/eval_all_benchmarks.py --config configs/eval_config.yaml --mock
"""

import argparse, json, logging, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml, wandb
from src.agent.loop import ReActAgent, VLLMBackend, MockLLM, ActionType
from src.agent.tools import build_tool_registry
from src.agent.reward import compute_reward, normalize_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("logs/eval_all_benchmarks.log")]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_gsm8k(split="test", max_samples=None) -> list[dict]:
    path = Path(f"data/gsm8k/{split}.jsonl")
    if path.exists():
        tasks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line: 
                    t = json.loads(line)
                    t.setdefault("benchmark", "gsm8k")
                    t.setdefault("format", "number")
                    tasks.append(t)
        if max_samples:
            tasks = random.sample(tasks, min(max_samples, len(tasks)))
        return tasks

    # fallback: download
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    tasks = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples: break
        gt = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"].strip()
        tasks.append({"task_id": f"gsm8k_{i:05d}", "query": row["question"],
                      "ground_truth": gt, "benchmark": "gsm8k", "format": "number"})
    return tasks


def load_math500(max_samples=None) -> list[dict]:
    from datasets import load_dataset
    logger.info("Loading MATH-500...")
    # HendrycksTest/MATH is the correct current HF dataset ID
    # MATH-500 is the standard 500-problem test subset used in papers
    try:
        ds = load_dataset("hendrycks/competition_math", split="test")
    except Exception:
        try:
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        except Exception as e:
            logger.error("Could not load MATH-500: %s", e)
            return []

    # Take up to max_samples, preferring harder problems (level 4-5)
    all_rows  = list(ds)
    hard      = [r for r in all_rows if str(r.get("level", "")).endswith(("4", "5"))]
    pool      = hard if len(hard) >= (max_samples or 100) else all_rows
    sample    = random.sample(pool, min(max_samples or 500, len(pool)))

    tasks = []
    for i, row in enumerate(sample):
        solution = row.get("solution", "")
        boxed    = re.search(r"\\\\boxed\\{([^}]+)\\}", solution)
        gt       = boxed.group(1).strip() if boxed else ""
        if not gt:
            continue
        tasks.append({
            "task_id":      f"math500_{i:05d}",
            "query":        row.get("problem", row.get("question", "")),
            "ground_truth": gt,
            "benchmark":    "math500",
            "format":       "number",
            "subject":      row.get("type", row.get("subject", "")),
            "level":        str(row.get("level", "")),
        })
    logger.info("MATH-500 loaded: %d problems", len(tasks))
    return tasks


def load_arc(subset="easy", max_samples=None) -> list[dict]:
    from datasets import load_dataset
    config = "ARC-Easy" if subset == "easy" else "ARC-Challenge"
    logger.info("Loading ARC-%s...", subset)
    ds = load_dataset("allenai/ai2_arc", config, split="test")
    sample = list(ds)
    if max_samples:
        sample = random.sample(sample, min(max_samples, len(sample)))
    tasks = []
    for i, row in enumerate(sample):
        choices = row["choices"]
        choice_text = " | ".join(
            f"{l}: {t}" for l, t in zip(choices["label"], choices["text"])
        )
        query = f"{row['question']}\n\nChoices: {choice_text}\n\nAnswer with just the letter (A, B, C, or D)."
        tasks.append({
            "task_id":      f"arc_{subset}_{i:05d}",
            "query":        query,
            "ground_truth": row["answerKey"],
            "benchmark":    f"arc_{subset}",
            "format":       "multiple_choice",
        })
    logger.info("ARC-%s loaded: %d problems", subset, len(tasks))
    return tasks


# ---------------------------------------------------------------------------
# Answer evaluation
# ---------------------------------------------------------------------------

def is_correct(predicted: str, ground_truth: str, fmt: str) -> bool:
    if not predicted or not ground_truth:
        return False
    if fmt == "number":
        return normalize_answer(predicted) == normalize_answer(ground_truth)
    elif fmt == "multiple_choice":
        # Extract single letter from response
        pred_clean = predicted.strip().upper()
        letters    = re.findall(r"\b([A-D])\b", pred_clean)
        pred_letter = letters[-1] if letters else pred_clean[0] if pred_clean else ""
        return pred_letter == ground_truth.strip().upper()
    return predicted.strip().lower() == ground_truth.strip().lower()


# ---------------------------------------------------------------------------
# Model loader with LoRA merge support
# ---------------------------------------------------------------------------

def load_model_for_eval(model_path: str, base_model: str, temperature: float,
                         max_tokens: int, mock: bool) -> VLLMBackend:
    if mock:
        return MockLLM()

    p = Path(model_path)
    # LoRA adapter — needs merging
    if (p / "adapter_config.json").exists() and not (p / "config.json").exists():
        merged = model_path.rstrip("/") + "_merged"
        if not (Path(merged) / "config.json").exists():
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            logger.info("Merging LoRA %s into %s...", model_path, base_model)
            Path(merged).mkdir(parents=True, exist_ok=True)
            tok   = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, device_map="auto")
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            model.save_pretrained(merged)
            tok.save_pretrained(merged)
            logger.info("Merge saved → %s", merged)
        model_path = merged

    return VLLMBackend(model_name=model_path, temperature=temperature,
                       max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Run one model on one benchmark
# ---------------------------------------------------------------------------

def run_benchmark(agent: ReActAgent, tasks: list[dict]) -> dict:
    results = []
    benchmark = tasks[0].get("benchmark", "gsm8k") if tasks else "unknown"
    fmt       = tasks[0].get("format", "number") if tasks else "number"

    for i, task in enumerate(tasks):
        try:
            traj = agent.run(
                task_id=task["task_id"],
                query=task["query"],
                ground_truth=task["ground_truth"],
            )
            compute_reward(traj)

            tool_calls = sum(1 for s in traj.steps if s.action.type == ActionType.TOOL_CALL)
            correct    = is_correct(traj.final_answer, task["ground_truth"], fmt)

            results.append({
                "task_id":      traj.task_id,
                "correct":      correct,
                "predicted":    traj.final_answer,
                "ground_truth": task["ground_truth"],
                "tool_calls":   tool_calls,
                "used_tool":    tool_calls > 0,
                "n_steps":      traj.total_steps,
                "reward":       traj.reward,
            })
        except Exception as e:
            logger.warning("Task %s failed: %s", task["task_id"], e)

        if (i + 1) % 25 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            logger.info("  [%s %d/%d] acc=%.1f%%", benchmark, i+1, len(tasks), 100*acc)

    n          = len(results)
    accuracy   = sum(r["correct"]   for r in results) / n if n else 0
    tool_rate  = sum(r["used_tool"] for r in results) / n if n else 0
    mean_steps = sum(r["n_steps"]   for r in results) / n if n else 0

    return {
        "benchmark":  benchmark,
        "n":          n,
        "accuracy":   accuracy,
        "tool_rate":  tool_rate,
        "mean_steps": mean_steps,
        "results":    results,
    }


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_table(all_results: dict) -> str:
    benchmarks = ["gsm8k", "math500", "arc_easy", "arc_challenge"]
    models     = list(all_results.keys())

    header = f"{'Model':<16}" + "".join(f"{b:>14}" for b in benchmarks)
    sep    = "-" * (16 + 14 * len(benchmarks))

    lines = ["\n" + "="*len(sep),
             "BENCHMARK RESULTS — ACCURACY",
             "="*len(sep), header, sep]

    for model in models:
        row = f"{model:<16}"
        for bench in benchmarks:
            if bench in all_results[model]:
                acc = all_results[model][bench]["accuracy"]
                row += f"{acc:>13.1%}"
            else:
                row += f"{'—':>14}"
        lines.append(row)

    lines.append(sep)
    lines.append("\nTOOL USE RATE")
    lines.append(sep)

    for model in models:
        row = f"{model:<16}"
        for bench in benchmarks:
            if bench in all_results[model]:
                tr = all_results[model][bench]["tool_rate"]
                row += f"{tr:>13.1%}"
            else:
                row += f"{'—':>14}"
        lines.append(row)

    lines.append("="*len(sep) + "\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/eval_config.yaml")
    parser.add_argument("--mock",        action="store_true")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--benchmarks",  nargs="+",
                        default=["gsm8k", "math500", "arc_easy", "arc_challenge"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    wandb.init(project=cfg.get("wandb_project", "gsm8k-react-agent"),
               name="all-benchmarks", tags=["eval", "comparison", "benchmarks"])

    base_model  = cfg.get("model_name", "Qwen/Qwen3-8B")
    temperature = cfg.get("temperature", 0.0)
    max_tokens  = cfg.get("max_tokens", 512)
    max_steps   = cfg.get("max_steps", 8)
    max_samples = args.max_samples

    # Models to eval — skip if checkpoint missing
    model_configs = [
        ("baseline",  base_model,                                        None),
        ("full_sft",  cfg.get("sft_full_checkpoint", "checkpoints/sft_full"), None),
        ("lora_grpo", cfg.get("rl_checkpoint",   "checkpoints/rl/final"),     base_model),
        ("full_grpo", cfg.get("rl_full_checkpoint","checkpoints/rl_full/final"), None),
    ]

    # Load benchmarks
    benchmark_tasks = {}
    if "gsm8k"         in args.benchmarks:
        benchmark_tasks["gsm8k"]         = load_gsm8k(max_samples=max_samples)
    if "math500"       in args.benchmarks:
        benchmark_tasks["math500"]       = load_math500(max_samples=min(max_samples, 100))
    if "arc_easy"      in args.benchmarks:
        benchmark_tasks["arc_easy"]      = load_arc("easy",      max_samples=max_samples)
    if "arc_challenge" in args.benchmarks:
        benchmark_tasks["arc_challenge"] = load_arc("challenge", max_samples=max_samples)

    tools      = build_tool_registry()
    all_results = {}

    for label, model_path, lora_base in model_configs:
        # Skip if checkpoint doesn't exist
        p = Path(model_path)
        if model_path != base_model and not p.exists():
            logger.warning("Skipping %s — checkpoint not found: %s", label, model_path)
            continue

        logger.info("\n%s\nEvaluating: %s\n%s", "="*50, label, "="*50)

        try:
            llm = load_model_for_eval(model_path, base_model, temperature,
                                       max_tokens, args.mock)
        except Exception as e:
            logger.error("Failed to load %s: %s", label, e)
            continue

        agent = ReActAgent(llm=llm, tools=tools, max_steps=max_steps)
        all_results[label] = {}

        for bench_name, tasks in benchmark_tasks.items():
            logger.info("  Running %s on %s (%d tasks)...", label, bench_name, len(tasks))
            result = run_benchmark(agent, tasks)
            all_results[label][bench_name] = result
            logger.info("  %s | %s | acc=%.1f%% | tool=%.1f%%",
                        label, bench_name,
                        100 * result["accuracy"],
                        100 * result["tool_rate"])
            wandb.log({
                f"{label}/{bench_name}/accuracy":  result["accuracy"],
                f"{label}/{bench_name}/tool_rate": result["tool_rate"],
                f"{label}/{bench_name}/mean_steps":result["mean_steps"],
            })

        # Unload model between evals to free GPU memory
        if not args.mock:
            import gc, torch
            del llm, agent
            gc.collect()
            torch.cuda.empty_cache()

    # Print + save results
    table = print_table(all_results)
    print(table)

    out = Path("data/results")
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "all_benchmarks.json", "w") as f:
        # Don't save per-task results for brevity
        summary = {
            model: {
                bench: {k: v for k, v in data.items() if k != "results"}
                for bench, data in benches.items()
            }
            for model, benches in all_results.items()
        }
        json.dump(summary, f, indent=2)

    with open(out / "summary_table.txt", "w") as f:
        f.write(table)

    logger.info("Saved → data/results/all_benchmarks.json")
    logger.info("Saved → data/results/summary_table.txt")
    wandb.finish()


if __name__ == "__main__":
    main()
