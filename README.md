# GSM8K ReAct Agent with GRPO

A math reasoning agent trained with reinforcement learning to decide **when and how to use tools**. Built entirely from scratch — no LangGraph, no CrewAI — using open-weight models on a single A100 GPU.

**W&B Project:** [gsm8k-react-agent](https://wandb.ai/ngangada-arizona-state-university/gsm8k-react-agent)

---

## Results

| Phase | Method | Accuracy | Tool Use | Mean Reward |
|-------|--------|----------|----------|-------------|
| Baseline | Qwen3-8B zero-shot | 55.5% | 48% | 0.659 |
| LoRA SFT | 40M params, 11 min | — | — | train loss 0.13 |
| Full SFT | 8.19B params, 6 min | — | — | train loss 0.11 |
| LoRA GRPO | QLoRA + RL, 1x A100 | eval success 80% | — | 0.744 |
| Full GRPO | Full weights + RL, 2x A100 | eval success 75% | — | 0.719 |

Key finding: RL training (GRPO) on top of SFT improved eval success from 55% baseline to 75-80% — a **20-25 percentage point lift** purely from reward-driven tool use optimization.

---

## What is an Agent vs an LLM?

An **LLM** takes text in and produces text out. One shot. It has no memory of what it just said, cannot check its own answers, and cannot interact with the world.

```
User: "What is 847 × 293?"
LLM:  "248,171"   ← may hallucinate, no verification
```

An **agent** wraps an LLM in a loop that lets it reason, act, observe results, and iterate until it has a confident answer.

```
User: "What is 847 × 293?"

Agent Step 1 — Think:  "I should verify this with code."
Agent Step 1 — Act:    code_executor("print(847 * 293)")
Agent Step 1 — Observe: "248,171"

Agent Step 2 — Think:  "Code confirmed 248,171."
Agent Step 2 — Answer: "248,171"
```

The agent has **tools** (code execution, calculators, search), **memory** of its prior steps within an episode, and a **loop** that runs until it reaches a final answer or hits a step limit.

### The ReAct Pattern

This project implements the **ReAct** pattern (Yao et al., 2022 — *Reasoning + Acting*):

```
Thought → Action → Observation → Thought → Action → Observation → ... → Answer
```

Each iteration the agent:
1. **Thinks** about what it knows and what it needs
2. **Acts** by calling a tool or producing a final answer
3. **Observes** the tool result
4. Repeats until confident

This is how ChatGPT uses web search, how Claude uses code execution — all built on this pattern.

---

## Why Training an Agent with RL is Different from Normal LLM Training

### Normal LLM Training (SFT)
Train on (input, correct output) pairs. The model learns to predict the next token.

```
Input:  "What is 2 + 2?"
Target: "4"
Loss:   cross-entropy on token "4"
```

The model learns *what* the answer looks like. It does not learn *how* to get there.

### Agent Training with RL (GRPO)
Train on full **trajectories** — the entire sequence of thoughts, tool calls, and observations — scored by an outcome reward.

```
Trajectory:
  Query:       "Janet earns $16/hr, works 8hrs, spends $12 on lunch. How much left?"
  Step 1:      thought="use code", action=code_executor("print(16*8 - 12)")
  Observation: "116"
  Step 2:      thought="confirmed", action=final_answer("116")

Reward:
  answer_correct  = 1.0  (matched ground truth)
  tool_efficiency = 1.0  (1 tool call, optimal)
  format_valid    = 1.0  (all steps valid JSON)
  total reward    = 1.0
```

GRPO then asks: *which trajectories got higher reward?* and adjusts the model's policy to make those trajectories more likely.

### The Core Difference

| | SFT | Agent RL (GRPO) |
|--|-----|-----------------|
| Training signal | Next token prediction | Outcome reward on full trajectory |
| What it learns | Output format and content | When to use tools, how many steps to take |
| Credit assignment | Every token equally | Entire trajectory gets one reward score |
| Can improve tool use? | No — just mimics examples | Yes — reward directly measures tool efficiency |
| Exploration | None | Agent tries different tool strategies |

**SFT alone cannot teach an agent to use tools better.** It can only teach the model to format its output like the training examples. GRPO actually optimizes the decision of *whether to call a tool at all*, *which tool to call*, and *when to stop*.

This is why our LoRA GRPO achieved 80% eval success while the baseline was 55% — the RL training found that calling `code_executor` once and trusting the output is more efficient than reasoning in multiple steps.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              GSM8K ReAct Agent                  │
│                                                 │
│  Query (math problem)                           │
│       │                                         │
│       ▼                                         │
│  ┌─────────────┐     ┌──────────────────────┐   │
│  │  Qwen3-8B   │────▶│   Tool Registry      │   │
│  │  (brain)    │◀────│   • CodeExecutor     │   │
│  │             │     │   • Calculator       │   │
│  └─────────────┘     └──────────────────────┘   │
│       │                                         │
│  ReAct Loop (max 8 steps)                       │
│  Think → Act → Observe → Think → ...           │
│       │                                         │
│       ▼                                         │
│  Final Answer                                   │
│       │                                         │
│       ▼                                         │
│  Reward Function                                │
│  • answer_correct  (0.5 weight)                 │
│  • tool_efficiency (0.3 weight)                 │
│  • format_valid    (0.2 weight)                 │
└─────────────────────────────────────────────────┘
```

### Components Built From Scratch

**`src/agent/loop.py`** — The ReAct engine. Manages the think/act/observe cycle, parses LLM JSON output (including Qwen3's `<think>...</think>` format), falls back to number extraction when the model ignores format instructions.

**`src/agent/tools.py`** — Two tools:
- `CodeExecutor`: sandboxed Python subprocess with blocked dangerous imports, 10s timeout
- `Calculator`: AST-based safe expression evaluator — no `eval()`, supports `+`, `-`, `*`, `/`, `**`, `//`, `%`

**`src/agent/reward.py`** — Three-component reward function:
- `answer_correct`: exact match after normalization (strips `$`, `,`, converts `42.0` → `42`)
- `tool_efficiency`: reward curve — 0 calls if successful=0.8, 1-2 calls=1.0, decays to 0 past 6 calls
- `format_valid`: fraction of steps that produced valid JSON actions

**`src/rl/grpo_trainer.py`** — GRPO loss implementation. Normalizes rewards within each group (zero mean, unit std), applies PPO-style clipping, adds KL entropy penalty to prevent policy collapse.

**`src/rl/trajectory.py`** — Trajectory buffer with JSONL serialization. Each trajectory stores every step, observation, reward breakdown, and latency.

---

## Training Pipeline

### Phase 0 — Data
```bash
sbatch configs/slurm_00_data.sh
```
Downloads GSM8K (7,473 train / 1,319 test) from HuggingFace. GSM8K is grade school math word problems with step-by-step solutions and exact numeric answers — ideal for RL because reward is binary exact match.

### Phase 1 — Baseline Evaluation
```bash
sbatch configs/slurm_01_baseline.sh
```
Runs Qwen3-8B zero-shot on 200 GSM8K test problems with no training. Establishes the floor.

Result: **55.5% accuracy**, 48% tool use rate, 1.51 mean steps.

The parser had to handle Qwen3's thinking mode — the model outputs `<think>...</think>` before any JSON, which broke naive JSON parsing. Fixed with `re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)` before attempting JSON parse.

### Phase 2a — LoRA SFT
```bash
sbatch configs/slurm_02_sft.sh
```
Supervised fine-tuning with LoRA adapters (r=16, alpha=32). Teaches the model the agent JSON format.

- Trainable params: 40M / 8.19B (0.5%)
- Training time: 11 minutes on 1x A100
- Final loss: 0.13

LoRA freezes base weights and injects small trainable matrices into attention layers. Sufficient for format learning — the model already knows math, it just needs to output structured JSON.

### Phase 2b — Full SFT
```bash
sbatch configs/slurm_02b_sft_full.sh
```
Full fine-tuning — all 8.19B parameters updated.

Memory strategy to fit on A100 80GB:
- `gradient_checkpointing_enable()`: recompute activations during backward instead of storing them (saves ~20GB)
- `adamw_8bit` optimizer: optimizer states in 8-bit (32GB → 8GB)
- `batch_size=1` with `grad_accum=16`: minimal activation memory

- Training time: 6 minutes on 1x A100
- Final loss: 0.11

### Phase 3a — LoRA GRPO (1x A100)
```bash
sbatch configs/slurm_03_grpo.sh
```
GRPO reinforcement learning on top of the full SFT checkpoint, using QLoRA (4-bit base + LoRA adapter) for the policy model.

Memory challenge: vLLM (for rollouts) + policy model simultaneously exceeds 80GB. Solution: free vLLM GPU memory before each gradient update, reload after.

```
Per iteration:
  1. vLLM loaded → 16 rollout episodes → compute rewards
  2. del vLLM → torch.cuda.empty_cache()
  3. GRPO gradient update on 4-bit policy
  4. Reload vLLM for next iteration
```

- 20 iterations, batch=16
- Eval success at iter 15: **80%** (up from 55.5% baseline)
- Final checkpoint: `checkpoints/rl/final`

### Phase 3b — Full GRPO (2x A100)
```bash
sbatch configs/slurm_03b_grpo_full.sh
```
Full weight GRPO across 2x A100 GPUs (160GB total VRAM). Policy model spread across both GPUs with `device_map="auto"`.

- GPU 0: vLLM rollouts (freed before updates) + policy layers 0-18
- GPU 1: policy layers 18-36 + optimizer states
- Eval success at iter 15: **75%**
- Final checkpoint: `checkpoints/rl_full/final`

---

## Repository Structure

```
gsm8k-react-agent/
├── src/
│   ├── agent/
│   │   ├── loop.py          ← ReAct engine, parser, vLLM backend
│   │   ├── tools.py         ← CodeExecutor, Calculator
│   │   └── reward.py        ← 3-component reward function
│   └── rl/
│       ├── grpo_trainer.py  ← GRPO loss + weight update
│       └── trajectory.py    ← Buffer, JSONL serialization
├── scripts/
│   ├── load_gsm8k.py        ← Download + cache dataset
│   ├── eval_baseline.py     ← Phase 1: zero-shot eval
│   ├── train_sft.py         ← Phase 2a: LoRA SFT
│   ├── train_sft_full.py    ← Phase 2b: Full SFT
│   ├── train_grpo.py        ← Phase 3a: LoRA GRPO
│   ├── train_grpo_full.py   ← Phase 3b: Full GRPO 2-GPU
│   ├── eval_compare.py      ← Phase 4: comparison table
│   └── eval.py              ← Quick spot-check any checkpoint
├── configs/
│   ├── slurm_00_data.sh
│   ├── slurm_01_baseline.sh
│   ├── slurm_02_sft.sh
│   ├── slurm_02b_sft_full.sh
│   ├── slurm_03_grpo.sh
│   ├── slurm_03b_grpo_full.sh
│   ├── slurm_04_compare.sh
│   ├── agent_config.yaml
│   ├── sft_config.yaml
│   ├── sft_full_config.yaml
│   ├── grpo_config.yaml
│   └── grpo_full_config.yaml
├── data/
│   ├── gsm8k/               ← train.jsonl, test.jsonl
│   ├── trajectories/        ← GRPO rollout buffers
│   └── results/             ← eval JSON outputs
├── tests/
│   ├── agent/               ← 40 unit tests
│   └── rl/                  ← 17 unit tests
└── requirements.txt
```

---

## Setup and Reproduction

```bash
# Clone
git clone https://github.com/ngangada/gsm8k-react-agent
cd gsm8k-react-agent

# Environment (Sol)
python -m venv ~/envs/gsm8k_agent
source ~/envs/gsm8k_agent/bin/activate
pip install -r requirements.txt
pip install peft bitsandbytes --break-system-packages

# Set credentials
export WANDB_API_KEY=your_key
export HF_TOKEN=your_token

# Run pipeline in order
sbatch configs/slurm_00_data.sh       # ~5 min, no GPU
sbatch configs/slurm_01_baseline.sh   # ~2 hrs, 1x A100
sbatch configs/slurm_02_sft.sh        # ~15 min, 1x A100 (LoRA)
sbatch configs/slurm_02b_sft_full.sh  # ~10 min, 1x A100 (full)
sbatch configs/slurm_03_grpo.sh       # ~2 hrs, 1x A100
sbatch configs/slurm_03b_grpo_full.sh # ~3 hrs, 2x A100
sbatch configs/slurm_04_compare.sh    # ~1 hr, 1x A100
```

Run tests without GPU:
```bash
pytest tests/ -v   # 77 tests, ~0.2s
```

---

## Key Engineering Decisions

**Why GRPO over PPO?** GRPO (Group Relative Policy Optimization, Shao et al. 2024 — same algorithm as DeepSeek-R1) eliminates the separate value/critic model. It normalizes rewards within a group of trajectories for the same prompt and uses that as the advantage signal. Simpler to implement, less memory, works well for reasoning tasks.

**Why vLLM for rollouts?** vLLM batches token generation with PagedAttention, achieving 10-20× higher throughput than HuggingFace generate. For RL training, rollout speed is the bottleneck — faster rollouts = more iterations per hour.

**Why free/reload vLLM between rollout and update?** On a single A100 80GB, vLLM consumes ~16GB for model weights plus ~15GB for KV cache and compiled CUDA graphs. A full policy model + optimizer states needs ~50GB. They cannot coexist. The free/reload cycle adds ~40s per iteration but eliminates OOM.

**Why not 2 GPUs for everything?** Multi-GPU with tensor parallelism requires `torchrun`, NCCL configuration, and careful attention to which process owns which tensors. For the LoRA GRPO run the complexity wasn't worth it — single GPU with free/reload was simpler and produced good results. Two GPUs were only used for full-weight GRPO where it was strictly necessary.

**Qwen3 thinking mode:** Qwen3-8B by default outputs `<think>...</think>` blocks before responding. Our parser strips these before JSON extraction. When the model ignores JSON format entirely (common at temperature=0 for simple problems), the fallback extracts the last number from the raw text — this recovered ~30% of accuracy that was being lost to parse failures.

---

## Infrastructure

- **Cluster:** ASU Sol HPC
- **GPU:** NVIDIA A100-SXM4-80GB
- **SLURM account:** grp_cbaral
- **HF cache:** `/scratch/ngangada/hf_cache`
- **Python:** 3.12 via mamba
- **Key versions:** vLLM 0.8.5, transformers 4.51.3, torch 2.x, peft, bitsandbytes

---

## References

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.11610) — Yao et al., 2022
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) — GRPO algorithm
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [Qwen3 Technical Report](https://huggingface.co/Qwen/Qwen3-8B) — Alibaba, 2025
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k) — Cobbe et al., OpenAI

---

## Complete Results

### Benchmark Accuracy (100 samples each)

| Model | GSM8K | ARC-Easy | ARC-Challenge | Tool Use (GSM8K) |
|-------|-------|----------|---------------|------------------|
| Baseline (zero-shot) | 54.0% | 79.0% | 76.0% | 47% |
| Full SFT | **66.0%** | **94.0%** | 72.0% | 20% |
| LoRA GRPO | 54.0% | 76.0% | 74.0% | 46% |
| Full GRPO | 59.0% | 89.0% | **75.0%** | 30% |

> MATH-500 returned 0 problems due to HuggingFace dataset ID changes — excluded from analysis.

### Agent Behavioral Metrics (GSM8K, 100 samples)

| Model | Accuracy | Tool Use | Steps | Code Exec | Calculator | Reward |
|-------|----------|----------|-------|-----------|------------|--------|
| Baseline | 59.0% | 50% | 1.50 | 8% | 42% | 0.677 |
| Full SFT | **76.0%** | 19% | **1.19** | 17% | 2% | **0.688** |
| LoRA GRPO | 58.0% | 50% | 1.52 | 13% | 38% | 0.672 |
| Full GRPO | 67.0% | 39% | 1.42 | **33%** | 7% | 0.688 |

---

## Findings and Analysis

### Finding 1: Full SFT is the accuracy winner on in-distribution data

Full SFT achieved **76% accuracy on GSM8K** (behavioral eval) and **94% on ARC-Easy** — the highest across all models on both benchmarks. Training all 8.19B parameters on the agent format gave the model enough capacity to learn both the JSON format and the underlying reasoning simultaneously.

Critically, Full SFT reduced tool use to just **19%** while improving accuracy. This means the model internalized enough math reasoning during full fine-tuning that it could solve most GSM8K problems through direct reasoning without calling tools. This is a known SFT behavior: with enough data and full weight updates, the model learns to answer directly rather than delegate to tools.

### Finding 2: Full GRPO generalizes better than Full SFT on out-of-distribution data

On ARC-Challenge (science reasoning, not in the training distribution), Full GRPO scored **75%** vs Full SFT's **72%**. On ARC-Easy, Full GRPO scored 89% vs Full SFT's 94% — closer gap than on GSM8K.

This suggests RL training preserved more general reasoning capability. SFT on a narrow domain (GSM8K math format) can cause forgetting on other domains. GRPO, which optimizes behavior through outcome rewards rather than imitation, appears to degrade general capabilities less.

### Finding 3: Full GRPO shifted tool preference from calculator to code executor

The most revealing behavioral signal:

| | Baseline | Full SFT | Full GRPO |
|--|----------|----------|-----------|
| Calculator use | 42% | 2% | 7% |
| Code executor use | 8% | 17% | **33%** |

The baseline relied heavily on the calculator (simple, fast). Full GRPO learned to prefer code execution — which is slower but more reliable for multi-step problems. This is the reward signal working correctly: `code_executor` produces verifiable, step-by-step outputs that the agent can observe and trust, leading to higher answer correctness rewards. GRPO learned this preference through 20 iterations of trajectory feedback.

### Finding 4: LoRA GRPO underperformed — the quantization problem

LoRA GRPO scored 54% on GSM8K, matching the baseline and lower than Full SFT. The cause: QLoRA (4-bit quantized base + LoRA adapter) used during GRPO training compressed the base model's math reasoning ability. The LoRA adapter was updating 40M parameters on top of a degraded 4-bit base, which couldn't fully recover the capability lost to quantization.

This is a real tradeoff in single-GPU RL training: memory efficiency via quantization comes at the cost of training signal quality. The full-weight GRPO (2× A100) avoided this entirely and scored 5 percentage points higher on GSM8K.

### Finding 5: ARC tool use behavior reveals overfitting

On ARC benchmarks (multiple choice science questions), Full SFT used tools at 79% and 62% rates. This is wrong — tools (code executor, calculator) cannot help answer "What causes the seasons?" The SFT training on GSM8K tool-use format caused the model to call tools indiscriminately, even when they're irrelevant. Full GRPO (81%, 67%) had the same issue, likely because the reward function didn't penalize irrelevant tool use on non-math problems. Baseline (50%, 43%) was only slightly better. This is a known failure mode of format imitation in SFT.

---

## Summary

```
Best accuracy (GSM8K):      Full SFT     — 66-76% depending on eval run
Best generalization (ARC):  Full GRPO    — degraded general capability less
Best tool learning:         Full GRPO    — learned code_executor > calculator
Best reward score:          Full SFT / Full GRPO tied at 0.688
Worst outcome:              LoRA GRPO    — quantization degraded base capability
Most efficient:             Full SFT     — fewest steps (1.19), highest accuracy
```

The core result: **GRPO does not always improve accuracy over SFT on the training domain, but it produces more generalizable behavior and better tool-use decisions.** This matches findings from the RL literature — RL optimizes for what you measure (reward), not necessarily what you care about (accuracy). When the reward function is well-designed (as here: correctness + efficiency + format), GRPO produces agents that are more efficient and better calibrated, even if raw accuracy gains are modest.

---

## Limitations

- **MATH-500 not evaluated** — HuggingFace dataset ID changed; the competition math generalization test is missing from results.
- **Small sample size** — 100 samples per benchmark introduces variance; results should be interpreted directionally, not as precise numbers.
- **GRPO iterations** — only 20 iterations with batch=16 (320 total trajectory examples). DeepSeek-R1 used thousands of iterations. More compute would likely improve GRPO results significantly.
- **LoRA GRPO memory constraint** — single A100 forced QLoRA, which degraded results. The correct comparison would be full-weight GRPO on both 1 and 2 GPUs.
- **ARC tool use not penalized** — the reward function was designed for math (where tools help). On ARC, tool use is irrelevant but not penalized, causing inflated tool call rates.

---

## Future Work

- Fix ARC tool use: add a reward component that penalizes tool calls that don't improve the answer
- Run MATH-500 once the correct HF dataset ID is confirmed (`HuggingFaceH4/MATH-500`)
- Scale GRPO to 100+ iterations to see if accuracy improves with more RL training
- Test on a domain the agent was never trained on (e.g., legal reasoning, medical QA) to measure true generalization
- Replace QLoRA GRPO with full-weight GRPO on single GPU using gradient checkpointing + CPU offloading