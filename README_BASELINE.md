# Baseline: mini-SWE-agent + Ollama + SWE-bench Verified

This folder wires the **official [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)** SWE-bench batch runner to **Ollama** (`qwen2.5-coder:14b` by default), then evaluates patches with the **SWE-bench harness** (FAIL_TO_PASS + PASS_TO_PASS).

## Prerequisites

1. **Docker Desktop** (Linux containers; SWE-bench images are `x86_64`).
2. **Ollama** running locally, model pulled:
   ```bash
   ollama pull qwen2.5-coder:14b
   ```
3. Python packages (also listed in `requirements.txt`):
   ```bash
   pip install mini-swe-agent swebench
   ```

## Files

| File | Purpose |
|------|---------|
| `baseline_config/ollama_swebench.yaml` | Overrides model + step limit; merged with built-in `swebench.yaml` |
| `baseline_config/model_registry.json` | LiteLLM cost registry for local Ollama (zero cost) |
| `run_baseline.py` | Runs `python -m minisweagent.run.benchmarks.swebench` with correct `-c` flags and env |
| `evaluate_baseline.py` | Runs `python -m swebench.harness.run_evaluation` and writes a JSON summary |

## 1) Generate patches (mini-SWE-agent)

From the project root:

```powershell
python run_baseline.py --output baseline_results --slice 0:5 --split test
```

- Adjust `--slice` (e.g. `0:10`) or omit slice handling by editing the script to pass an empty slice (see `run_baseline.py` `--slice` default `0:5`).
- Outputs: `baseline_results/preds.json`, per-instance trajectories, logs.

Environment set automatically by `run_baseline.py`:

- `MSWEA_SILENT_STARTUP=1` — avoids Windows console Unicode issues when importing mini-SWE-agent.
- `LITELLM_MODEL_REGISTRY_PATH` — points to `baseline_config/model_registry.json`.
- `MSWEA_COST_TRACKING=ignore_errors`

### Single-instance debugging (optional)

Use the same env vars as `run_baseline.py` (`MSWEA_SILENT_STARTUP`, `LITELLM_MODEL_REGISTRY_PATH`), then call `python -m minisweagent.run.benchmarks.swebench_single` with `-c` pointing to the built-in `minisweagent/config/benchmarks/swebench.yaml` plus `baseline_config/ollama_swebench.yaml`. See [mini-SWE-agent SWE-bench docs](https://mini-swe-agent.com/latest/usage/swebench/).

## 2) Evaluate patches (SWE-bench harness)

```powershell
python evaluate_baseline.py --preds baseline_results/preds.json --run-id my_baseline_v1 --sanitize-model-name
```

- **`--sanitize-model-name`**: On Windows, model strings like `ollama/qwen2.5-coder:14b` contain `:` which can break log paths. This rewrites `model_name_or_path` in a temp file used only for evaluation.

Use the same dataset id as mini-SWE-agent’s `verified` subset: **`princeton-nlp/SWE-Bench_Verified`** (default in `evaluate_baseline.py`).

The harness applies each patch in Docker and runs the official test script; an instance is **resolved** only when both FAIL_TO_PASS and PASS_TO_PASS criteria are met (see `swebench.harness.grading`).

Summary JSON: `evaluation_summary_<run_id>.json` (or `--summary-out`).

## Metrics

- **Resolved**: `true` in each `logs/run_evaluation/<run_id>/.../report.json` when the full F2P + P2P resolution holds.
- Per-test breakdown is under `tests_status` in the same `report.json`; `evaluate_baseline.py` copies the main F2P/P2P lists into the summary JSON.

## Comparing to your experiments

Keep the same `preds.json` shape and `evaluate_baseline.py` pipeline so attacker/defender + RAG runs are comparable to this baseline.
