# SWE-bench Bug Injection Pipeline

An **AutoGen multi-agent system** that automatically injects realistic, subtle bugs into repositories from the [SWE-bench](https://swe-bench.github.io/) benchmark dataset.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    OrchestratorAgent                     │
│  Controls the end-to-end pipeline for each instance      │
└──────┬──────────────────┬──────────────────┬─────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
│AnalyzerAgent│  │InjectorAgent │  │  ValidatorAgent  │
│             │  │              │  │                  │
│ Reads the   │  │ 1. Tries AST │  │ Reviews the bug: │
│ source file │  │    mutators  │  │ - syntactically  │
│ and picks:  │  │ 2. Falls back│  │   valid?         │
│ · strategy  │  │    to LLM if │  │ - realistic?     │
│ · line hint │  │    needed    │  │ - subtle enough? │
└─────────────┘  └──────────────┘  └──────────────────┘
```

### Agent Responsibilities

| Agent | Role |
|---|---|
| **OrchestratorAgent** | Drives the full pipeline; picks target files; collects reports |
| **AnalyzerAgent** | Reads source; picks best injection strategy & line via LLM |
| **InjectorAgent** | Applies deterministic AST mutation or falls back to LLM |
| **ValidatorAgent** | Scores realism and subtlety of the injected bug (1–5) |

---

## Bug Strategies

| Strategy | Description |
|---|---|
| `off_by_one` | Shift an integer constant by ±1 |
| `wrong_operator` | Swap a comparison operator (`<` → `<=`, `==` → `!=`, …) |
| `logic_inversion` | Swap `and` ↔ `or` in boolean expressions |
| `wrong_arithmetic` | Swap `+` ↔ `-` or `*` ↔ `//` |
| `missing_return` | Remove the `return` statement from a function |
| `wrong_variable` | Swap two local variable names |
| `wrong_constant` | Flip a boolean constant (`True` ↔ `False`) |
| `missing_condition` | Remove an `if` guard, always executing the body |

Deterministic strategies use Python's `ast` module — they are always syntax-safe. The LLM fallback is used when no valid AST mutation site is found.

---

## Project Structure

```
swebench_bug_injector/
├── main.py            # CLI entry point
├── config.py          # Centralised configuration
├── agents.py          # AutoGen agent definitions
├── bug_strategies.py  # AST-based deterministic mutators
├── repo_manager.py    # Git clone / patch management
├── swebench_loader.py # HuggingFace dataset loader
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run the pipeline

```bash
# Process 3 instances from the SWE-bench test split
python main.py --n 3

# Target specific instance IDs
python main.py --ids astropy__astropy-12907 django__django-11019

# Use the dev split
python main.py --n 5 --split dev
```

---

## Output

### Patches
Each successful injection produces a unified diff in `./patches/<instance_id>.patch`:

```diff
--- a/astropy/io/fits/card.py
+++ b/astropy/io/fits/card.py
@@ -234,7 +234,7 @@
-        if len(value) <= 68:
+        if len(value) < 68:
             return value
```

### JSON Report
A full run report is saved to `./reports/run_<timestamp>.json`:

```json
[
  {
    "instance_id": "astropy__astropy-12907",
    "repo": "astropy/astropy",
    "base_commit": "abc123",
    "status": "success",
    "strategy": "wrong_operator",
    "target_file": "astropy/io/fits/card.py",
    "score": 4,
    "validation": {
      "valid": true,
      "score": 4,
      "feedback": "Subtle operator change that could cause boundary-condition failures."
    },
    "patch": "--- a/..."
  }
]
```

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `LLM_CONFIG` | GPT-4o | AutoGen LLM configuration |
| `SWEBENCH_SPLIT` | `"test"` | Dataset split to use |
| `MAX_INSTANCES` | `5` | Max instances per run |
| `ENABLED_BUG_TYPES` | all | Which strategies to enable |
| `MAX_INJECTION_RETRIES` | `3` | AST mutation retry limit |
| `REPOS_DIR` | `"./repos"` | Where repos are cloned |
| `PATCHES_DIR` | `"./patches"` | Where patches are saved |

---

## Extending

### Adding a new strategy

1. Write a function `inject_<name>(source: str) -> str | None` in `bug_strategies.py`
2. Register it in the `STRATEGIES` dict at the bottom of that file
3. Add its name to `ENABLED_BUG_TYPES` in `config.py`

### Switching LLM

Edit `LLM_CONFIG` in `config.py` to use Azure, Ollama, or any AutoGen-compatible provider.
