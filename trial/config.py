"""
config.py - Central configuration for the SWE-bench bug injection system.
"""

import os

# ── GitHub API ───────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")  # optional; raises rate-limit from 60 → 5000 req/hr

# ── LLM / AutoGen ────────────────────────────────────────────────────────────
LLM_CONFIG = {
    "config_list": [
        {
            "model": "qwen2.5-coder:7b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",  # Ollama requires a non-empty string but ignores the value
            "price": [0, 0],  # free local model — silences AutoGen's cost warning
        }
    ],
    "temperature": 0.2,
    "timeout": 120,
    "cache_seed": None,  # disable caching so every run is fresh
}

# ── SWE-bench ─────────────────────────────────────────────────────────────────
SWEBENCH_DATASET = "princeton-nlp/SWE-bench"   # HuggingFace dataset id
SWEBENCH_SPLIT   = "test"                       # "train" | "test" | "dev"

# ── Repository cloning ────────────────────────────────────────────────────────
REPOS_DIR = "./repos"           # local directory where repos are cloned
PATCHES_DIR = "./patches"       # where generated .patch files are saved
REPORTS_DIR = "./reports"       # where JSON reports are saved

# ── Bug injection ─────────────────────────────────────────────────────────────
# Maximum number of SWE-bench instances to process per run
MAX_INSTANCES = 5

# Which bug categories to enable
ENABLED_BUG_TYPES = [
    "off_by_one",
    "wrong_operator",
    "logic_inversion",
    "missing_return",
    "wrong_variable",
    "wrong_constant",
    "missing_condition",
]

# How many injection attempts per file before giving up
MAX_INJECTION_RETRIES = 3
