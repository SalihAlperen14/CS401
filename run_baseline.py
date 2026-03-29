#!/usr/bin/env python3
"""
Run mini-SWE-agent (official SWE-bench batch script) with Ollama + Qwen2.5-Coder-14B.

Requires:
  - Docker (SWE-bench images)
  - Ollama running with: ollama pull qwen2.5-coder:14b
  - pip: mini-swe-agent

Usage (from project root):
  python run_baseline.py --output baseline_results --slice 0:5

Environment:
  MSWEA_SILENT_STARTUP=1  (set automatically) avoids Windows console Unicode issues on import.
  LITELLM_MODEL_REGISTRY_PATH  (set automatically) points to baseline_config/model_registry.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def find_builtin_swebench_yaml() -> Path:
    """Locate minisweagent/config/benchmarks/swebench.yaml without importing minisweagent."""
    import site

    roots: list[Path] = []
    u = getattr(site, "getusersitepackages", lambda: None)()
    if isinstance(u, str):
        roots.append(Path(u))
    elif isinstance(u, list):
        roots.extend(Path(p) for p in u)
    for p in site.getsitepackages():
        roots.append(Path(p))
    for root in roots:
        cand = root / "minisweagent" / "config" / "benchmarks" / "swebench.yaml"
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "Could not find minisweagent/config/benchmarks/swebench.yaml. "
        "Install mini-swe-agent: pip install mini-swe-agent"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run mini-SWE-agent SWE-bench batch with Ollama.")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("baseline_results"),
        help="Output directory (preds.json, trajectories, logs)",
    )
    p.add_argument(
        "--subset",
        default="verified",
        help="SWE-bench subset key (verified, lite, full) or HuggingFace dataset id",
    )
    p.add_argument("--split", default="test", help="Dataset split (use 'test' for Verified)")
    p.add_argument(
        "--slice",
        default="0:5",
        help="Instance slice, e.g. 0:5 for first five. Ignored if --all-instances is set.",
    )
    p.add_argument(
        "--all-instances",
        action="store_true",
        help="Run on the full subset (no --slice passed to mini-SWE-agent)",
    )
    p.add_argument("--workers", "-w", type=int, default=1, help="Parallel workers")
    p.add_argument(
        "--model",
        "-m",
        default="ollama/qwen2.5-coder:14b",
        help="LiteLLM model id (Ollama)",
    )
    p.add_argument(
        "--step-limit",
        type=int,
        default=30,
        help="Max agent steps (merged into config as agent.step_limit)",
    )
    p.add_argument(
        "--redo-existing",
        action="store_true",
        help="Re-run instances already present in preds.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _project_root()
    ollama_cfg = root / "baseline_config" / "ollama_swebench.yaml"
    registry = root / "baseline_config" / "model_registry.json"
    if not ollama_cfg.is_file():
        print(f"Missing config: {ollama_cfg}", file=sys.stderr)
        return 2
    if not registry.is_file():
        print(f"Missing registry: {registry}", file=sys.stderr)
        return 2

    builtin_sw = find_builtin_swebench_yaml()
    env = os.environ.copy()
    env["MSWEA_SILENT_STARTUP"] = "1"
    env["LITELLM_MODEL_REGISTRY_PATH"] = str(registry.resolve())
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    # Help UTF-8 on Windows consoles when Rich prints
    env.setdefault("PYTHONIOENCODING", "utf-8")

    cmd: list[str] = [
        sys.executable,
        "-m",
        "minisweagent.run.benchmarks.swebench",
        "--subset",
        args.subset,
        "--split",
        args.split,
        "--output",
        str(args.output.resolve()),
        "--workers",
        str(args.workers),
        "-m",
        args.model,
        "-c",
        str(builtin_sw),
        "-c",
        str(ollama_cfg.resolve()),
        "-c",
        f"agent.step_limit={args.step_limit}",
    ]
    if not args.all_instances and args.slice:
        cmd.extend(["--slice", args.slice])
    if args.redo_existing:
        cmd.append("--redo-existing")

    print("Command:", " ".join(cmd))
    if args.dry_run:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, cwd=str(root), env=env)
    return int(r.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
