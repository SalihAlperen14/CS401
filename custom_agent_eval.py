#!/usr/bin/env python3
"""
Custom SWE-agent evaluation pipeline (no memory / RAG — pure baseline).

For each SWE-bench Verified instance the script:
  1. Spins up a Docker container from the official SWE-bench instance image.
  2. Runs a simple agent loop (Ollama qwen2.5-coder:14b) that can execute
     bash commands inside the container to explore + fix the issue.
  3. Extracts the agent's patch via `git diff`.
  4. Resets the repo, applies the patch, and runs the FAIL_TO_PASS /
     PASS_TO_PASS tests to decide whether the instance is *resolved*.

IMPORTANT: The agent never sees FAIL_TO_PASS / PASS_TO_PASS lists;
           evaluation is strictly post-hoc.

Usage:
  python custom_agent_eval.py --num 1                       # first instance
  python custom_agent_eval.py --num 5 --max-steps 15        # first 5
  python custom_agent_eval.py --instance-ids sympy__sympy-20590
  python custom_agent_eval.py --dry-run                     # show config

Requirements:
  pip install datasets ollama
  Docker Desktop running
  ollama pull qwen2.5-coder:14b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path("eval_logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("swe_eval")

# SWE-bench Docker images use conda env `testbed`; `docker exec` does not activate conda,
# so bare `pytest` is often missing from PATH. Use this interpreter explicitly.
# Override if your image differs: set SWE_TESTBED_PYTHON=/path/to/python
TESTBED_PYTHON = os.environ.get("SWE_TESTBED_PYTHON", "/opt/miniconda3/envs/testbed/bin/python")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class InstanceResult:
    instance_id: str
    resolved: bool = False
    patch: str = ""
    fail_to_pass_passed: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_passed: int = 0
    pass_to_pass_total: int = 0
    agent_steps: int = 0
    error: str = ""


@dataclass
class RunSummary:
    total: int = 0
    resolved_count: int = 0
    resolution_rate: float = 0.0
    avg_f2p_rate: float = 0.0
    avg_p2p_rate: float = 0.0
    instances: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------
def _swebench_image_name(instance_id: str) -> str:
    docker_safe = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_safe}:latest".lower()


def docker_pull(image: str, timeout: int = 900) -> None:
    log.info("Pulling image %s (may take minutes on first run) ...", image)
    subprocess.run(["docker", "pull", image], check=True, timeout=timeout)


class DockerSandbox:
    """Thin wrapper: start / exec / stop a container."""

    def __init__(self, image: str, workdir: str = "/testbed"):
        self.image = image
        self.workdir = workdir
        self.name = f"swe_eval_{uuid.uuid4().hex[:8]}"
        self._start()

    def _start(self) -> None:
        cmd = [
            "docker", "run", "-d",
            "--name", self.name,
            "-w", self.workdir,
            "--rm",
            self.image,
            "sleep", "2h",
        ]
        log.info("Starting container %s ...", self.name)
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)

    def exec(self, command: str, timeout: int = 180) -> tuple[str, int]:
        """Run *command* in the container; return (output, returncode)."""
        cmd = [
            "docker", "exec",
            "-w", self.workdir,
            "-e", "PAGER=cat",
            "-e", "PIP_PROGRESS_BAR=off",
            self.name,
            "bash", "-c", command,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "[COMMAND TIMED OUT]", 1
        out = (r.stdout + r.stderr).strip()
        if len(out) > 8000:
            out = out[:4000] + "\n\n... [TRUNCATED] ...\n\n" + out[-4000:]
        return out or "[no output]", r.returncode

    def close(self) -> None:
        subprocess.run(
            ["docker", "rm", "-f", self.name],
            capture_output=True, timeout=30,
        )
        log.info("Removed container %s", self.name)


# ---------------------------------------------------------------------------
# Agent (no memory — pure baseline)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an autonomous AI software engineer. Your task is to fix the GitHub
issue described below.

ENVIRONMENT RULES
- You have a bash terminal. The repository is in /testbed.
- Explore the code, find the bug, and edit the files to fix it.
- To run a command, output it inside a single ```bash ... ``` block.
- Output exactly ONE bash block per turn (no more).
- After fixing the bug, output the exact string: COMPLETE_TASK_AND_SUBMIT
  (with NO bash block in that final message).

RECOMMENDED WORKFLOW
1. Read relevant source files (grep, find, cat).
2. Write a small script to reproduce the bug.
3. Edit source code with sed / cat-heredoc.
4. Re-run the reproduction script to confirm the fix.
5. Output COMPLETE_TASK_AND_SUBMIT.
"""


def run_agent(
    sandbox: DockerSandbox,
    problem_statement: str,
    model: str = "qwen2.5-coder:14b",
    max_steps: int = 10,
) -> tuple[str, int]:
    """Drive the agent loop. Returns (patch_text, steps_taken)."""
    import ollama as _ollama

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"ISSUE:\n{problem_statement}"},
    ]

    for step in range(1, max_steps + 1):
        log.info("  Step %d/%d — querying model ...", step, max_steps)

        try:
            resp = _ollama.chat(model=model, messages=messages)
            assistant_text: str = resp["message"]["content"]
        except Exception as e:
            log.error("  Model error: %s", e)
            messages.append({"role": "assistant", "content": f"[model error: {e}]"})
            break

        log.info("  Agent output (first 300 chars): %s", assistant_text[:300])
        messages.append({"role": "assistant", "content": assistant_text})

        if "COMPLETE_TASK_AND_SUBMIT" in assistant_text:
            log.info("  Agent signalled completion at step %d.", step)
            break

        bash_match = re.search(r"```bash\n(.*?)```", assistant_text, re.DOTALL)
        if bash_match:
            cmd = bash_match.group(1).strip()
            log.info("  Executing: %s", cmd[:200])
            observation, rc = sandbox.exec(cmd)
            log.info("  Return code: %d | output (first 300): %s", rc, observation[:300])
            messages.append({"role": "user", "content": f"OBSERVATION (rc={rc}):\n{observation}"})
        else:
            hint = (
                "You did not provide a ```bash ... ``` block. "
                "Please provide exactly one bash block, or say COMPLETE_TASK_AND_SUBMIT."
            )
            messages.append({"role": "user", "content": hint})

    patch_text, _ = sandbox.exec(
        "cd /testbed && git diff --no-color"
    )
    if patch_text == "[no output]":
        patch_text = ""

    traj_path = LOG_DIR / "trajectories"
    traj_path.mkdir(exist_ok=True)
    (traj_path / f"messages_{uuid.uuid4().hex[:8]}.json").write_text(
        json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return patch_text, step


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _parse_pytest_results(output: str) -> dict[str, str]:
    """Parse pytest -v output into {test_id: PASSED|FAILED|ERROR}."""
    results: dict[str, str] = {}
    for line in output.splitlines():
        m = re.match(r"^(.*?)\s+(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)", line)
        if m:
            test_id = m.group(1).strip()
            status = m.group(2).strip()
            results[test_id] = status
    return results


def _normalize_test_id(tid: str) -> str:
    """Strip leading ./ or path prefix normalisation quirks."""
    return tid.lstrip("./")


def run_tests_in_container(
    sandbox: DockerSandbox,
    test_ids: list[str],
    timeout: int = 300,
    testbed_python: str | None = None,
) -> dict[str, str]:
    """Run each test ID individually; return {test_id: PASSED|FAILED|ERROR|NOT_FOUND}."""
    if not test_ids:
        return {}

    py = testbed_python or TESTBED_PYTHON
    results: dict[str, str] = {}
    all_outputs: list[str] = []

    for tid in test_ids:
        quoted = shlex.quote(tid)
        cmd = f"cd /testbed && {py} -m pytest -xvs {quoted} 2>&1"
        out, rc = sandbox.exec(cmd, timeout=timeout)
        all_outputs.append(f"=== {tid} (rc={rc}) ===\n{out}\n")

        if "command not found" in out or "No module named pytest" in out:
            log.error("  pytest unavailable; set SWE_TESTBED_PYTHON if needed.")
            results[tid] = "ERROR"
            continue

        if "ERROR: not found" in out or "no tests ran" in out:
            results[tid] = "NOT_FOUND"
            continue

        parsed = _parse_pytest_results(out)
        if parsed:
            status = next(iter(parsed.values()))
            results[tid] = status
        elif rc == 0:
            results[tid] = "PASSED"
        else:
            results[tid] = "FAILED"

        log.info("  test %s -> %s (rc=%d)", tid, results[tid], rc)

    log.info("  Ran %d test(s): %s", len(test_ids), dict(results))

    (LOG_DIR / "test_outputs").mkdir(exist_ok=True)
    (LOG_DIR / "test_outputs" / f"tests_{uuid.uuid4().hex[:8]}.txt").write_text(
        "\n".join(all_outputs), encoding="utf-8"
    )
    return results


def evaluate_patch(
    sandbox: DockerSandbox,
    patch: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    testbed_python: str | None = None,
) -> InstanceResult:
    """Reset repo, apply patch, run F2P + P2P tests, compute metrics."""
    result = InstanceResult(instance_id="", fail_to_pass_total=len(fail_to_pass), pass_to_pass_total=len(pass_to_pass))

    sandbox.exec("cd /testbed && git checkout -- . && git clean -fd")

    if patch.strip():
        apply_cmd = f"cd /testbed && git apply --allow-empty - <<'ENDOFPATCH'\n{patch}\nENDOFPATCH"
        out, rc = sandbox.exec(apply_cmd)
        if rc != 0:
            log.warning("  Patch apply failed: %s", out[:500])
            result.error = f"patch_apply_failed: {out[:500]}"
            return result
    else:
        # Empty patch: no `git apply`; working tree stays exactly at base_commit (after checkout above).
        result.error = "empty_patch"

    if fail_to_pass:
        f2p_results = run_tests_in_container(sandbox, fail_to_pass, testbed_python=testbed_python)
        for tid in fail_to_pass:
            status = f2p_results.get(tid, "NOT_FOUND")
            if status in ("PASSED", "XFAIL", "XPASS"):
                result.fail_to_pass_passed += 1

    if pass_to_pass:
        p2p_results = run_tests_in_container(sandbox, pass_to_pass, testbed_python=testbed_python)
        for tid in pass_to_pass:
            status = p2p_results.get(tid, "NOT_FOUND")
            if status in ("PASSED", "XFAIL", "XPASS"):
                result.pass_to_pass_passed += 1

    result.resolved = (
        result.fail_to_pass_passed == result.fail_to_pass_total
        and result.pass_to_pass_passed == result.pass_to_pass_total
        and result.fail_to_pass_total > 0
    )
    return result


# ---------------------------------------------------------------------------
# Pipeline: run one instance end-to-end
# ---------------------------------------------------------------------------
def process_instance(
    instance: dict,
    model: str,
    max_steps: int,
    pull_timeout: int,
    force_empty_patch: bool = False,
    testbed_python: str | None = None,
) -> InstanceResult:
    iid = instance["instance_id"]
    log.info("=" * 60)
    log.info("Processing %s", iid)
    log.info("=" * 60)

    image = _swebench_image_name(iid)

    try:
        docker_pull(image, timeout=pull_timeout)
    except Exception as e:
        log.error("Image pull failed for %s: %s", iid, e)
        return InstanceResult(instance_id=iid, error=f"image_pull_failed: {e}")

    sandbox = DockerSandbox(image)
    try:
        problem = instance["problem_statement"]

        if force_empty_patch:
            log.info("  force_empty_patch=True: skipping agent loop, using empty patch.")
            patch, steps = "", 0
        else:
            patch, steps = run_agent(sandbox, problem, model=model, max_steps=max_steps)
        log.info("  Agent finished in %d steps, patch length: %d chars", steps, len(patch))

        f2p_raw = instance["FAIL_TO_PASS"]
        p2p_raw = instance["PASS_TO_PASS"]
        fail_to_pass: list[str] = json.loads(f2p_raw) if isinstance(f2p_raw, str) else list(f2p_raw)
        pass_to_pass: list[str] = json.loads(p2p_raw) if isinstance(p2p_raw, str) else list(p2p_raw)

        result = evaluate_patch(
            sandbox, patch, fail_to_pass, pass_to_pass, testbed_python=testbed_python
        )
        result.instance_id = iid
        result.patch = patch
        result.agent_steps = steps

        log.info(
            "  Result: resolved=%s | F2P=%d/%d | P2P=%d/%d",
            result.resolved,
            result.fail_to_pass_passed,
            result.fail_to_pass_total,
            result.pass_to_pass_passed,
            result.pass_to_pass_total,
        )
        return result

    except Exception as e:
        log.exception("Unhandled error for %s", iid)
        return InstanceResult(instance_id=iid, error=str(e))
    finally:
        sandbox.close()


# ---------------------------------------------------------------------------
# CLI + batch orchestration
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Custom SWE-agent eval pipeline (no memory)")
    p.add_argument("--num", "-n", type=int, default=1, help="Number of instances to evaluate")
    p.add_argument("--offset", type=int, default=0, help="Start index in the dataset")
    p.add_argument("--instance-ids", nargs="+", default=[], help="Specific instance IDs to run")
    p.add_argument("--model", "-m", default="qwen2.5-coder:14b", help="Ollama model name")
    p.add_argument("--max-steps", type=int, default=10, help="Agent loop iterations")
    p.add_argument("--pull-timeout", type=int, default=900, help="Docker pull timeout (s)")
    p.add_argument("--output", "-o", type=Path, default=Path("custom_eval_results"), help="Output dir")
    p.add_argument(
        "--testbed-python",
        default=os.environ.get("SWE_TESTBED_PYTHON", TESTBED_PYTHON),
        help="Python inside SWE-bench Docker image for pytest (default: conda env testbed)",
    )
    p.add_argument(
        "--force-empty-patch",
        action="store_true",
        help="Skip agent run and evaluate with an empty patch (pipeline sanity check)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print config and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    log.info("Config: %s", vars(args))
    if args.dry_run:
        return 0

    from datasets import load_dataset
    ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    log.info("Dataset loaded: %d instances total", len(ds))

    if args.instance_ids:
        instances = [inst for inst in ds if inst["instance_id"] in args.instance_ids]
        if not instances:
            log.error("None of %s found in dataset", args.instance_ids)
            return 1
    else:
        instances = list(ds.select(range(args.offset, min(args.offset + args.num, len(ds)))))

    log.info("Running %d instance(s) ...", len(instances))

    args.output.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, str]] = []
    all_results: list[InstanceResult] = []

    for inst in instances:
        result = process_instance(
            inst,
            args.model,
            args.max_steps,
            args.pull_timeout,
            force_empty_patch=args.force_empty_patch,
            testbed_python=args.testbed_python,
        )
        all_results.append(result)

        predictions.append({
            "instance_id": result.instance_id,
            "model_name_or_path": args.model,
            "model_patch": result.patch,
        })

        preds_path = args.output / "predictions.jsonl"
        with open(preds_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(predictions[-1], ensure_ascii=False) + "\n")

    resolved = [r for r in all_results if r.resolved]
    total = len(all_results)

    f2p_rates = [
        r.fail_to_pass_passed / r.fail_to_pass_total
        for r in all_results if r.fail_to_pass_total > 0
    ]
    p2p_rates = [
        r.pass_to_pass_passed / r.pass_to_pass_total
        for r in all_results if r.pass_to_pass_total > 0
    ]

    summary = RunSummary(
        total=total,
        resolved_count=len(resolved),
        resolution_rate=len(resolved) / total if total else 0.0,
        avg_f2p_rate=sum(f2p_rates) / len(f2p_rates) if f2p_rates else 0.0,
        avg_p2p_rate=sum(p2p_rates) / len(p2p_rates) if p2p_rates else 0.0,
        instances=[asdict(r) for r in all_results],
    )

    results_path = args.output / "results.json"
    results_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Results written to %s", results_path)

    print("\n" + "=" * 60)
    print(f"  TOTAL: {total}  |  RESOLVED: {len(resolved)}  |  RATE: {summary.resolution_rate:.1%}")
    print(f"  Avg F2P rate: {summary.avg_f2p_rate:.1%}  |  Avg P2P rate: {summary.avg_p2p_rate:.1%}")
    print("=" * 60)

    for r in all_results:
        tag = "RESOLVED" if r.resolved else "FAILED"
        print(f"  [{tag}] {r.instance_id}  F2P={r.fail_to_pass_passed}/{r.fail_to_pass_total}  P2P={r.pass_to_pass_passed}/{r.pass_to_pass_total}  err={r.error or '-'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
