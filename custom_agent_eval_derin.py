#!/usr/bin/env python3
"""
Custom SWE-agent evaluation pipeline (no memory / RAG — pure baseline).

For each SWE-bench Verified instance the script:
  1. Spins up a Docker container from the official SWE-bench instance image.
  2. Runs a simple agent loop (local Ollama model) that can execute
     bash commands inside the container to explore + fix the issue.
  3. Extracts the agent's patch via `git diff`.
  4. Resets the repo, applies the SWE-bench test_patch (adds F2P/P2P test
     nodes), then applies the model patch, and runs FAIL_TO_PASS /
     PASS_TO_PASS tests to decide whether the instance is *resolved*.

IMPORTANT: The agent never sees FAIL_TO_PASS / PASS_TO_PASS lists;
           evaluation is strictly post-hoc.

Usage:
  python custom_agent_eval_derin.py --num 1                          # first instance
  python custom_agent_eval_derin.py --num 5 --max-steps 15           # first 5
  python custom_agent_eval_derin.py --instance-ids sympy__sympy-20590
  python custom_agent_eval_derin.py --dry-run                        # show config

Requirements:
  pip install datasets ollama
  Docker Desktop running
  ollama pull qwen2.5-coder:14b        (or whichever model you pass via --model)
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

    def __init__(
        self,
        image: str,
        workdir: str = "/testbed",
        testbed_python: str | None = None,
    ):
        self.image = image
        self.workdir = workdir
        self.name = f"swe_eval_{uuid.uuid4().hex[:8]}"
        py = testbed_python or TESTBED_PYTHON
        # POSIX path for Linux container (avoid Windows Path backslashes in PATH=).
        self._testbed_bin = py.rsplit("/", 1)[0]
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
        # Prepend conda testbed bin so `python`/`pytest` match SWE-bench (avoids system Python w/o package).
        path_env = f"{self._testbed_bin}:/usr/local/bin:/usr/bin:/bin"
        cmd = [
            "docker", "exec",
            "-w", self.workdir,
            "-e", f"PATH={path_env}",
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
You are an autonomous AI software engineer. You will receive a GitHub issue \
description and must produce a minimal source-code fix inside the repository.

═══════════════════════════════════════════════════
 ENVIRONMENT
═══════════════════════════════════════════════════
• The repository is checked out at /testbed. It is already installed and \
importable — `python` points to the SWE-bench conda testbed.
• Do NOT run `pip install <project>` or `pip install -e .`; that would \
shadow /testbed and break evaluation.
• Interactive editors (nano, vim, emacs) are NOT available. Use `sed`, `ed`, \
or shell heredocs (`cat <<'EOF' > file`) to edit files.
• You may create small throwaway scripts (e.g. /tmp/repro.py) to reproduce \
the bug, but the actual fix MUST be edits to tracked files under /testbed \
so that `git diff` is non-empty.

═══════════════════════════════════════════════════
 COMMUNICATION PROTOCOL
═══════════════════════════════════════════════════
• To run a shell command, output it inside exactly ONE ```bash ... ``` block.
• Output only ONE bash block per turn — no more.
• Do NOT include COMPLETE_TASK_AND_SUBMIT in the same message as a bash block.
• When you are done, send a message whose ONLY content is the line:
      COMPLETE_TASK_AND_SUBMIT
  (no other text, no bash block).

═══════════════════════════════════════════════════
 RECOMMENDED WORKFLOW  (follow this order)
═══════════════════════════════════════════════════

STEP 1 — UNDERSTAND THE ISSUE
  Read the issue carefully. Identify: which module / class / function is \
mentioned? What is the expected vs. actual behaviour?

STEP 2 — LOCATE THE RELEVANT CODE
  Use grep / find to narrow down. Examples:
    grep -rn "function_name" /testbed/<package> --include="*.py"
    find /testbed -type f -name "*.py" | xargs grep -l "ClassName"
  Then cat the file(s) to read the surrounding logic.

STEP 3 — REPRODUCE THE BUG
  Write a small script to /tmp/repro.py and run it with `python /tmp/repro.py`.
  Confirm you see the wrong behaviour described in the issue.

STEP 4 — FIND THE ROOT CAUSE
  Read the code paths you found in Step 2. Identify the exact line(s) that \
cause the wrong behaviour. Think about edge cases.

STEP 5 — APPLY A MINIMAL FIX
  Edit only the necessary line(s) in tracked source files. Use `sed -i` for \
single-line changes, or a heredoc/patch for multi-line changes. Examples:
    sed -i 's/old_code/new_code/' /testbed/pkg/module.py
    # or for multi-line:
    cat <<'PATCH' | patch -p1 -d /testbed
    --- a/pkg/module.py
    +++ b/pkg/module.py
    @@ -42,3 +42,3 @@
    -        wrong_line
    +        fixed_line
    PATCH

STEP 6 — VERIFY THE FIX
  Re-run your reproduction script. Confirm the output is now correct.

STEP 7 — CONFIRM GIT DIFF
  Run `git diff` to make sure your changes are to tracked files and the diff \
is non-empty. If git diff is empty you have NOT fixed anything.

STEP 8 — SUBMIT
  Send a message containing only:
      COMPLETE_TASK_AND_SUBMIT

═══════════════════════════════════════════════════
 COMMON MISTAKES TO AVOID
═══════════════════════════════════════════════════
✗ pip install / pip install -e . (shadows /testbed source)
✗ Editing only untracked files (git diff will be empty → no patch)
✗ Using nano/vim (not installed, command will hang)
✗ Sending COMPLETE_TASK_AND_SUBMIT before git diff is non-empty
✗ Sending COMPLETE_TASK_AND_SUBMIT together with a bash block
✗ Making unnecessary changes to unrelated files
"""


def _has_tracked_diff_for_patch(sandbox: DockerSandbox) -> bool:
    """
    True if `git diff` / `git diff --cached` would be non-empty (tracked file edits).

    `git status --porcelain` alone is wrong: untracked files (e.g. test_separability.py)
    make porcelain non-empty while `git diff` stays empty, so the saved patch is blank.
    """
    _, rc_worktree = sandbox.exec("cd /testbed && git diff --quiet")
    _, rc_index = sandbox.exec("cd /testbed && git diff --cached --quiet")
    # git diff --quiet: exit 1 if there are differences, 0 if none
    return rc_worktree != 0 or rc_index != 0


def _agent_wants_submit(assistant_text: str) -> bool:
    """
    True only when the model clearly submits completion, not when it quotes instructions.

    Accepts:
      - whole message exactly COMPLETE_TASK_AND_SUBMIT (after strip)
      - last non-empty line exactly COMPLETE_TASK_AND_SUBMIT, with no ``` fences in the message
    """
    t = assistant_text.strip()
    if "```" in t:
        return False
    if t == "COMPLETE_TASK_AND_SUBMIT":
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return bool(lines) and lines[-1] == "COMPLETE_TASK_AND_SUBMIT"


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

        # Prefer executing bash in this turn; do not treat instruction echoes as submit.
        bash_match = re.search(r"```bash\n(.*?)```", assistant_text, re.DOTALL)
        if bash_match:
            cmd = bash_match.group(1).strip()
            log.info("  Executing: %s", cmd[:200])
            observation, rc = sandbox.exec(cmd)
            log.info("  Return code: %d | output (first 300): %s", rc, observation[:300])
            messages.append({"role": "user", "content": f"OBSERVATION (rc={rc}):\n{observation}"})
            continue

        if _agent_wants_submit(assistant_text):
            if _has_tracked_diff_for_patch(sandbox):
                log.info(
                    "  Agent signalled completion at step %d (tracked files differ; patch non-empty).",
                    step,
                )
                break
            log.warning(
                "  Ignoring premature COMPLETE_TASK_AND_SUBMIT: no tracked diff "
                "(untracked-only or clean tree — `git diff` would be empty)."
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "SUBMIT_REJECTED: `git diff` is still empty. Only edits to tracked "
                        "repository files count (new untracked scripts alone are not enough). "
                        "Modify files under /testbed that are already in git, then submit again."
                    ),
                }
            )
            continue

        hint = (
            "Provide exactly one ```bash ... ``` block to run a command, "
            "or send a message whose only content is the line COMPLETE_TASK_AND_SUBMIT "
            "(do not repeat the instructions)."
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

    return patch_text, step  # step == last completed loop index (1..max_steps)


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


def _normalize_patch_text(patch: str) -> str:
    """
    Normalize patch text for `git apply`.

    Do not use str.strip() on whole patches: trailing unified-diff context lines
    are often a single leading space plus newline (` \\n`); stripping removes them
    and corrupts the hunk (e.g. 'corrupt patch at line N').
    """
    if not patch:
        return ""
    s = patch.replace("\r\n", "\n").replace("\r", "\n")
    return s.lstrip("\ufeff")


def _apply_git_patch(sandbox: DockerSandbox, patch: str) -> tuple[bool, str]:
    """
    Apply a unified diff with progressively looser git settings.

    Agent-generated `git diff` often trips strict whitespace checks (trailing
    space on context lines). SWE-bench–style eval still needs the patch to apply
    to run tests.
    """
    attempts: list[tuple[str, str]] = [
        ("--whitespace=nowarn", "suppress trailing-whitespace rejects"),
        ("--ignore-whitespace", "ignore space when matching context (last resort)"),
    ]
    patch = _normalize_patch_text(patch)
    if not patch:
        return True, ""

    last_out = ""
    for flags, why in attempts:
        apply_cmd = f"cd /testbed && git apply {flags} - <<'ENDOFPATCH'\n{patch}\nENDOFPATCH"
        out, rc = sandbox.exec(apply_cmd)
        if rc == 0:
            log.info("  git apply OK (%s)", why)
            return True, ""
        last_out = out
        log.warning("  git apply failed %s: %s", flags, out[:400].replace("\n", " "))
    return False, last_out


def evaluate_patch(
    sandbox: DockerSandbox,
    patch: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    testbed_python: str | None = None,
    test_patch: str | None = None,
) -> InstanceResult:
    """Reset repo, apply SWE-bench test_patch (if any), then model patch, run F2P + P2P tests."""
    result = InstanceResult(instance_id="", fail_to_pass_total=len(fail_to_pass), pass_to_pass_total=len(pass_to_pass))

    sandbox.exec("cd /testbed && git checkout -- . && git clean -fd")

    tp = _normalize_patch_text(test_patch or "")
    if tp.strip():
        ok, err_out = _apply_git_patch(sandbox, tp)
        if not ok:
            log.warning("  test_patch apply failed: %s", err_out[:800])
            result.error = f"test_patch_apply_failed: {err_out[:800]}"
            return result
        log.info("  test_patch applied (F2P/P2P node ids now match dataset).")

    patch_norm = _normalize_patch_text(patch)
    if patch_norm.strip():
        # No --allow-empty: older git in SWE-bench images does not support it (empty patches are skipped above).
        ok, err_out = _apply_git_patch(sandbox, patch_norm)
        if not ok:
            log.warning("  Patch apply failed after retries: %s", err_out[:800])
            result.error = f"patch_apply_failed: {err_out[:800]}"
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

    sandbox = DockerSandbox(image, testbed_python=testbed_python)
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
        test_patch_raw = instance.get("test_patch") or ""
        test_patch_str = test_patch_raw if isinstance(test_patch_raw, str) else str(test_patch_raw)

        result = evaluate_patch(
            sandbox,
            patch,
            fail_to_pass,
            pass_to_pass,
            testbed_python=testbed_python,
            test_patch=test_patch_str,
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