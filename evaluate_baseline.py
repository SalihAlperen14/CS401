#!/usr/bin/env python3
"""
Evaluate mini-SWE-agent preds.json with the official SWE-bench harness.

The harness runs FAIL_TO_PASS and PASS_TO_PASS tests inside Docker; an instance is
*resolved* only when both F2P and P2P criteria are met (see swebench.harness.grading).

Usage:
  python evaluate_baseline.py --preds baseline_results/preds.json --run-id my_baseline_v1

Windows note: model names with ':' can break log paths. Use --sanitize-model-name to
rewrite model_name_or_path in a temporary JSON for evaluation only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def load_preds(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("preds.json must be a JSON object mapping instance_id -> prediction")
    return data


def sanitize_preds(preds: dict[str, Any], replacement: str) -> dict[str, Any]:
    """Replace model_name_or_path for filesystem-safe harness output on Windows."""
    out: dict[str, Any] = {}
    for iid, p in preds.items():
        p = dict(p)
        p["model_name_or_path"] = replacement
        out[iid] = p
    return out


def find_reports(run_id: str) -> list[Path]:
    base = Path("logs") / "run_evaluation" / run_id
    if not base.is_dir():
        return []
    return sorted(base.rglob("report.json"))


def summarize_report(report_path: Path) -> dict[str, Any]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    # report.json is keyed by instance_id
    instance_id = next(iter(data.keys()))
    block = data[instance_id]
    summary: dict[str, Any] = {
        "instance_id": instance_id,
        "resolved": block.get("resolved", False),
        "patch_exists": block.get("patch_exists"),
        "patch_successfully_applied": block.get("patch_successfully_applied"),
    }
    ts = block.get("tests_status")
    if isinstance(ts, dict):
        f2p = ts.get("FAIL_TO_PASS") or {}
        p2p = ts.get("PASS_TO_PASS") or {}
        summary["fail_to_pass"] = {
            "success": list(f2p.get("success", [])),
            "failure": list(f2p.get("failure", [])),
        }
        summary["pass_to_pass"] = {
            "success": list(p2p.get("success", [])),
            "failure": list(p2p.get("failure", [])),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run SWE-bench harness on preds.json")
    p.add_argument(
        "--preds",
        type=Path,
        default=Path("baseline_results/preds.json"),
        help="Path to preds.json from mini-SWE-agent",
    )
    p.add_argument(
        "--dataset",
        "-d",
        default="princeton-nlp/SWE-Bench_Verified",
        help="Must match mini-SWE-agent subset (verified -> princeton-nlp/SWE-Bench_Verified on HF)",
    )
    p.add_argument("--split", "-s", default="test")
    p.add_argument("--run-id", "-id", required=True, help="Unique run id for logs")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--timeout", "-t", type=int, default=1800)
    p.add_argument("--cache-level", default="env", choices=["none", "base", "env", "instance"])
    p.add_argument(
        "--sanitize-model-name",
        action="store_true",
        help="Rewrite model_name_or_path to a safe string (recommended on Windows)",
    )
    p.add_argument(
        "--safe-model-name",
        default="ollama_qwen2.5-coder-14b",
        help="Replacement model name when --sanitize-model-name is used",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Write JSON summary here (default: evaluation_summary_<run_id>.json)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print harness command only",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preds_path = args.preds
    if not args.dry_run and not preds_path.is_file():
        print(f"Predictions file not found: {preds_path}", file=sys.stderr)
        return 2

    if args.dry_run:
        preds_data = {}
    else:
        preds_data = load_preds(preds_path)
    eval_path: Path = preds_path
    tmp_dir: tempfile.TemporaryDirectory[str] | None = None

    if not args.dry_run and args.sanitize_model_name:
        tmp_dir = tempfile.TemporaryDirectory(prefix="swebench_preds_")
        eval_path = Path(tmp_dir.name) / "preds.json"
        sanitized = sanitize_preds(preds_data, args.safe_model_name)
        eval_path.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "-d",
        args.dataset,
        "-s",
        args.split,
        "-p",
        str(eval_path.resolve()),
        "--max_workers",
        str(args.max_workers),
        "-t",
        str(args.timeout),
        "--cache_level",
        args.cache_level,
        "-id",
        args.run_id,
    ]

    print("Harness command:", " ".join(cmd))
    if args.dry_run:
        if tmp_dir:
            tmp_dir.cleanup()
        return 0

    code = 1
    try:
        proc = subprocess.run(cmd)
        code = int(proc.returncode)
    finally:
        if tmp_dir:
            tmp_dir.cleanup()

    if code != 0:
        return code

    reports = find_reports(args.run_id)
    per_instance = [summarize_report(r) for r in reports]
    resolved_n = sum(1 for x in per_instance if x.get("resolved"))
    summary = {
        "run_id": args.run_id,
        "dataset": args.dataset,
        "split": args.split,
        "predictions_file": str(preds_path),
        "total_reports": len(per_instance),
        "resolved_count": resolved_n,
        "resolved_rate": (resolved_n / len(per_instance)) if per_instance else 0.0,
        "instances": per_instance,
    }

    out = args.summary_out or Path(f"evaluation_summary_{args.run_id}.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
