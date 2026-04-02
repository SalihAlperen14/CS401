"""
main.py - Entry point for the SWE-bench bug injection pipeline.

Usage
-----
# Inject bugs into the first 3 SWE-bench test instances
python main.py --n 3

# Target specific instances
python main.py --ids astropy__astropy-12907 django__django-11019

# Use a custom number and a specific split
python main.py --n 5 --split dev
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.progress import track

from config import MAX_INSTANCES, REPORTS_DIR, GITHUB_TOKEN
from swebench_loader import load_swebench_instances
from repo_manager import RepoManager
from github_fetcher import GitHubFetcher
from agents import OrchestratorAgent

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SWE-bench bug injection pipeline")
    parser.add_argument("--n",     type=int,  default=MAX_INSTANCES, help="Number of instances to process")
    parser.add_argument("--split", type=str,  default="test",        help="SWE-bench split: train|test|dev")
    parser.add_argument("--ids",   nargs="+", default=None,          help="Specific instance IDs to process")
    parser.add_argument("--skip-clone", action="store_true",         help="Skip repo setup (use existing clones)")
    parser.add_argument("--remote",     action="store_true",         help="Fetch files via GitHub API — no local clone (uses GITHUB_TOKEN env var)")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Load instances
    console.rule("[bold cyan]SWE-bench Bug Injection Pipeline[/]")
    instances = load_swebench_instances(n=args.n, instance_ids=args.ids)
    console.print(f"Loaded [bold]{len(instances)}[/] instances from SWE-bench [{args.split}]")

    orchestrator = OrchestratorAgent()
    reports = []

    if args.remote:
        console.print("[bold cyan]Mode:[/] GitHub API (no local clone)")
        fetcher = GitHubFetcher(token=GITHUB_TOKEN or None)

        # 2a. Process each instance using GitHub API
        for instance in track(instances, description="Injecting bugs (remote) …"):
            console.print(f"\n[bold yellow]▶ {instance.instance_id}[/] ({instance.repo})")
            try:
                report = orchestrator.run_remote(instance, fetcher)
            except Exception as exc:
                logger.exception("Unexpected error for %s", instance.instance_id)
                report = {
                    "instance_id": instance.instance_id,
                    "status": "error",
                    "error": str(exc),
                }
            reports.append(report)
            _print_report(report)

    else:
        console.print("[bold cyan]Mode:[/] local clone")

        # 2b. Process each instance using local git clone
        for instance in track(instances, description="Injecting bugs …"):
            console.print(f"\n[bold yellow]▶ {instance.instance_id}[/] ({instance.repo})")

            repo_manager = RepoManager(instance)

            try:
                if not args.skip_clone:
                    repo_manager.setup()
                report = orchestrator.run(instance, repo_manager)
            except Exception as exc:
                logger.exception("Unexpected error for %s", instance.instance_id)
                report = {
                    "instance_id": instance.instance_id,
                    "status": "error",
                    "error": str(exc),
                }
            finally:
                # Always reset so partial mutations don't pollute subsequent runs
                try:
                    repo_manager.reset()
                except Exception:
                    pass

            reports.append(report)
            _print_report(report)

    # 3. Save aggregated report
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(REPORTS_DIR) / f"run_{timestamp}.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    console.print(f"\n[bold green]✓[/] Report saved → {report_path}")

    # 4. Summary table
    _print_summary(reports)


def _print_report(report: dict) -> None:
    status = report.get("status", "unknown")
    color  = "green" if status == "success" else "red"
    console.print(
        f"  Status: [{color}]{status}[/]  |  "
        f"Strategy: [cyan]{report.get('strategy', 'N/A')}[/]  |  "
        f"File: [dim]{report.get('target_file', 'N/A')}[/]  |  "
        f"Score: [bold]{report.get('score', '-')}[/]"
    )
    if report.get("validation"):
        console.print(f"  Feedback: [italic]{report['validation'].get('feedback', '')}[/]")


def _print_summary(reports: list) -> None:
    console.rule("[bold]Summary[/]")
    success = sum(1 for r in reports if r.get("status") == "success")
    total   = len(reports)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Instance ID", style="cyan",  no_wrap=True)
    table.add_column("Status",      style="white")
    table.add_column("Strategy",    style="yellow")
    table.add_column("Score",       style="green",  justify="center")
    table.add_column("File",        style="dim",    no_wrap=False)

    for r in reports:
        status_str = "[green]✓ success[/]" if r.get("status") == "success" else "[red]✗ failed[/]"
        table.add_row(
            r.get("instance_id", ""),
            status_str,
            r.get("strategy", ""),
            str(r.get("score", "")),
            r.get("target_file", ""),
        )

    console.print(table)
    console.print(f"\n[bold]{success}/{total}[/] instances successfully injected.")


if __name__ == "__main__":
    main()
