"""
agents.py - AutoGen agent definitions for the bug injection pipeline.

Agent roles
-----------
OrchestratorAgent
    Drives the overall workflow: picks files, delegates analysis and injection,
    collects results, and writes the final report.

AnalyzerAgent
    Reads a Python file and identifies the best injection point + strategy,
    returning structured JSON.

InjectorAgent
    Applies the chosen strategy (or falls back to an LLM-guided injection)
    and validates the result parses correctly.

ValidatorAgent
    Reviews the injected code and confirms the bug is realistic, non-trivial,
    and would not be immediately caught by a linter.
"""

from __future__ import annotations

import difflib
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from github_fetcher import GitHubFetcher

import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

from config import LLM_CONFIG, ENABLED_BUG_TYPES, MAX_INJECTION_RETRIES, GITHUB_TOKEN
from bug_strategies import STRATEGIES
from repo_manager import RepoManager
from swebench_loader import SWEInstance

logger = logging.getLogger(__name__)


# ── System prompts ─────────────────────────────────────────────────────────────

_ANALYZER_SYSTEM = """
You are an expert Python static-analysis agent working on a bug injection benchmark.

Given the content of a Python source file, you must:
1. Identify the single BEST location in the file to inject a subtle, realistic bug.
2. Invent an appropriate bug type name that precisely describes the mutation (e.g.
   "swap_loop_bounds", "negate_base_case", "drop_accumulator_reset").
3. Briefly explain what change to make and why a developer might make that mistake.

Do NOT limit yourself to a fixed list of categories — choose whatever bug fits
the code's logic best and would be hard to spot in a code review.

Respond ONLY with a JSON object (no markdown fences) with these keys:
{
  "strategy":    "<descriptive_bug_type_name>",
  "line_hint":   <approximate 1-based line number>,
  "explanation": "<one or two sentences: what to change and why it is realistic>"
}
""".strip()

_INJECTOR_SYSTEM = """
You are an expert Python mutation-testing agent.

You will receive:
- A Python source file
- A specific bug type to inject
- An approximate line number to target
- An explanation of why that location was chosen

Your task: inject a subtle, realistic bug of the requested type near the target line.
Rules:
- The mutated file MUST remain syntactically valid Python.
- Change as few characters as possible.
- Do NOT add comments that reveal the bug.
- Do NOT change imports or class/function signatures unless strictly necessary.

Respond ONLY with the complete mutated Python source file, no markdown fences.
""".strip()

_VALIDATOR_SYSTEM = """
You are a strict code-review agent for a bug injection benchmark.

You will receive:
- The original Python source
- The mutated (buggy) Python source
- The claimed bug type

Evaluate:
1. Is the mutation syntactically valid Python?
2. Is the bug subtle enough to evade a simple linter?
3. Is the bug realistic (could a human developer have made this mistake)?
4. Does the bug clearly differ from the original?

Respond ONLY with a JSON object (no markdown fences):
{
  "valid": true | false,
  "score": <integer 1-5>,   // 5 = perfect, 1 = trivially obvious or broken
  "feedback": "<one sentence>"
}
""".strip()


# ── Helper: call the LLM via AutoGen ConversableAgent ─────────────────────────

def _single_turn(system_prompt: str, user_message: str) -> str:
    """
    Run a single-turn conversation with an AssistantAgent and return its reply.
    Uses a silent UserProxyAgent that never asks for human input.
    """
    assistant = AssistantAgent(
        name="assistant",
        system_message=system_prompt,
        llm_config=LLM_CONFIG,
    )
    proxy = UserProxyAgent(
        name="proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
    )
    proxy.initiate_chat(assistant, message=user_message, silent=True)
    # The last message from the assistant
    return assistant.last_message()["content"].strip()


# ── AnalyzerAgent ─────────────────────────────────────────────────────────────

class AnalyzerAgent:
    """Chooses the best file + injection strategy for a SWE-bench instance."""

    def analyze(
        self,
        source: str,
        filename: str,
        enabled_strategies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dict with keys: strategy, line_hint, explanation.

        When *enabled_strategies* is None the agent freely invents a bug type
        suited to the code (remote / API-based workflow).
        When *enabled_strategies* is provided the agent must choose from that list
        (local clone workflow — backward-compatible).

        Falls back to a safe default if the LLM response is malformed.
        """
        if enabled_strategies:
            strategy_hint = f"Available bug types: {enabled_strategies}\n"
        else:
            strategy_hint = (
                "Invent a bug type name that best fits this code's logic — "
                "do not limit yourself to any fixed list.\n"
            )

        prompt = (
            f"File: {filename}\n"
            f"{strategy_hint}\n"
            f"Source code:\n```python\n{source[:3000]}\n```"
        )
        raw = _single_turn(_ANALYZER_SYSTEM, prompt)

        try:
            result = json.loads(raw)
            if not result.get("strategy"):
                raise ValueError("Missing 'strategy' key")
            # When a fixed list was supplied, validate the choice
            if enabled_strategies and result["strategy"] not in enabled_strategies:
                raise ValueError(f"Unknown strategy: {result['strategy']}")
            return result
        except Exception as exc:
            logger.warning("Analyzer LLM response malformed (%s); using fallback.", exc)
            fallback_strategy = random.choice(enabled_strategies) if enabled_strategies else "wrong_operator"
            return {
                "strategy": fallback_strategy,
                "line_hint": 1,
                "explanation": "Fallback: analyzer response was malformed.",
            }


# ── InjectorAgent ─────────────────────────────────────────────────────────────

class InjectorAgent:
    """
    Injects a bug into a source file.

    Strategy priority:
      1. Deterministic AST mutator (fast, always syntactically valid)
      2. LLM-guided injection (flexible but may produce invalid syntax)
    """

    def inject(
        self,
        source: str,
        strategy: str,
        line_hint: int,
        explanation: str,
    ) -> Optional[str]:
        # 1. Try the deterministic AST strategy only when it's a known built-in type.
        #    Agent-chosen free-form strategies won't match any key in STRATEGIES and
        #    will fall through to LLM injection below.
        if strategy in STRATEGIES:
            for attempt in range(MAX_INJECTION_RETRIES):
                result = STRATEGIES[strategy](source)
                if result and result != source:
                    logger.info(
                        "Deterministic injection succeeded (strategy=%s, attempt=%d).",
                        strategy, attempt + 1,
                    )
                    return result
            logger.info(
                "Deterministic injection produced no change for strategy=%s; "
                "falling through to LLM injection.",
                strategy,
            )

        # 2. LLM-guided injection — the primary path for agent-chosen strategies.
        logger.info("Using LLM injection for strategy=%s.", strategy)
        prompt = (
            f"Bug type: {strategy}\n"
            f"Target line (approx): {line_hint}\n"
            f"Reason: {explanation}\n\n"
            f"Source code:\n{source}"
        )
        mutated = _single_turn(_INJECTOR_SYSTEM, prompt)

        # Strip accidental markdown fences
        if mutated.startswith("```"):
            lines = mutated.splitlines()
            mutated = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # Validate syntax
        try:
            compile(mutated, "<string>", "exec")
            return mutated
        except SyntaxError as e:
            logger.error("LLM injection produced invalid syntax: %s", e)
            return None


# ── ValidatorAgent ────────────────────────────────────────────────────────────

class ValidatorAgent:
    """Reviews an injected bug and scores its quality."""

    def validate(
        self,
        original: str,
        mutated: str,
        strategy: str,
    ) -> Dict[str, Any]:
        prompt = (
            f"Bug type: {strategy}\n\n"
            f"=== ORIGINAL ===\n{original[:2000]}\n\n"
            f"=== MUTATED ===\n{mutated[:2000]}"
        )
        raw = _single_turn(_VALIDATOR_SYSTEM, prompt)

        try:
            result = json.loads(raw)
            return result
        except Exception:
            logger.warning("Validator response malformed; assuming valid.")
            return {"valid": True, "score": 3, "feedback": "Could not parse validator response."}


# ── OrchestratorAgent ─────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    High-level controller that drives the full bug injection pipeline
    for a single SWE-bench instance.
    """

    def __init__(self):
        self.analyzer  = AnalyzerAgent()
        self.injector  = InjectorAgent()
        self.validator = ValidatorAgent()

    def run(self, instance: SWEInstance, repo_manager: RepoManager) -> Dict[str, Any]:
        """
        Execute the full pipeline for one SWE-bench instance.

        Returns a report dict.
        """
        logger.info("=== Orchestrating instance: %s ===", instance.instance_id)

        report: Dict[str, Any] = {
            "instance_id": instance.instance_id,
            "repo": instance.repo,
            "base_commit": instance.base_commit,
            "status": "failed",
            "strategy": None,
            "target_file": None,
            "validation": None,
            "patch": None,
        }

        # 1. Collect candidate files (prefer files mentioned in ground-truth patch)
        candidates = instance.changed_files or repo_manager.list_python_files()
        if not candidates:
            report["error"] = "No Python files found."
            return report

        # Filter to files that actually exist
        existing = [f for f in candidates if (Path(repo_manager.repo_dir) / f).exists()]
        if not existing:
            existing = repo_manager.list_python_files()[:10]

        # 2. Pick a file and analyze it
        target_file = random.choice(existing[:5])  # sample from top candidates
        logger.info("Target file: %s", target_file)

        try:
            source = repo_manager.read_file(target_file)
        except Exception as e:
            report["error"] = f"Could not read {target_file}: {e}"
            return report

        # 3. Analyzer: choose strategy
        analysis = self.analyzer.analyze(source, target_file, ENABLED_BUG_TYPES)
        strategy    = analysis["strategy"]
        line_hint   = analysis.get("line_hint", 1)
        explanation = analysis.get("explanation", "")
        logger.info("Strategy chosen: %s (line ~%d)", strategy, line_hint)

        report["strategy"]    = strategy
        report["target_file"] = target_file
        report["analysis"]    = analysis

        # 4. Injector: mutate the file
        mutated = self.injector.inject(source, strategy, line_hint, explanation)
        if mutated is None:
            report["error"] = "Injection failed after all retries."
            return report

        # 5. Write mutated file to disk
        repo_manager.write_file(target_file, mutated)
        logger.info("Mutated source written to %s", target_file)

        # 6. Validator: review quality
        validation = self.validator.validate(source, mutated, strategy)
        report["validation"] = validation

        if not validation.get("valid", False):
            logger.warning("Validator rejected mutation: %s", validation.get("feedback"))
            repo_manager.reset()
            report["error"] = "Validation failed: " + validation.get("feedback", "")
            return report

        # 7. Generate patch
        patch = repo_manager.create_patch(instance.instance_id)
        report["patch"]  = patch
        report["status"] = "success"
        report["score"]  = validation.get("score", 0)

        logger.info(
            "Instance %s completed. Score=%s, Strategy=%s",
            instance.instance_id, report["score"], strategy,
        )
        return report

    def run_remote(self, instance: SWEInstance, fetcher: "GitHubFetcher") -> Dict[str, Any]:
        """
        Execute the full pipeline for one SWE-bench instance using the GitHub API.

        Files are fetched on-demand via *fetcher* — no local clone required.
        The agents freely decide what kind of bug to inject based on the code
        they read (no predetermined strategy list).
        The final patch is computed in-memory with difflib.
        """
        logger.info("=== Orchestrating (remote) instance: %s ===", instance.instance_id)

        report: Dict[str, Any] = {
            "instance_id": instance.instance_id,
            "repo": instance.repo,
            "base_commit": instance.base_commit,
            "status": "failed",
            "strategy": None,
            "target_file": None,
            "validation": None,
            "patch": None,
        }

        # 1. Fetch changed Python files from GitHub at the base (pre-fix) commit
        candidates = fetcher.fetch_changed_files(instance)
        if not candidates:
            report["error"] = "No Python files could be fetched from GitHub."
            return report

        # 2. Pick a target file (prefer files that appear earlier in the patch)
        target_file = random.choice(list(candidates.keys())[:5])
        source = candidates[target_file]
        logger.info("Target file: %s", target_file)

        report["target_file"] = target_file

        # 3. Analyzer: agent freely chooses bug type and injection location
        analysis = self.analyzer.analyze(source, target_file)  # no strategy list
        strategy    = analysis["strategy"]
        line_hint   = analysis.get("line_hint", 1)
        explanation = analysis.get("explanation", "")
        logger.info("Strategy chosen: %s (line ~%d)", strategy, line_hint)

        report["strategy"] = strategy
        report["analysis"] = analysis

        # 4. Injector: mutate the source in-memory
        mutated = self.injector.inject(source, strategy, line_hint, explanation)
        if mutated is None:
            report["error"] = "Injection failed."
            return report

        # 5. Validator: review quality
        validation = self.validator.validate(source, mutated, strategy)
        report["validation"] = validation

        if not validation.get("valid", False):
            logger.warning("Validator rejected mutation: %s", validation.get("feedback"))
            report["error"] = "Validation failed: " + validation.get("feedback", "")
            return report

        # 6. Compute in-memory unified diff (no git required)
        patch = "".join(difflib.unified_diff(
            source.splitlines(keepends=True),
            mutated.splitlines(keepends=True),
            fromfile=f"a/{target_file}",
            tofile=f"b/{target_file}",
        ))

        report["patch"]  = patch
        report["status"] = "success"
        report["score"]  = validation.get("score", 0)

        logger.info(
            "Instance %s completed (remote). Score=%s, Strategy=%s",
            instance.instance_id, report["score"], strategy,
        )
        return report
