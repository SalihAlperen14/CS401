"""
swebench_loader.py - Load and pre-process SWE-bench instances.

Each instance contains:
  - instance_id   : unique identifier
  - repo          : GitHub repo slug (e.g. "astropy/astropy")
  - base_commit   : the commit SHA *before* the fix
  - patch         : the ground-truth fix diff
  - test_patch    : the test diff that catches the bug
  - problem_statement : natural-language description of the issue
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from datasets import load_dataset

from config import SWEBENCH_DATASET, SWEBENCH_SPLIT, MAX_INSTANCES

logger = logging.getLogger(__name__)


@dataclass
class SWEInstance:
    instance_id: str
    repo: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str = ""
    # Parsed from `patch` at load time
    changed_files: List[str] = field(default_factory=list)


def _extract_changed_files(patch: str) -> List[str]:
    """Return the list of Python files touched by a unified diff."""
    files: List[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:].strip()
            if path.endswith(".py") and path not in files:
                files.append(path)
    return files


def load_swebench_instances(
    n: Optional[int] = None,
    instance_ids: Optional[List[str]] = None,
) -> List[SWEInstance]:
    """
    Load SWE-bench instances from HuggingFace.

    Args:
        n:            Maximum number of instances to return.
        instance_ids: If given, only return those specific instance ids.

    Returns:
        List of SWEInstance dataclasses.
    """
    logger.info("Loading SWE-bench dataset '%s' (split=%s) …", SWEBENCH_DATASET, SWEBENCH_SPLIT)
    ds = load_dataset(SWEBENCH_DATASET, split=SWEBENCH_SPLIT)

    instances: List[SWEInstance] = []
    limit = n or MAX_INSTANCES

    for row in ds:
        if instance_ids and row["instance_id"] not in instance_ids:
            continue

        inst = SWEInstance(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            patch=row.get("patch", ""),
            test_patch=row.get("test_patch", ""),
            problem_statement=row.get("problem_statement", ""),
            hints_text=row.get("hints_text", ""),
            changed_files=_extract_changed_files(row.get("patch", "")),
        )
        instances.append(inst)

        if len(instances) >= limit:
            break

    logger.info("Loaded %d SWE-bench instances.", len(instances))
    return instances
