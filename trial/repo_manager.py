"""
repo_manager.py - Clone SWE-bench repos and apply / create patches.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import git  # GitPython

from config import REPOS_DIR, PATCHES_DIR
from swebench_loader import SWEInstance

logger = logging.getLogger(__name__)


class RepoManager:
    """Manages a local clone of a GitHub repository at a specific commit."""

    def __init__(self, instance: SWEInstance):
        self.instance = instance
        self.repo_dir = Path(REPOS_DIR) / instance.instance_id
        self._git_repo: Optional[git.Repo] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def setup(self) -> Path:
        """Clone (or reuse) the repo and check out the base commit."""
        Path(REPOS_DIR).mkdir(parents=True, exist_ok=True)
        Path(PATCHES_DIR).mkdir(parents=True, exist_ok=True)

        if self.repo_dir.exists():
            logger.info("Reusing existing clone at %s", self.repo_dir)
            self._git_repo = git.Repo(self.repo_dir)
        else:
            clone_url = f"https://github.com/{self.instance.repo}.git"
            logger.info("Cloning %s …", clone_url)
            self._git_repo = git.Repo.clone_from(
                clone_url, self.repo_dir, depth=50
            )

        # Checkout the pre-fix commit
        logger.info("Checking out base commit %s", self.instance.base_commit[:8])
        self._git_repo.git.checkout(self.instance.base_commit)
        return self.repo_dir

    def read_file(self, relative_path: str) -> str:
        """Read a file from the cloned repo."""
        full_path = self.repo_dir / relative_path
        return full_path.read_text(encoding="utf-8", errors="replace")

    def write_file(self, relative_path: str, content: str) -> None:
        """Overwrite a file in the cloned repo."""
        full_path = self.repo_dir / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def list_python_files(self, subdir: str = "") -> list[str]:
        """Return repo-relative paths of all .py files under `subdir`."""
        base = self.repo_dir / subdir if subdir else self.repo_dir
        files = []
        for p in base.rglob("*.py"):
            # Skip test files and vendored code
            parts = p.parts
            if any(part in ("test", "tests", "vendor", ".tox", "venv") for part in parts):
                continue
            files.append(str(p.relative_to(self.repo_dir)))
        return files

    def create_patch(self, output_name: str = "") -> str:
        """
        Create a unified diff of all uncommitted changes.
        Saves the patch to PATCHES_DIR and returns the patch text.
        """
        assert self._git_repo is not None, "Call setup() first."
        diff = self._git_repo.git.diff()
        if not diff:
            logger.warning("No changes detected - patch is empty.")
            return ""

        name = output_name or self.instance.instance_id
        patch_path = Path(PATCHES_DIR) / f"{name}.patch"
        patch_path.write_text(diff, encoding="utf-8")
        logger.info("Patch written to %s", patch_path)
        return diff

    def reset(self) -> None:
        """Hard-reset the repo to the base commit (undo injected bugs)."""
        if self._git_repo:
            self._git_repo.git.checkout("--", ".")
            logger.info("Repo reset to base commit.")
