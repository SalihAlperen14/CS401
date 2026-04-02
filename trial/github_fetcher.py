"""
github_fetcher.py - Fetch repository files via GitHub REST API + whatthepatch.

No local cloning required. Files are fetched on-demand at a specific commit.
Unified diffs are parsed with whatthepatch to identify changed files.
"""

from __future__ import annotations

import base64
import logging
from typing import Dict, List, Optional

import requests
import whatthepatch

from swebench_loader import SWEInstance

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


class GitHubFetcher:
    """
    Fetches source files from GitHub using the REST API.
    Parses unified diffs using whatthepatch to identify changed paths.
    No local clone required — all files are fetched on-demand.
    """

    def __init__(self, token: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def fetch_file(self, repo: str, path: str, ref: str) -> str:
        """Fetch the content of a single file at the given commit ref."""
        url = f"{GITHUB_API_BASE}/repos/{repo}/contents/{path}"
        resp = self.session.get(url, params={"ref": ref}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return data.get("content", "")

    def parse_patch(self, patch: str) -> Dict[str, List]:
        """
        Parse a unified diff string using whatthepatch.

        Returns {relative_path: [changes]} where each change is a
        (old_lineno, new_lineno, line_text) namedtuple from whatthepatch.
        """
        result: Dict[str, List] = {}
        for diff in whatthepatch.parse_patch(patch):
            if diff.header is None:
                continue
            # new_path is like "b/path/to/file.py" — strip the leading "b/"
            raw_path = diff.header.new_path or ""
            path = raw_path[2:] if raw_path.startswith("b/") else raw_path.lstrip("/")
            if path and path != "dev/null":
                result[path] = diff.changes or []
        return result

    def fetch_changed_files(self, instance: SWEInstance) -> Dict[str, str]:
        """
        Fetch the pre-fix content of every Python file touched by instance.patch.

        Returns {relative_path: file_content_at_base_commit}.
        """
        patch_files = self.parse_patch(instance.patch)
        contents: Dict[str, str] = {}

        for path in patch_files:
            if not path.endswith(".py"):
                continue
            try:
                content = self.fetch_file(instance.repo, path, instance.base_commit)
                contents[path] = content
                logger.info("Fetched %s @ %s", path, instance.base_commit[:8])
            except requests.HTTPError as exc:
                logger.warning("Could not fetch %s: HTTP %s", path, exc.response.status_code)
            except Exception as exc:
                logger.warning("Unexpected error fetching %s: %s", path, exc)

        return contents
