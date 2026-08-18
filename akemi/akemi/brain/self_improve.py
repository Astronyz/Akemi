import asyncio
import tempfile
import subprocess
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import structlog

from akemi.akemi.core.config import get_settings
from akemi.akemi.storage import get_db
from akemi.akemi.storage.models import EventType
from akemi.akemi.brain import BrainProviderFactory

logger = structlog.get_logger()


class SelfImprover:
    """Self-improvement module: analyzes errors, generates fixes, creates PRs."""

    def __init__(self):
        self.settings = get_settings()
        self.si_settings = self.settings.self_improve
        self._enabled = self.si_settings.enabled
        self._last_run: Optional[datetime] = None
        self._pr_count_today = 0
        self._pr_date = datetime.utcnow().date()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def can_run(self) -> bool:
        """Check if self-improvement can run."""
        if not self._enabled:
            return False

        # Check cooldown
        if self._last_run:
            elapsed = datetime.utcnow() - self._last_run
            if elapsed < timedelta(hours=self.si_settings.interval_hours):
                return False

        # Check daily PR limit
        today = datetime.utcnow().date()
        if today != self._pr_date:
            self._pr_count_today = 0
            self._pr_date = today

        if self._pr_count_today >= self.si_settings.max_prs_per_day:
            return False

        return True

    async def run_improvement_cycle(self) -> Dict[str, Any]:
        """Run one self-improvement cycle."""
        if not self.can_run():
            return {"status": "skipped", "reason": "cooldown or limit reached"}

        logger.info("Starting self-improvement cycle")
        self._last_run = datetime.utcnow()

        try:
            # 1. Collect recent errors
            errors = self._collect_errors()
            if not errors:
                return {"status": "no_errors", "message": "No recent errors to analyze"}

            # 2. Analyze with LLM
            fix_proposal = await self._analyze_errors(errors)

            if not fix_proposal:
                return {"status": "no_fix", "message": "LLM couldn't generate a fix"}

            # 3. Apply fix in isolated worktree
            applied = await self._apply_fix(fix_proposal)

            if not applied:
                return {"status": "apply_failed", "message": "Failed to apply fix"}

            # 4. Run tests
            if self.si_settings.test_before_pr:
                test_passed = await self._run_tests()
                if not test_passed:
                    await self._rollback_fix()
                    return {"status": "test_failed", "message": "Tests failed after fix"}

            # 5. Create PR
            pr_url = await self._create_pr(fix_proposal)

            self._pr_count_today += 1

            return {
                "status": "success",
                "pr_url": pr_url,
                "errors_analyzed": len(errors),
                "fix_summary": fix_proposal.get("summary", ""),
            }

        except Exception as e:
            logger.error("Self-improvement cycle failed", error=str(e))
            return {"status": "error", "error": str(e)}

    def _collect_errors(self) -> List[Dict[str, Any]]:
        """Collect recent error events from database."""
        db = get_db()
        since = datetime.utcnow() - timedelta(hours=self.si_settings.interval_hours * 2)
        error_events = db.get_error_events(since=since, limit=50)

        # Filter and deduplicate
        seen = set()
        unique_errors = []
        for event in error_events:
            key = (event.get("error_type", ""), event.get("message", "")[:200])
            if key not in seen:
                seen.add(key)
                unique_errors.append(event)

        return unique_errors[:20]  # Limit to 20 unique errors

    async def _analyze_errors(self, errors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze errors and propose a fix."""
        brain = BrainProviderFactory.create(
            self.settings.brain.provider,
            model=getattr(self.settings.brain, f"{self.settings.brain.provider}_model", ""),
            api_key=getattr(self.settings.brain, f"{self.settings.brain.provider}_api_key", None),
        )
        await brain.initialize()

        # Build prompt
        error_text = "\n\n".join([
            f"Error Type: {e.get('error_type', 'Unknown')}\n"
            f"Message: {e.get('message', '')}\n"
            f"Traceback: {e.get('traceback', '')[:500]}\n"
            f"Context: {e.get('context', '')}"
            for e in errors
        ])

        prompt = f"""Analyze these errors from an autonomous AI agent running on Windows:

{error_text}

The agent has these components:
- Audio capture (WASAPI loopback) + VAD + STT (faster-whisper)
- Vision (screenshot + OCR + frame diff)
- Brain (LLM: {self.settings.brain.provider})
- TTS (Piper)
- Local control API (FastAPI)
- SQLite storage

Propose a minimal, focused fix as a git diff. Return JSON with:
{{
  "summary": "One-line summary of the fix",
  "files": [
    {{"path": "relative/path.py", "diff": "unified diff content"}}
  ],
  "tests_to_run": ["pytest command"]
}}"""

        from akemi.akemi.brain import BrainMessage
        messages = [
            BrainMessage(role="system", content="You are an expert Python developer. Generate minimal, safe fixes."),
            BrainMessage(role="user", content=prompt),
        ]

        response = await brain.generate(messages, temperature=0.3, max_tokens=4000)
        await brain.close()

        try:
            import json
            return json.loads(response.text)
        except Exception as e:
            logger.error("Failed to parse LLM fix proposal", error=str(e))
            return None

    async def _apply_fix(self, fix_proposal: Dict[str, Any]) -> bool:
        """Apply fix in isolated git worktree."""
        repo_path = Path.cwd()

        # Create worktree
        worktree_path = repo_path / ".akemi_worktree"
        branch_name = f"akemi-fix-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        try:
            # Create worktree
            subprocess.run(
                ["git", "worktree", "add", "-b", branch_name, str(worktree_path)],
                check=True,
                capture_output=True,
                cwd=repo_path,
            )

            # Apply each file diff
            for file_change in fix_proposal.get("files", []):
                file_path = worktree_path / file_change["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Apply patch using git apply
                result = subprocess.run(
                    ["git", "apply", "--whitespace=nowarn", "-"],
                    input=file_change["diff"].encode(),
                    capture_output=True,
                    cwd=worktree_path,
                )

                if result.returncode != 0:
                    logger.error("Failed to apply diff", path=file_change["path"], error=result.stderr.decode())
                    return False

            # Commit changes
            subprocess.run(
                ["git", "add", "-A"],
                check=True,
                cwd=worktree_path,
            )
            subprocess.run(
                ["git", "commit", "-m", f"akemi: {fix_proposal.get('summary', 'Auto-fix')}"],
                check=True,
                cwd=worktree_path,
            )

            # Push branch
            subprocess.run(
                ["git", "push", "origin", branch_name],
                check=True,
                cwd=worktree_path,
            )

            # Cleanup worktree
            subprocess.run(
                ["git", "worktree", "remove", str(worktree_path)],
                capture_output=True,
                cwd=repo_path,
            )

            return True

        except subprocess.CalledProcessError as e:
            logger.error("Git operation failed", error=e.stderr.decode() if e.stderr else str(e))
            # Try cleanup
            try:
                subprocess.run(["git", "worktree", "remove", str(worktree_path)], capture_output=True, cwd=repo_path)
            except Exception:
                pass
            return False

    async def _run_tests(self) -> bool:
        """Run test suite."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "-x", "-q"],
                capture_output=True,
                timeout=300,
                cwd=Path.cwd(),
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error("Tests timed out")
            return False
        except Exception as e:
            logger.error("Test run failed", error=str(e))
            return False

    async def _rollback_fix(self) -> None:
        """Rollback the applied fix (placeholder)."""
        logger.info("Rolling back fix (manual intervention may be needed)")

    async def _create_pr(self, fix_proposal: Dict[str, Any]) -> Optional[str]:
        """Create a GitHub PR."""
        if not self.si_settings.github_token:
            logger.warning("No GitHub token, skipping PR creation")
            return None

        try:
            from github import Github
            g = Github(self.si_settings.github_token)
            repo = g.get_repo(self.si_settings.github_repository)

            branch_name = f"akemi-fix-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

            pr = repo.create_pull(
                title=f"[akemi] {fix_proposal.get('summary', 'Auto-fix')}",
                body=f"""## Automated Fix by Akemi

**Summary:** {fix_proposal.get('summary', 'Auto-fix')}

**Errors Analyzed:** {fix_proposal.get('errors_analyzed', 'N/A')}

**Changes:**
{chr(10).join([f'- `{f["path"]}`' for f in fix_proposal.get('files', [])])}

**Tests:** Run `{', '.join(fix_proposal.get('tests_to_run', ['pytest']))}` to verify.

---
*This PR was created automatically by Akemi's self-improvement module.*
""",
                head=branch_name,
                base=repo.default_branch,
            )

            logger.info("PR created", url=pr.html_url, number=pr.number)
            return pr.html_url

        except Exception as e:
            logger.error("Failed to create PR", error=str(e))
            return None


# Global instance
_self_improver: Optional[SelfImprover] = None


def get_self_improver() -> SelfImprover:
    global _self_improver
    if _self_improver is None:
        _self_improver = SelfImprover()
    return _self_improver


async def run_self_improve_loop() -> None:
    """Background task to run self-improvement periodically."""
    improver = get_self_improver()

    while True:
        try:
            if improver.enabled and improver.can_run():
                result = await improver.run_improvement_cycle()
                logger.info("Self-improvement cycle completed", result=result)
        except Exception as e:
            logger.error("Self-improvement loop error", error=str(e))

        # Sleep for 1 hour before checking again
        await asyncio.sleep(3600)