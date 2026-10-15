"""Self-update helper for the KiwiMateCoder CLI."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markup import escape

PACKAGE_NAME = "kiwimatecoder"
GITHUB_REPO_URL = "https://github.com/Kyle8933/kiwimatecoder.git"
GITHUB_REPO_OWNER = "Kyle8933"
GITHUB_REPO_NAME = "kiwimatecoder"
UPDATE_STATE_NAME = "update_check.json"
DEFAULT_CHECK_INTERVAL_HOURS = 24


@dataclass(frozen=True)
class UpdateInfo:
    """Result of checking for a newer KiwiMateCoder version."""

    update_available: bool
    current_version: str
    latest_version: str
    current_sha: str | None = None
    latest_sha: str | None = None
    commits_behind: int | None = None
    error: str | None = None


def _update_state_path() -> Path:
    from kiwimatecoder.config import CONFIG_DIR

    return CONFIG_DIR / UPDATE_STATE_NAME


def load_update_state() -> dict:
    """Load the cached update-check state (last check time, last seen version)."""
    path = _update_state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_update_state(state: dict) -> None:
    """Persist the update-check state."""
    from kiwimatecoder.config import ensure_config_dir

    ensure_config_dir()
    path = _update_state_path()
    try:
        path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def _get_remote_head_sha(branch: str = "main") -> str | None:
    """Return the short SHA of the remote branch HEAD via ``git ls-remote``."""
    try:
        cp = subprocess.run(
            ["git", "ls-remote", GITHUB_REPO_URL, f"refs/heads/{branch}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if cp.returncode != 0:
        return None
    line = cp.stdout.strip().splitlines()
    if not line:
        return None
    full_sha = line[0].split()[0]
    return full_sha[:7] if len(full_sha) >= 7 else full_sha


def _get_latest_release_tag() -> str | None:
    """Return the latest GitHub release tag, or None if unavailable."""
    import httpx

    from kiwimatecoder import network

    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/latest"
    try:
        with httpx.Client(
            timeout=8.0,
            follow_redirects=True,
            **network.current_options(),
        ) as client:
            response = client.get(url, headers={"Accept": "application/vnd.github+json"})
            if response.status_code != 200:
                return None
            data = response.json()
            tag = data.get("tag_name")
            return str(tag) if tag else None
    except Exception:
        return None


def check_for_update(
    *, force: bool = False, console: Console | None = None
) -> UpdateInfo:
    """Check whether a newer KiwiMateCoder version is available.

    For source checkouts, compares the local HEAD SHA against the remote
    branch HEAD. For packaged installs, queries the GitHub releases API.
    Results are cached in ``~/.kiwimatecoder/update_check.json`` to avoid
    hitting the network on every startup; ``force`` bypasses the cache.

    Respects offline mode and never raises — failures return an ``UpdateInfo``
    with ``error`` set.
    """
    from kiwimatecoder import __version__, network

    if network.offline_enabled():
        return UpdateInfo(
            update_available=False,
            current_version=__version__,
            latest_version=__version__,
            error="offline mode is enabled",
        )

    state = load_update_state()
    now = time.time()

    if not force:
        last_check = state.get("last_check")
        if isinstance(last_check, (int, float)):
            from kiwimatecoder.config import get_update

            interval_hours = get_update().get("check_interval_hours", DEFAULT_CHECK_INTERVAL_HOURS)
            if now - last_check < interval_hours * 3600:
                cached_available = state.get("update_available", False)
                return UpdateInfo(
                    update_available=bool(cached_available),
                    current_version=str(state.get("current_version", __version__)),
                    latest_version=str(state.get("latest_version", __version__)),
                    current_sha=state.get("current_sha"),
                    latest_sha=state.get("latest_sha"),
                    commits_behind=state.get("commits_behind"),
                )

    current_version = __version__
    source_root = find_source_root()

    if source_root is not None and _has_git_remote(source_root):
        local_sha = _get_short_sha(source_root)
        branch = _get_branch(source_root) or "main"
        remote_sha = _get_remote_head_sha(branch)

        if remote_sha is None:
            return UpdateInfo(
                update_available=False,
                current_version=current_version,
                latest_version=current_version,
                current_sha=local_sha,
                error="could not reach GitHub",
            )

        if local_sha is None:
            return UpdateInfo(
                update_available=False,
                current_version=current_version,
                latest_version=current_version,
                latest_sha=remote_sha,
                error="could not determine local commit",
            )

        behind = _commits_behind(source_root, branch)
        # Fetch first to get an accurate behind count
        if behind is None:
            try:
                subprocess.run(
                    ["git", "-C", str(source_root), "fetch", "origin", branch],
                    capture_output=True,
                    timeout=15,
                )
                behind = _commits_behind(source_root, branch)
            except (OSError, subprocess.TimeoutExpired):
                behind = None

        update_available = local_sha != remote_sha and (behind is None or behind > 0)
        result = UpdateInfo(
            update_available=update_available,
            current_version=current_version,
            latest_version=current_version,
            current_sha=local_sha,
            latest_sha=remote_sha,
            commits_behind=behind,
        )
    else:
        latest_tag = _get_latest_release_tag()
        if latest_tag is None:
            return UpdateInfo(
                update_available=False,
                current_version=current_version,
                latest_version=current_version,
                error="could not reach GitHub",
            )
        update_available = latest_tag.lstrip("v") != current_version.lstrip("v")
        result = UpdateInfo(
            update_available=update_available,
            current_version=current_version,
            latest_version=latest_tag,
        )

    save_update_state(
        {
            "last_check": now,
            "update_available": result.update_available,
            "current_version": result.current_version,
            "latest_version": result.latest_version,
            "current_sha": result.current_sha,
            "latest_sha": result.latest_sha,
            "commits_behind": result.commits_behind,
        }
    )
    return result


def should_check_for_update() -> bool:
    """Whether the periodic auto-check should run now (config + interval)."""
    from kiwimatecoder.config import get_update

    settings = get_update()
    if not settings.get("auto_check", True):
        return False
    state = load_update_state()
    last_check = state.get("last_check")
    if not isinstance(last_check, (int, float)):
        return True
    interval_hours = settings.get("check_interval_hours", DEFAULT_CHECK_INTERVAL_HOURS)
    return time.time() - last_check >= interval_hours * 3600


def build_update_command(ref: str | None = None) -> list[str]:
    """Return the fallback update command for packaged (non-Git) installs.

    KiwiMateCoder is not published to PyPI, so the fallback installs from the
    GitHub repository instead of a PyPI package name. ``--force-reinstall`` is
    required because the package version string is often unchanged between
    commits, so ``--upgrade`` alone can be a no-op ("Requirement already
    satisfied"). ``ref`` pins a branch, tag, or commit SHA.
    """
    target = f"git+{GITHUB_REPO_URL}"
    if ref:
        target = f"{target}@{ref}"
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        target,
    ]


def build_git_pull_command(source_root: Path) -> list[str]:
    """Return the command used to fetch source updates for a Git checkout."""
    return ["git", "-C", str(source_root), "pull", "--ff-only"]


def build_git_checkout_command(
    source_root: Path, ref: str, *, current_branch: str | None = None
) -> list[str]:
    """Return the command that puts a Git checkout on ``ref``.

    When ``ref`` is the branch already checked out, a fast-forward pull is
    used so local commits are never merged unexpectedly; every other ref
    (tag, SHA, or other branch) is a plain ``git checkout``.
    """
    if current_branch is not None and ref == current_branch:
        return build_git_pull_command(source_root)
    return ["git", "-C", str(source_root), "checkout", ref]


def build_source_install_command(source_root: Path) -> list[str]:
    """Return the command used to reinstall KiwiMateCoder from a source checkout."""
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "-e",
        str(source_root),
    ]


def find_source_root(package_file: Path | None = None) -> Path | None:
    """Find the project root for editable/local source installs."""
    current = (package_file or Path(__file__)).resolve()
    for candidate in (current.parent, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / PACKAGE_NAME).is_dir()
        ):
            return candidate
    return None


def _has_git_remote(source_root: Path) -> bool:
    probe = subprocess.run(
        ["git", "-C", str(source_root), "remote", "get-url", "origin"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return probe.returncode == 0


def _git(args: list[str], source_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(source_root), *args],
        capture_output=True,
        text=True,
    )


def _get_short_sha(source_root: Path) -> str | None:
    """Return the short SHA of HEAD, or None if it cannot be determined."""
    cp = _git(["rev-parse", "--short", "HEAD"], source_root)
    if cp.returncode != 0:
        return None
    return cp.stdout.strip() or None


def _get_branch(source_root: Path) -> str | None:
    """Return the current branch name, or None when detached/unknown."""
    cp = _git(["rev-parse", "--abbrev-ref", "HEAD"], source_root)
    if cp.returncode != 0:
        return None
    branch = cp.stdout.strip()
    return None if branch in ("", "HEAD") else branch


def _commits_behind(source_root: Path, branch: str | None) -> int | None:
    """Return how many commits HEAD is behind origin/<branch>, or None if unknown."""
    if not branch:
        return None
    cp = _git(["rev-list", "--count", f"HEAD..origin/{branch}"], source_root)
    if cp.returncode != 0:
        return None
    try:
        return int(cp.stdout.strip())
    except ValueError:
        return None


def _fetch(source_root: Path, console: Console) -> int:
    """Best-effort `git fetch origin`. Failures are non-fatal."""
    command = ["git", "-C", str(source_root), "fetch", "origin"]
    console.print(f"[dim]{escape(shlex.join(command))}[/dim]")
    try:
        completed = subprocess.run(command, text=True)
    except OSError as exc:
        console.print(f"[yellow]Could not fetch from origin: {escape(str(exc))}[/yellow]")
        return 1
    if completed.returncode != 0:
        console.print(
            "[yellow]git fetch failed; continuing with a direct pull.[/yellow]"
        )
    return completed.returncode


def _run(command: list[str], console: Console) -> int:
    console.print(f"[dim]{escape(shlex.join(command))}[/dim]")
    try:
        completed = subprocess.run(command, text=True)
    except OSError as exc:
        console.print(f"[red]Could not start command: {escape(str(exc))}[/red]")
        return 1
    return completed.returncode


def run_update(console: Console | None = None, ref: str | None = None) -> int:
    """Update KiwiMateCoder in the current Python environment.

    ``ref`` pins a branch, tag, or commit SHA for the update. In a Git
    checkout the current branch is fast-forwarded when it matches, and any
    other ref is a plain ``git checkout`` followed by a reinstall; packaged
    installs target ``git+<repo>@<ref>``.
    """
    console = console or Console()
    console.print("[cyan]Updating KiwiMateCoder...[/cyan]")
    if ref:
        console.print(f"[dim]Target ref: {escape(ref)}[/dim]")

    source_root = find_source_root()
    if source_root is not None and _has_git_remote(source_root):
        console.print(f"[dim]Detected source checkout: {escape(str(source_root))}[/dim]")

        old_sha = _get_short_sha(source_root)
        _fetch(source_root, console)
        branch = _get_branch(source_root)

        if ref is not None:
            if branch is not None and ref == branch and _commits_behind(source_root, branch) == 0:
                console.print(
                    "[green]Already on "
                    + f"{escape(ref)} (commit {old_sha or 'unknown'}).[/green]"
                )
                return 0
            checkout = build_git_checkout_command(
                source_root, ref, current_branch=branch
            )
            console.print(
                f"[cyan]Checking out {escape(ref)} from {old_sha or 'unknown'}…[/cyan]"
            )
            checkout_code = _run(checkout, console)
            if checkout_code != 0:
                console.print(
                    f"[red]Update failed while checking out {escape(ref)}.[/red] "
                    + "[yellow]Commit/stash local changes or verify the ref "
                    + "exists (try `git fetch --tags origin`), then try "
                    + "again.[/yellow]"
                )
                return checkout_code

            new_sha = _get_short_sha(source_root)
            if new_sha == old_sha:
                console.print(
                    "[green]Already on "
                    + f"{escape(ref)} (commit {old_sha or 'unknown'}).[/green]"
                )
                return 0
            console.print(
                f"[dim]Updated {old_sha or 'unknown'} → "
                + f"{new_sha or escape(ref)} (ref {escape(ref)}).[/dim]"
            )
            code = _run(build_source_install_command(source_root), console)
        else:
            behind = _commits_behind(source_root, branch)

            if behind == 0:
                console.print(
                    "[green]Already on the latest version "
                    + f"(commit {old_sha or 'unknown'}).[/green]"
                )
                return 0

            if old_sha and behind and branch:
                console.print(
                    f"[cyan]Updating from {old_sha} "
                    + f"({behind} commit(s) behind origin/{escape(branch)})…[/cyan]"
                )
            else:
                console.print(
                    f"[cyan]Updating KiwiMateCoder from {old_sha or 'unknown'}…[/cyan]"
                )

            pull_code = _run(build_git_pull_command(source_root), console)
            if pull_code != 0:
                console.print(
                    "[red]Update failed while pulling from Git.[/red] "
                    + "[yellow]Commit/stash local changes or resolve Git errors, "
                    + "then try again.[/yellow]"
                )
                return pull_code

            new_sha = _get_short_sha(source_root)
            if new_sha == old_sha:
                console.print(
                    "[green]Already on the latest version "
                    + f"(commit {old_sha or 'unknown'}).[/green]"
                )
                return 0

            if old_sha and new_sha:
                console.print(f"[dim]Updated {old_sha} → {new_sha}.[/dim]")

            code = _run(build_source_install_command(source_root), console)
    else:
        code = _run(build_update_command(ref), console)
        if code != 0:
            target = f"git+{GITHUB_REPO_URL}"
            if ref:
                target = f"{target}@{ref}"
            console.print(
                f"[red]Update failed with exit code {code}.[/red] "
                + "[yellow]KiwiMateCoder is not on PyPI; install from a source "
                + "checkout or run: pip install --upgrade --force-reinstall "
                + f"{escape(target)}[/yellow]"
            )
            return code

    if code == 0:
        console.print("[green]KiwiMateCoder update complete.[/green]")
    else:
        console.print(f"[red]Update failed with exit code {code}.[/red]")
    return code
