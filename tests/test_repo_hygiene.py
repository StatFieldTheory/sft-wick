"""The published tree holds code and its documentation only.

Agent working files (instructions, task state, handoffs, review notes),
development records and local absolute paths are kept out of version
control.  ``.gitignore`` and the sdist excludes stop most of them; these
tests catch what slips past, such as a file added with ``git add -f`` or a
path pasted into a docstring.  They read ``git ls-files`` and skip outside
a git checkout (an unpacked source distribution).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Tracked paths that are never part of the published tree.
FORBIDDEN_PATHS = re.compile(
    r"""(?ix)
    (^|/)(claude|claude\.local|agents?|gemini|codex)\.md$
    | (^|/)\.(agents|agent-state|claude|codex|cursor)/
    | (^|/)\.cursorrules$ | (^|/)\.aider
    | (^|/)handoff | (^|/)session[^/]*\.md$ | (^|/)referee_brief\.md$
    | ^docs/research/
    """
)

#: Local absolute paths: a home directory or a machine temp directory.
LOCAL_PATH = re.compile(
    r"/Users/[A-Za-z0-9._-]+/|/home/[a-z_][a-z0-9_-]*/|/private/(tmp|var)/"
    r"|[A-Za-z]:\\Users\\"
)

#: Development task identifiers such as T-003, T003, t003_name or a path
#: segment /T001_x.  A letter, digit or ``+`` before the ID, or a base64
#: character after it, rules a match out, so encoded notebook images do not
#: match.
TASK_ID = re.compile(r"(?<![A-Za-z0-9+])[Tt]-?0\d\d(?![A-Za-z0-9+/=])")

#: Names of development phases, which say nothing about the code.
PHASE_NAME = re.compile(r"(?i)\btranches?\b")

#: Files that name the patterns above in order to exclude them.
CONTENT_ALLOWLIST = {"tests/test_repo_hygiene.py"}


def _tracked_files() -> list[str]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return [name for name in out.decode().split("\0") if name]


def _text(name: str) -> str | None:
    path = ROOT / name
    if not path.is_file():
        return None
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def test_RH1_no_agent_files_or_development_records_tracked():
    bad = [name for name in _tracked_files() if FORBIDDEN_PATHS.search(name)]
    assert bad == [], f"agent or development files are tracked: {bad}"


def test_RH2_no_local_absolute_paths_in_tracked_text():
    hits = []
    for name in _tracked_files():
        if name in CONTENT_ALLOWLIST:
            continue
        text = _text(name)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if LOCAL_PATH.search(line):
                hits.append(f"{name}:{lineno}")
    assert hits == [], f"local absolute paths in tracked files: {hits}"


def test_RH3_no_development_task_ids_in_tracked_text():
    hits = []
    for name in _tracked_files():
        if name in CONTENT_ALLOWLIST:
            continue
        text = _text(name)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if TASK_ID.search(line) or PHASE_NAME.search(line):
                hits.append(f"{name}:{lineno}")
    assert hits == [], f"development task IDs or phase names in tracked files: {hits}"


@pytest.mark.parametrize(
    "name, forbidden",
    [
        ("CLAUDE.md", True),
        ("docs/AGENTS.md", True),
        (".agents/skills/x/SKILL.md", True),
        (".agent-state/tasks/T-1/task.json", True),
        ("docs/research/T003_x/REPORT.md", True),
        ("HANDOFF.md", True),
        ("README.md", False),
        ("src/sft_wick/evaluate.py", False),
        ("docs/user_guide/resources.rst", False),
        ("tests/test_repo_hygiene.py", False),
    ],
)
def test_RH4_path_rule_is_not_vacuous(name, forbidden):
    assert bool(FORBIDDEN_PATHS.search(name)) is forbidden


@pytest.mark.parametrize(
    "line, local",
    [
        ("see /Users/someone/notes.md", True),
        ("--json /private/tmp/out.json", True),
        ("cache at /home/user/.cache/x", True),
        ("https://github.com/StatFieldTheory/sft-wick", False),
        ("python tools/benchmark_gl_memory.py --json gl-memory.json", False),
    ],
)
def test_RH5_local_path_rule_is_not_vacuous(line, local):
    assert bool(LOCAL_PATH.search(line)) is local


@pytest.mark.parametrize(
    "line, flagged",
    [
        ("see T-003 for the plan", True),
        ("docs/notes/T001_revalidation.md", True),
        ('"t003_joint_statistic_reference", path', True),
        ("requires the final seventh-tranche reference", True),
        ("iVBORw0KGgoAAAANSUhEUgAt012Bc", False),
        ("+t012/", False),
        ("dt003 = 0.5", False),
        ("T0123", False),
        ("a few tenths of a percent", False),
        ("release 0.7.0 and 0.6.1", False),
    ],
)
def test_RH6_task_id_and_phase_rules_are_not_vacuous(line, flagged):
    assert bool(TASK_ID.search(line) or PHASE_NAME.search(line)) is flagged
