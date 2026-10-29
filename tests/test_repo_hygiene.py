"""The published tree holds code and its documentation only.

Agent working files (instructions, task state, handoffs, review notes),
development records and local absolute paths are kept out of version
control.  ``.gitignore`` and the sdist excludes stop most of them; these
tests catch what slips past, such as a file added with ``git add -f`` or a
path pasted into a docstring.  Development task IDs are kept out of file
names and commit messages as well.  The tests read ``git ls-files`` and
``git log`` and skip outside a git checkout (an unpacked source
distribution).

``flagged_commit_messages`` and ``flagged_commit_content`` take any
``git log`` revision arguments, so the commits of a push can be checked
before they leave the machine, for example
``flagged_commit_content(ROOT, sha, "--not", "--remotes=origin")``.
"""
from __future__ import annotations

import os
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

#: Development task identifiers: T-003, T003, t003_name, a path segment
#: /T001_x, and any other three-digit task number with the subtask IDs built
#: on it, such as T-104, T104-A01 and T104-B05-name.  A letter, digit or
#: ``+`` before the ID, or a base64 character after it, rules a match out, so
#: encoded notebook images do not match.  ``development_name`` adds the case
#: of an ID followed by ``/``.
TASK_ID = re.compile(r"(?<![A-Za-z0-9+])[Tt]-?\d{3}(?![A-Za-z0-9+/=])")

#: Identifiers of planning notes kept outside the repository, such as
#: RB-XYZ-07 or rb_xyz_07.
NOTE_ID = re.compile(r"(?i)(?<![A-Za-z0-9])rb[-_][a-z]{3}[-_]\d\d(?!\d)")

#: Names of development phases, which say nothing about the code.
PHASE_NAME = re.compile(r"(?i)\btranches?\b")

#: A run of base64 characters long enough to be encoded data, not a path.
BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{40,}=*")

#: Files that name the patterns above in order to exclude them.
CONTENT_ALLOWLIST = {"tests/test_repo_hygiene.py"}

#: Full hashes of published commits whose message and content are accepted
#: as they are, so that a flagged commit already in public history does not
#: fail every later run.
PUBLISHED_COMMIT_ALLOWLIST: frozenset[str] = frozenset()


def development_name(text: str) -> bool:
    """True if ``text`` holds a development task ID, note ID or phase name.

    The second form drops base64 runs and reads ``/`` as a space, so an ID
    followed by a slash (a branch ``t104/x``, a path ``notes/T104/x``) is
    caught while encoded images still are not.
    """
    forms = (text, BASE64_RUN.sub(" ", text).replace("/", " "))
    rules = (TASK_ID, NOTE_ID, PHASE_NAME)
    return any(rule.search(form) for form in forms for rule in rules)


def _git(*args: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True
    ).stdout
    return out.decode("utf-8", errors="replace")


def _require_checkout() -> None:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")


def _tracked_files() -> list[str]:
    _require_checkout()
    return [name for name in _git("ls-files", "-z").split("\0") if name]


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


def _diff_path(header: str) -> str:
    """The path of a ``diff --git a/<path> b/<path>`` header, whose two names
    agree under ``--no-renames``."""
    names = header[len("diff --git ") :]
    return names[2 : (len(names) - 1) // 2]


def flagged_commit_messages(
    cwd: Path, *revs: str, env: dict[str, str] | None = None
) -> list[str]:
    """Message lines that name development work, as ``<hash>: <line>``, in the
    commits ``git log <revs>`` lists."""
    log = _git("log", "--format=%H%x1f%B%x1e", *revs, cwd=cwd, env=env)
    hits = []
    for record in log.split("\x1e"):
        sha, _, message = record.strip("\n").partition("\x1f")
        if sha in PUBLISHED_COMMIT_ALLOWLIST:
            continue
        lines = message.splitlines()
        hits.extend(f"{sha[:12]}: {line}" for line in lines if development_name(line))
    return hits


def flagged_commit_content(
    cwd: Path, *revs: str, env: dict[str, str] | None = None
) -> list[str]:
    """What the commits ``git log <revs>`` lists add that must not be published.

    An added agent file, or a file whose name holds a task ID, is reported as
    ``<hash> <path>``; an added line that holds a task ID or a local path as
    ``<hash> <path>: <line>``.  A renamed file is read as a deletion and an
    addition, so its full content is checked.  Binary content is not read.
    """
    log = _git(
        "-c", "core.quotePath=false", "log", "--format=%x1e%H", "--patch",
        "--unified=0", "--no-renames", "--diff-filter=d", "--no-ext-diff",
        "--no-textconv", "--no-color", *revs, cwd=cwd, env=env,
    )  # fmt: skip
    hits = []
    for record in log.split("\x1e"):
        sha, _, patch = record.partition("\n")
        if not sha or sha in PUBLISHED_COMMIT_ALLOWLIST:
            continue
        path, in_hunk = "", False
        for line in patch.splitlines():
            if line.startswith("diff --git "):
                path, in_hunk = _diff_path(line), False
                if FORBIDDEN_PATHS.search(path) or development_name(path):
                    hits.append(f"{sha[:12]} {path}")
            elif line.startswith("@@"):
                in_hunk = True
            elif in_hunk and line.startswith("+") and path not in CONTENT_ALLOWLIST:
                text = line[1:]
                if development_name(text) or LOCAL_PATH.search(text):
                    hits.append(f"{sha[:12]} {path}: {text}")
    return hits


def _isolated_git_env() -> dict[str, str]:
    """Environment for a scratch repository: no inherited ``GIT_*`` variables
    (git exports ``GIT_DIR`` to a hook that runs these tests) and no user or
    system configuration."""
    env = {key: val for key, val in os.environ.items() if not key.startswith("GIT_")}
    ident = {"NAME": "test", "EMAIL": "test@example.invalid"}
    for role in ("AUTHOR", "COMMITTER"):
        env.update({f"GIT_{role}_{key}": value for key, value in ident.items()})
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return env


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
            if development_name(line):
                hits.append(f"{name}:{lineno}")
    assert hits == [], f"task IDs, note IDs or phase names in tracked files: {hits}"


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
        ("see T-104 for the plan", True),
        ("feat: schema-first parser (T104-A01)", True),
        ("notes for T104-B05-name", True),
        ("notes/t104_concept.md", True),
        ("see notes/T104/README.md", True),
        ("t104/a01-parser", True),
        ("closes RB-XYZ-07", True),
        ("closes rb_xyz_07", True),
        ("a T1040 grid", False),
        ("yfydHndKJCNDfKvXqBD5CAG4xp49e+T555+XRYsW", False),
        ("88+M+/t999/73D5I88+M+/t999/73D5I88+M+/t999/73D5I", False),
        ("/t104==", False),
        ("/t104+Q", False),
        ("ORB-XYZ-07", False),
    ],
)
def test_RH6_task_id_and_phase_rules_are_not_vacuous(line, flagged):
    assert development_name(line) is flagged


def test_RH7_no_development_task_ids_in_commit_messages():
    # A shallow clone (CI) holds only the fetched commits.
    _require_checkout()
    hits = flagged_commit_messages(ROOT, "HEAD")
    assert hits == [], f"task IDs, note IDs or phase names in commit messages: {hits}"


def test_RH8_commit_rules_are_not_vacuous(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    decoy = tmp_path / "decoy.git"
    monkeypatch.setenv("GIT_DIR", str(decoy))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    env = _isolated_git_env()

    def git(*args: str) -> str:
        return _git(*args, cwd=repo, env=env)

    def commit(message: str, files: dict[str, str | None]) -> None:
        for name, text in files.items():
            if text is None:
                git("rm", "-q", name)
                continue
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
            git("add", name)
        git("commit", "-q", "--allow-empty", "-m", message)

    # The flagged commits are reachable only through the second parent of a
    # merge, and what they add is deleted again before the clean tip.
    git("init", "-q")
    commit("feat: start", {"README.md": "clean\n", "guide.txt": "clean\n"})
    trunk = git("symbolic-ref", "--short", "HEAD").strip()
    git("checkout", "-q", "-b", "side")
    added = {
        "notes.md": "see T104-A01\nfrom /Users/someone/x\n",
        ".agent-state/task.json": "{}\n",
        "scripts/t104/probe.py": "x = 1\n",
    }
    commit("feat: notes\n\nPart of T104-A01.", added)
    commit("fix: drop the notes", {"notes.md": None, ".agent-state/task.json": None})
    (repo / "docs").mkdir()
    git("mv", "guide.txt", "docs/AGENTS.md")
    commit("refactor: move the guide", {})
    git("checkout", "-q", trunk)
    git("merge", "-q", "--no-ff", "--no-edit", "-m", "merge: side", "side")
    commit("docs: tip", {"README.md": "still clean\n"})

    def without_hash(hits: list[str]) -> list[str]:
        return sorted(hit.split(" ", 1)[1] for hit in hits)

    messages = flagged_commit_messages(repo, "HEAD", env=env)
    content = flagged_commit_content(repo, "HEAD", env=env)
    assert without_hash(messages) == ["Part of T104-A01."]
    assert without_hash(content) == [
        ".agent-state/task.json",
        "docs/AGENTS.md",
        "notes.md: from /Users/someone/x",
        "notes.md: see T104-A01",
        "scripts/t104/probe.py",
    ]
    assert not decoy.exists()


def test_RH9_no_development_task_ids_in_tracked_paths():
    bad = [name for name in _tracked_files() if development_name(name)]
    assert bad == [], f"tracked paths that name development work: {bad}"
