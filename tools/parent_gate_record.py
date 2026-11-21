"""The replay of the parent-commit gate over history, its record and its page.

Loaded by ``tools/parent_gate.py`` for its ``replay`` and ``render``
commands; the functions that judge or need the gate's tables take the gate
module as the argument ``gate``.  The population of a range is every
non-merge commit whose subject starts with ``fix`` and that adds or
modifies a ``tests/test_*.py`` (a renamed file counts as added under its new
path).  The record, ``docs/verification/parent_gate_replay.json``, holds
one entry per commit and the figures computed from them; the page,
``docs/verification/parent_gate.rst``, is rendered from the record alone.
Before either is written its text is checked with the local-path and
development-name rules of ``tests/test_repo_hygiene.py``.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import functools
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO_ROOT = TOOLS.parent
HYGIENE = REPO_ROOT / "tests" / "test_repo_hygiene.py"
FORMAT = 1
SELECTION = ("non-merge commits whose subject starts with fix and that add or modify "
             "tests/test_*.py (a renamed file counts as added under its new path)")
POPULATION_PATTERN = r"^tests/test_[^/]*\.py$"
#: Witnesses kept per commit in the record.
PUBLIC_WITNESSES = 5
RUN_NAMES = ("head", "parent_own", "overlay")
EM_DASH = chr(0x2014)
#: Two hyphens between spaces, used as a dash in a few subjects.
DOUBLE_HYPHEN = " " + "-" * 2 + " "


@dataclasses.dataclass(frozen=True)
class CommitInfo:
    """A commit of a population: its sha, subject and counted test files."""

    sha: str
    subject: str
    touched: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class DependencyTrigger:
    """A commit whose defect needs a dependency version the replay lacks.

    The defect occurs from ``minimum`` (inclusive) up to ``below``
    (exclusive); either may be None.  ``changelog_heading`` is the heading
    of ``CHANGELOG.md`` that states the version.
    """

    dependency: str
    minimum: str | None
    below: str | None
    changelog_heading: str


#: Full sha -> trigger.  An entry is added after a replay, for a commit whose
#: verdict is passes_on_parent and whose defect needs another dependency
#: version; the replay then classes it environment, side dependency.
DEPENDENCY_TRIGGERS: dict[str, DependencyTrigger] = {
    # Its test of hook registration with an existing worker pool states that
    # the defect appears from joblib 1.6, which uses a standalone cloudpickle.
    "25a2855b61bcb43054bb12a636c32fbebcd4ae9c": DependencyTrigger(
        dependency="joblib", minimum="1.6", below=None,
        changelog_heading="### Fixed: hook serialization with joblib 1.6"),
}


def _git(repo: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    proc = subprocess.run(["git", "-C", str(repo), *args], env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        raise RuntimeError(f"git {' '.join(args)}: {lines[-1] if lines else 'failed'}")
    return proc.stdout


# ---- Population ----

def population(repo: Path, rev_range: str, *, prefix: str = "fix",
               pattern: str = POPULATION_PATTERN) -> list[CommitInfo]:
    """The commits of ``rev_range`` the replay judges, oldest first."""
    log = _git(repo, "log", "--no-merges", "--reverse", "--format=%x00%H%x1f%s",
               "--name-status", "-M", rev_range)
    matcher = re.compile(pattern)
    commits = []
    for block in log.split("\x00")[1:]:
        head, _, rest = block.partition("\n")
        sha, subject = head.split("\x1f", 1)
        if not subject.startswith(prefix):
            continue
        touched = []
        for line in rest.splitlines():
            fields = line.split("\t")
            if len(fields) < 2 or fields[0][:1] not in "AMR":
                continue
            if matcher.match(fields[-1]) and fields[-1] not in touched:
                touched.append(fields[-1])
        if touched:
            commits.append(CommitInfo(sha, subject, tuple(touched)))
    return commits


def resolve_range(repo: Path, rev_range: str) -> str:
    """``rev_range`` with each end written as a full sha."""
    ends = rev_range.split("..")
    return "..".join(_git(repo, "rev-parse", "--verify", f"{end}^{{commit}}").strip()
                     for end in ends)


def _test_count(repo: Path, commit: CommitInfo) -> int:
    count = 0
    for path in commit.touched:
        text = _git(repo, "show", f"{commit.sha}:{path}")
        count += len(re.findall(r"^\s*(?:async\s+)?def test", text, flags=re.M))
    return count


def replay_order(repo: Path, commits: Sequence[CommitInfo],
                 first: Sequence[str]) -> list[CommitInfo]:
    """``first`` in the order given, then by ``def test`` nodes, most first."""
    by_sha = {c.sha: c for c in commits}
    leading = []
    for rev in first:
        sha = _git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()
        if sha in by_sha and by_sha[sha] not in leading:
            leading.append(by_sha[sha])
    rest = [c for c in commits if c not in leading]
    counts = {c.sha: _test_count(repo, c) for c in rest}
    return leading + sorted(rest, key=lambda c: -counts[c.sha])


# ---- A judgement as JSON and as Markdown ----

def _counts(results: Sequence, keys: Sequence[str]) -> dict[str, int]:
    counts = dict.fromkeys(keys, 0)
    for result in results:
        counts[result.outcome] += 1
    return counts


def _per_file(run, keys: Sequence[str]) -> dict[str, dict[str, int]]:
    files: dict[str, list] = {path: [] for path in run.files}
    for result in run.results:
        files.setdefault(result.id.split("::", 1)[0], []).append(result)
    return {path: _counts([r for r in results if "::" in r.id], keys)
            for path, results in files.items()}


def _run_json(run, gate: types.ModuleType, public: bool) -> dict | None:
    if run is None:
        return None
    data = {"files": list(run.files), "exit": run.exit, "has_report": run.has_report,
            "seconds": round(run.seconds, 3), "peak_rss_mb": round(run.peak_rss_mb, 1),
            "maxrss_mb": round(run.maxrss_mb, 1),
            "timed_out": run.timed_out, "memory_killed": run.memory_killed,
            "counts": _counts(run.results, gate.COUNT_KEYS),
            "per_file": _per_file(run, gate.COUNT_KEYS)}
    if not public:
        data.update(command=list(run.command), tail=list(run.tail))
    return data


def _test_rows(j, gate: types.ModuleType) -> list[dict]:
    """One row per test id of any run: its outcome on each tree and how it failed."""
    by_run = {name: {r.id: r for r in run.results} if run else None
              for name, run in j.runs.items()}
    ids: list[str] = []
    for results in by_run.values():
        ids += [i for i in (results or {}) if i not in ids]
    existing = set(by_run["parent_own"] or {})
    rows = []
    for test_id in ids:
        found = [by_run[name][test_id] for name in ("overlay", "parent_own", "head")
                 if by_run[name] and test_id in by_run[name]]
        shown = next((r for r in found if r.outcome in gate.FAILING), found[0])
        row = {"id": test_id}
        for name in RUN_NAMES:
            results = by_run[name]
            row[name] = (None if results is None
                         else results[test_id].outcome if test_id in results else "absent")
        row.update(phase=shown.phase, kind=shown.kind, exception=shown.exception,
                   message=shown.message, existing=test_id in existing)
        rows.append(row)
    return rows


def judgement_json(j, gate: types.ModuleType, *, public: bool = False) -> dict:
    """A judgement as JSON; ``public`` drops messages, output and commands."""
    rows = _test_rows(j, gate)
    witnesses = [w.id for w in j.witnesses]
    if public:
        witnesses = witnesses[:PUBLIC_WITNESSES]
        rows = [{k: v for k, v in row.items() if k != "message"}
                for row in rows if row["id"] in witnesses]
    return {
        "format": gate.FORMAT, "rules_digest": gate.rules_digest(),
        "head": j.head, "parent": j.parent, "base": j.base, "parent_rule": j.parent_rule,
        "subject": j.subject, "verdict": j.verdict, "decision": j.decision,
        "side": j.side, "cause": j.cause, "reason": j.reason,
        "src_changed": j.src_changed, "package": j.package,
        "touched": [dataclasses.asdict(t) for t in j.touched],
        "runs": {name: _run_json(run, gate, public) for name, run in j.runs.items()},
        "tests": rows, "witnesses": witnesses,
        "settings": dataclasses.asdict(j.settings), "environment": dict(j.environment),
    }


def _summary_counts(run, path: str, failing: Sequence[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    if run is None:
        return found
    for result in run.results:
        if result.id.split("::", 1)[0] != path or "::" not in result.id:
            continue
        key = result.kind if result.outcome in failing else result.outcome
        key = "time or memory" if key in ("timeout", "memory") else key
        found[key] = found.get(key, 0) + 1
    return found


def _summary_table(j, failing: Sequence[str]) -> list[str]:
    lines = ["| file | head: passed | parent's own: passed / failed | on the parent's "
             "source: assertion / crash / missing name / time or memory / absent / passed |",
             "|---|---|---|---|"]
    overlay_keys = ("assertion", "crash", "missing_api", "time or memory", "absent", "passed")
    for touched in j.touched:
        head = _summary_counts(j.runs["head"], touched.path, failing)
        own = _summary_counts(j.runs["parent_own"], touched.old_path or "", failing)
        over = _summary_counts(j.runs["overlay"], touched.path, failing)
        not_passed = sum(v for k, v in own.items() if k not in ("passed", "skipped"))
        own_text = (f"{own.get('passed', 0)} / {not_passed}"
                    if j.runs["parent_own"] and touched.old_path else "not run")
        over_text = (" / ".join(str(over.get(k, 0)) for k in overlay_keys)
                     if j.runs["overlay"] else "not run")
        head_text = head.get("passed", 0) if j.runs["head"] else "not run"
        lines.append(f"| `{touched.path}` | {head_text} | {own_text} | {over_text} |")
    return lines


def summary_markdown(j, gate: types.ModuleType) -> str:
    """A judgement as Markdown for a job summary."""
    parent = ("merge base of the base branch and the head" if j.parent_rule == "merge_base"
              else "first parent of the head")
    lines = [f"### Parent-commit gate: {j.verdict} ({j.decision})", "",
             f"Head {j.head[:7]}, parent {j.parent[:7]} ({parent}).", "", j.reason + "."]
    if j.verdict == "crash_only":
        lines.append("The parent's source fails these tests only by raising; the gate "
                     "cannot tell a fixed crash from an option the parent refuses.")
    if j.touched and j.runs["head"] is not None:
        lines += ["", *_summary_table(j, gate.FAILING)]
    if j.witnesses:
        shown = ", ".join(f"`{w.id}` ({w.exception})" for w in j.witnesses[:PUBLIC_WITNESSES])
        lines += ["", f"Witnesses: {shown}"]
    return "\n".join(lines) + "\n"


# ---- Replay ----

def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _keep_reports(work: Path, out_dir: Path, sha: str) -> None:
    for run in RUN_NAMES:
        for suffix, folder, ext in (("xml", "junit", "xml"), ("log", "logs", "txt")):
            source = work / f"{run}.{suffix}"
            if source.is_file():
                shutil.move(str(source), out_dir / folder / f"{sha}-{run}.{ext}")


def _judge_one(commit: CommitInfo, *, repo: Path, gate: types.ModuleType, out_dir: Path,
               settings, expected: Mapping, keep_trees: bool, log) -> dict | None:
    path = out_dir / "commits" / f"{commit.sha}.json"
    old = _read_json(path)
    if old is not None and all(old.get(k) == v for k, v in expected.items()):
        log("reused", _now(), _now(), commit.sha, old["verdict"],
            gate.DECISION[old["verdict"]])
        return old
    start, began = _now(), time.monotonic()
    work = out_dir / "trees" / commit.sha
    shutil.rmtree(work, ignore_errors=True)
    try:
        judgement = gate.judge(repo, commit.sha, settings=settings, work_dir=work,
                               keep_trees=keep_trees)
    except (gate.GateError, gate.tree.HarnessError, OSError, RuntimeError) as exc:
        log("error", start, _now(), commit.sha, str(exc).replace("\t", " "))
        return None
    finally:
        _keep_reports(work, out_dir, commit.sha)
        if not keep_trees:
            shutil.rmtree(work, ignore_errors=True)
    full = gate.to_json(judgement)
    seconds = time.monotonic() - began
    full["replay"] = {"start": start, "end": _now(), "seconds": round(seconds, 3)}
    _write_json(path, full)
    runs = [r for r in full["runs"].values() if r]
    head = full["runs"]["head"]
    overlay = full["runs"]["overlay"]
    log("run", start, full["replay"]["end"], commit.sha, full["verdict"], full["decision"],
        f"{seconds:.1f}", f"{max([r['peak_rss_mb'] for r in runs] or [0.0]):.0f}",
        str(head["counts"]["passed"] if head else 0),
        str(sum(overlay["counts"][k] for k in ("failed", "error", "not_run")) if overlay else 0))
    return full


def replay(repo: Path, commits: Sequence[CommitInfo], *, gate: types.ModuleType,
           out_dir: Path, jobs: int = 2, first: Sequence[str] = (), settings=None,
           keep_trees: bool = False) -> dict[str, dict | None]:
    """Judge ``commits``, ``jobs`` at a time; sha -> per-commit JSON, None on a tool error.

    A commit whose ``commits/<sha>.json`` under ``out_dir`` was made with the
    same rules digest, settings and environment is not judged again.  At
    most ``jobs`` runs are live at a time, and each is killed above the
    memory limit of ``settings``.  When the replay is stopped (a signal, an exception in this
    thread) it kills the live runs, starts no further commit, writes an
    ``interrupted`` line to ``replay.log``, clears the stop once its threads
    have ended, so later runs in the process start, and raises again.
    """
    settings = settings if settings is not None else gate.Settings()
    for folder in ("commits", "junit", "logs", "trees"):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
    expected = {"rules_digest": gate.rules_digest(),
                "settings": dataclasses.asdict(settings),
                "environment": gate.environment_versions()}
    lock = threading.Lock()

    def log(*fields: str) -> None:
        with lock, open(out_dir / "replay.log", "a", encoding="utf-8") as sink:
            sink.write("\t".join(fields) + "\n")

    order = replay_order(repo, commits, first)
    one = functools.partial(_judge_one, repo=repo, gate=gate, out_dir=out_dir,
                            settings=settings, expected=expected, keep_trees=keep_trees,
                            log=log)
    jobs = max(1, jobs)
    print(f"replay: {len(order)} commits, {jobs} at a time; each run is killed above "
          f"{settings.max_rss_gb:g} GB of {gate.MEMORY_MEASURE}, so up to "
          f"{jobs * settings.max_rss_gb:g} GB for {jobs} runs", flush=True)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
    try:
        results = list(pool.map(one, order))
    except BaseException as exc:
        gate.run_tools.stop_all()
        pool.shutdown(wait=True, cancel_futures=True)
        gate.run_tools.clear_stop()
        log("interrupted", _now(), _now(), type(exc).__name__)
        raise
    pool.shutdown(wait=True)
    return {commit.sha: result for commit, result in zip(order, results)}


# ---- Record ----

def public_subject(subject: str) -> str:
    """``subject`` with an em dash or :data:`DOUBLE_HYPHEN` written as ``", "``."""
    return re.sub(rf"\s*{EM_DASH}\s*", ", ", subject).replace(DOUBLE_HYPHEN, ", ")


def commit_entry(full: Mapping, decisions: Mapping[str, str]) -> dict:
    """The record's entry for one per-commit judgement."""
    rows = {row["id"]: row for row in full.get("tests", [])}
    witnesses = [{"id": i, "phase": rows[i]["phase"], "kind": rows[i]["kind"],
                  "exception": rows[i]["exception"], "existing": rows[i]["existing"]}
                 for i in full["witnesses"][:PUBLIC_WITNESSES] if i in rows]
    runs = full["runs"]
    per_file = {}
    for touched in full["touched"]:
        names = {"head": touched["path"], "parent_own": touched["old_path"],
                 "overlay": touched["path"]}
        per_file[touched["path"]] = {
            run: (runs[run]["per_file"].get(name) if runs.get(run) and name else None)
            for run, name in names.items()}
    present = [r for r in runs.values() if r]
    seconds = full.get("replay", {}).get("seconds", sum(r["seconds"] for r in present))
    return {
        "sha": full["head"], "subject": public_subject(full["subject"]),
        "verdict": full["verdict"], "gate_verdict": full["verdict"],
        "decision": decisions[full["verdict"]], "side": full["side"], "cause": full["cause"],
        "reason": full["reason"], "seconds": seconds,
        "peak_rss_mb": max([r["peak_rss_mb"] for r in present] or [0.0]),
        "touched": [t["path"] for t in full["touched"]], "per_file": per_file,
        "witnesses": witnesses,
    }


def apply_trigger(entry: Mapping, trigger: DependencyTrigger | None) -> dict:
    """``entry`` classed environment (side dependency) when ``trigger`` applies."""
    entry = dict(entry)
    if trigger is None:
        return entry
    if entry["gate_verdict"] != "passes_on_parent":
        entry["trigger_unused"] = True
        return entry
    entry.update(verdict="environment", side="dependency", cause="failure",
                 decision="undetermined",
                 reason=f"the defect needs {trigger.dependency} "
                        f"{_version_range(trigger)}, which the replay environment lacks")
    return entry


def _version_tuple(version: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(?:\.\d+)*", version):
        raise ValueError(f"version {version!r} is not a dotted list of integers")
    return tuple(int(part) for part in version.split("."))


def version_in_range(version: str, minimum: str | None, below: str | None) -> bool:
    """Whether ``version`` lies in ``[minimum, below)``; an absent dependency does not."""
    if version == "absent":
        return False
    value = _version_tuple(version)
    return ((minimum is None or value >= _version_tuple(minimum))
            and (below is None or value < _version_tuple(below)))


def _version_range(trigger: DependencyTrigger) -> str:
    parts = [f">= {trigger.minimum}"] if trigger.minimum else []
    parts += [f"< {trigger.below}"] if trigger.below else []
    return ", ".join(parts)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def section(range_text: str, entries: Sequence[Mapping], *, counted: bool,
            spec: str | None = None) -> dict:
    """One range of the record: its entries and the figures computed from them."""
    by_verdict: dict[str, int] = {}
    for entry in entries:
        by_verdict[entry["verdict"]] = by_verdict.get(entry["verdict"], 0) + 1
    env = [e for e in entries if e["verdict"] == "environment"]
    environment = {"total": len(env)}
    environment.update({side: sum(e["side"] == side for e in env)
                        for side in ("head", "parent", "dependency")})
    environment.update({cause: sum(e["cause"] == cause for e in env)
                        for cause in ("failure", "timeout", "memory")})
    d = len(entries) - len(env)
    a, k = by_verdict.get("accepted", 0), by_verdict.get("no_code_change", 0)
    c, o = by_verdict.get("crash_only", 0), environment["dependency"]
    passed = sum(1 for e in entries if e["verdict"] != "environment" and e["decision"] == "pass")
    return {
        "range": range_text, "spec": spec or range_text, "selection": SELECTION,
        "counted": counted, "commits": [dict(e) for e in entries],
        "population": len(entries), "environment": environment, "denominator": d,
        "by_verdict": dict(sorted(by_verdict.items())), "accepted": passed,
        "rate": _ratio(passed, d),
        "figures": {
            "crash_only rejected, no_code_change passes": _ratio(a + k, d),
            "crash_only passes, no_code_change passes": _ratio(a + k + c, d),
            "crash_only rejected, no_code_change left out": _ratio(a, d - k),
            "crash_only passes, no_code_change left out": _ratio(a + c, d - k),
            "dependency triggers counted as rejections": _ratio(a + k, d + o),
        },
    }


def build_record(gate: types.ModuleType, *, counted: dict, earlier: dict | None,
                 settings: Mapping, environment: Mapping) -> dict:
    return {"format": FORMAT, "rules_digest": gate.rules_digest(),
            "settings": dict(settings), "environment": dict(environment),
            "decisions": dict(gate.DECISION), "counted": counted, "earlier": earlier}


@functools.lru_cache(maxsize=None)
def _hygiene() -> types.ModuleType:
    """``tests/test_repo_hygiene.py``, loaded by path for its patterns."""
    spec = importlib.util.spec_from_file_location("_parent_gate_hygiene", HYGIENE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strings(value, where: str):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield f"{where}.{key} (key)", str(key)
            yield from _strings(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _strings(item, f"{where}[{index}]")
    elif isinstance(value, str):
        yield where, value


def publish_problems(record: Mapping, page: str | None = None) -> list[str]:
    """Fields of the record and lines of the page that must not be published."""
    hygiene = _hygiene()

    def bad(text: str) -> bool:
        return bool(hygiene.LOCAL_PATH.search(text)) or hygiene.development_name(text) \
            or EM_DASH in text

    problems = [f"record {where}: {text[:120]}"
                for where, text in _strings(record, "record") if bad(text)]
    for number, line in enumerate((page or "").splitlines(), 1):
        if bad(line):
            problems.append(f"page line {number}: {line[:120]}")
    return problems


def write_record(record: Mapping, path: Path) -> int:
    """Write ``record`` as JSON; 2, writing nothing, if it holds what must not be published."""
    problems = publish_problems(record, render_page(record))
    if problems:
        for problem in problems:
            print(f"parent_gate: refused: {problem}", file=sys.stderr)
        return 2
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 0


def trigger_versions() -> dict[str, str]:
    """The installed version of every dependency a trigger names."""
    import importlib.metadata
    versions = {}
    for trigger in DEPENDENCY_TRIGGERS.values():
        try:
            versions[trigger.dependency] = importlib.metadata.version(trigger.dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[trigger.dependency] = "absent"
    return versions


def replay_main(args, gate: types.ModuleType) -> int:
    """The ``replay`` command of ``tools/parent_gate.py``."""
    repo = Path(args.repo).resolve()
    settings = gate._settings_from(args)
    counted = population(repo, args.rev_range)
    earlier = population(repo, args.earlier_range) if args.earlier_range else []
    every = counted + [c for c in earlier if c not in counted]
    results = replay(repo, every, gate=gate, out_dir=Path(args.out).resolve(), jobs=args.jobs,
                     first=args.first, settings=settings, keep_trees=args.keep_trees)
    missing = [c.sha for c in every if results.get(c.sha) is None]
    print(f"replay: {len(every) - len(missing)} of {len(every)} commits judged; "
          f"log in {Path(args.out) / 'replay.log'}")
    if missing:
        print(f"replay: no result for {', '.join(s[:12] for s in missing)}", file=sys.stderr)
        return 2
    if args.record is None:
        return 0

    def entries(commits: Sequence[CommitInfo]) -> list[dict]:
        return [apply_trigger(commit_entry(results[c.sha], gate.DECISION),
                              DEPENDENCY_TRIGGERS.get(c.sha)) for c in commits]

    record = build_record(
        gate,
        counted=section(resolve_range(repo, args.rev_range), entries(counted),
                        counted=True, spec=args.rev_range),
        earlier=(section(resolve_range(repo, args.earlier_range), entries(earlier),
                         counted=False, spec=args.earlier_range)
                 if args.earlier_range else None),
        settings=dataclasses.asdict(settings),
        environment={**gate.environment_versions(), **trigger_versions()})
    return write_record(record, Path(args.record))


def render_main(args, gate: types.ModuleType) -> int:
    """The ``render`` command of ``tools/parent_gate.py``."""
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    page = render_page(record)
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out and args.out.exists() else ""
        if current != page:
            print(f"{args.out} is stale; run python tools/parent_gate.py render")
            return 1
        return 0
    problems = publish_problems({}, page)
    if problems:
        for problem in problems:
            print(f"parent_gate: refused: {problem}", file=sys.stderr)
        return 2
    if args.out is None:
        sys.stdout.write(page)
    else:
        args.out.write_text(page, encoding="utf-8")
    return 0


# ---- Page ----

def _load_page():
    """``tools/parent_gate_page.py``, loaded by path."""
    spec = importlib.util.spec_from_file_location("_parent_gate_page", TOOLS / "parent_gate_page.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The page of a record: :func:`parent_gate_page.render_page`.
render_page = _load_page().render_page
