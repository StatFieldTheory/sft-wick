"""Data types of the parent-commit gate and the reading of a JUnit report.

Loaded by ``tools/parent_gate.py``, which re-exports these names.  The
definitions here decide verdicts (the kind of a failure is read here), so
:func:`rules_digest` of ``tools/parent_gate.py`` covers this file too.
Standard library only.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

MESSAGE_LIMIT = 300
#: Exception classes whose failure in a test or fixture is a witness.
ASSERTION_CLASSES = ("AssertionError", "Failed")
#: Exception classes that can name something the parent's source lacks.
MISSING_NAME_CLASSES = ("ImportError", "ModuleNotFoundError", "AttributeError", "NameError")
#: Outcomes of a test that did not pass.
FAILING = ("failed", "error", "not_run")
COUNT_KEYS = ("passed", "failed", "error", "skipped", "not_run", "absent")

_CANNOT_IMPORT = re.compile(r"cannot import name '([^']+)' from '([^']+)'")
_NO_MODULE = re.compile(r"No module named '([^']+)'")
_MODULE_OBJECT_ATTR = re.compile(r"<module '([^']+)' from [^>]*> has no attribute '([^']+)'")
#: The message of ``monkeypatch.setattr("package.name", ...)``.
_MODULE_AT_ATTR = re.compile(r"'module' object at ([\w.]+) has no attribute '([^']+)'")
_MODULE_ATTR = re.compile(r"module '([^']+)' has no attribute '([^']+)'")
_ANY_ATTR = re.compile(r"(?:has no attribute|does not have the attribute) '([^']+)'")
_NOT_DEFINED = re.compile(r"name '([^']+)' is not defined")
_KEYWORD = re.compile(r"(?:([\w.<>]+)\(\) )?got an unexpected keyword argument '([^']+)'")
_CLASS_TOKEN = re.compile(r"([A-Za-z_][\w.]*)(?::\s|:$|$)")
_E_LINE = re.compile(r"^E\s+(.*)$")
_FRAME = re.compile(r"^(\S.*?):(\d+): in (\S.*)$")
#: The lines pytest writes between the exceptions of a chained traceback.
CHAIN_SEPARATORS = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:")


# ---- Data ----

@dataclasses.dataclass(frozen=True)
class Settings:
    """How the gate runs the tests.

    ``time_limit`` is the limit of the head run in seconds; the parent's own
    run and the overlay run get ``min(time_limit, max(overlay_floor,
    overlay_factor * head seconds))``.  ``max_rss_gb`` is the limit on the
    summed memory of a run's processes, measured as ``MEMORY_MEASURE`` of
    ``tools/parent_gate_run.py`` says (the physical footprint on macOS).
    ``package`` is the import name of the package; None reads it from the
    head's tree.
    """

    markers: str = "not slow"
    time_limit: float = 3600.0
    overlay_factor: float = 3.0
    overlay_floor: float = 300.0
    max_rss_gb: float = 6.0
    pytest_workers: int = 0
    hash_seed: str = "0"
    loky_max_cpu: int = 2
    source_root: str = "src"
    package: str | None = None


@dataclasses.dataclass(frozen=True)
class Touched:
    """A test file the change adds (A), modifies (M) or renames (R)."""

    path: str
    status: str
    old_path: str | None


@dataclasses.dataclass(frozen=True)
class TestResult:
    """One test id in one run.

    ``outcome`` is passed, failed, error, skipped, not_run or absent; a
    collection error has the file as its id.  ``kind`` is assertion, crash,
    missing_api, timeout or memory for a test that did not pass.
    ``existing`` is set on a witness: the parent's own run collected the id.
    """

    __test__ = False

    id: str
    outcome: str
    phase: str | None = None
    kind: str | None = None
    exception: str | None = None
    message: str = ""
    existing: bool | None = None


@dataclasses.dataclass(frozen=True)
class Run:
    """One pytest run of the touched files on one tree.

    ``peak_rss_mb`` is the largest polled memory of the run's processes and
    ``maxrss_mb`` the largest resident set of the pytest process alone.
    """

    name: str
    files: tuple[str, ...]
    exit: int | None
    has_report: bool
    seconds: float
    peak_rss_mb: float
    timed_out: bool
    memory_killed: bool
    results: tuple[TestResult, ...]
    tail: tuple[str, ...]
    command: tuple[str, ...] = ()
    maxrss_mb: float = 0.0


@dataclasses.dataclass(frozen=True)
class Verdict:
    """The result of :func:`decide`."""

    verdict: str
    side: str | None
    cause: str | None
    reason: str
    witnesses: tuple[TestResult, ...] = ()


@dataclasses.dataclass(frozen=True)
class Judgement:
    """The gate's judgement of one change."""

    head: str
    parent: str
    base: str | None
    parent_rule: str
    subject: str
    src_changed: bool
    touched: tuple[Touched, ...]
    runs: Mapping[str, Run | None]
    verdict: str
    decision: str
    side: str | None
    cause: str | None
    reason: str
    witnesses: tuple[TestResult, ...]
    settings: Settings
    environment: Mapping[str, str]
    package: str | None = None


@dataclasses.dataclass(frozen=True)
class Names:
    """What the source under a source root defines, read from its syntax."""

    modules: frozenset[str]
    module_level: Mapping[str, frozenset[str]]
    class_level: frozenset[str]
    parameters: Mapping[str, frozenset[str]]

    def in_module(self, module: str, name: str) -> bool:
        return (name in self.module_level.get(module, frozenset())
                or f"{module}.{name}" in self.modules)

    def anywhere(self, name: str) -> bool:
        return any(name in names for names in self.module_level.values())


# ---- Names a source tree defines ----

def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for element in target.elts for n in _target_names(element)]
    return []


def _statement_names(body: Sequence[ast.stmt]) -> set[str]:
    """Names bound at the level of ``body``, through if, try, with and loops."""
    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(n for t in node.targets for n in _target_names(t))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            names.update(_target_names(node.target))
        elif isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names if a.name != "*")
        for block in ("body", "orelse", "finalbody"):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.update(_statement_names(getattr(node, block, None) or []))
        for handler in getattr(node, "handlers", None) or []:
            names.update(_statement_names(handler.body))
    return names


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = node.args
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    return {a.arg for a in every if a is not None}


def _is_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name == "dataclass":
            return True
    return False


def _class_parameters(node: ast.ClassDef) -> set[str]:
    """The keywords a call of the class takes: its ``__init__``, ``__new__`` and fields."""
    found: set[str] = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and item.name in ("__init__", "__new__"):
            found.update(_parameters(item))
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) \
                and _is_dataclass(node):
            found.add(item.target.id)
    return found


def _function_parameters(tree: ast.Module, parameters: dict[str, set[str]]) -> None:
    """File each function's parameters under its name, and each class's under the class."""
    methods = {id(item) for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
               for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
               and item.name in ("__init__", "__new__")}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if _is_dataclass(node) or any(id(item) in methods for item in node.body):
                parameters.setdefault(node.name, set()).update(_class_parameters(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and id(node) not in methods:
            parameters.setdefault(node.name, set()).update(_parameters(node))


def defined_names(src_dir: Path) -> Names:
    """Modules, module-level names, class-level names and parameters under ``src_dir``."""
    modules: set[str] = set()
    module_level: dict[str, set[str]] = {}
    class_level: set[str] = set()
    parameters: dict[str, set[str]] = {}
    for path in sorted(src_dir.rglob("*.py")):
        parts = path.relative_to(src_dir).with_suffix("").parts
        parts = parts[:-1] if parts[-1] == "__init__" else parts
        if not parts:
            continue
        try:
            parsed = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        module = ".".join(parts)
        modules.add(module)
        module_level.setdefault(module, set()).update(_statement_names(parsed.body))
        if len(parts) > 1:
            module_level.setdefault(".".join(parts[:-1]), set()).add(parts[-1])
        _function_parameters(parsed, parameters)
        for node in ast.walk(parsed):
            if isinstance(node, ast.ClassDef):
                class_level.update(_statement_names(node.body))
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                class_level.add(node.attr)
    return Names(frozenset(modules),
                 {m: frozenset(n) for m, n in module_level.items()},
                 frozenset(class_level),
                 {f: frozenset(p) for f, p in parameters.items()})


# ---- Reading a report ----

def _first_line(text: str) -> str:
    return text.strip().split("\n", 1)[0][:MESSAGE_LIMIT]


def _class_from_message(message: str) -> str | None:
    first = message.strip().split("\n", 1)[0]
    if first.startswith("[XPASS(strict)]"):
        return "Failed"
    if first.startswith("assert ") or first == "assert":
        return "AssertionError"
    match = _CLASS_TOKEN.match(first)
    return match.group(1).rsplit(".", 1)[-1] if match else None


def _class_from_text(text: str) -> tuple[str | None, str]:
    """The last ``E   Name:`` line before the first blank ``E`` line."""
    found: tuple[str | None, str] = (None, "")
    for line in text.splitlines():
        match = _E_LINE.match(line.rstrip())
        if line.rstrip() == "E":
            break
        if not match:
            continue
        content = match.group(1)
        if content.startswith("assert "):
            found = ("AssertionError", content)
            continue
        token = _CLASS_TOKEN.match(content)
        if token and (":" in content or content == token.group(1)):
            found = (token.group(1).rsplit(".", 1)[-1], content)
    return found


def exception_of(element) -> tuple[str | None, str, str]:
    """Exception class, phase and message of a ``failure`` or ``error`` element."""
    message = element.get("message") or ""
    text = element.text or ""
    if element.tag == "failure":
        phase = "call"
    elif message == "collection failure":
        phase = "collection"
    else:
        phase = "setup"
        for prefix, named in (('failed on setup with "', "setup"),
                              ('failed on teardown with "', "teardown")):
            if message.startswith(prefix):
                phase, message = named, message[len(prefix):]
                message = message[:-1] if message.endswith('"') else message
                break
    found = _class_from_message(message) if phase != "collection" else None
    if found is None:
        found, line = _class_from_text(text)
        message = line or message
    return found, phase, message


def last_exception(text: str) -> str:
    """The traceback text of the last exception of a chained traceback."""
    lines = text.splitlines()
    cut = max((i for i, line in enumerate(lines) if line.strip() in CHAIN_SEPARATORS),
              default=-1)
    return "\n".join(lines[cut + 1:])


def source_frame(text: str, tree_dir: Path, source_root: str) -> bool:
    """Whether the frame that raised the last exception lies under the source root.

    The frame is the last ``path:line: in name`` line before the first ``E``
    line of the last exception of the traceback: the one that failed the test.
    """
    frame = None
    for line in last_exception(text).splitlines():
        if line.startswith("E "):
            break
        match = _FRAME.match(line)
        if match:
            frame = match.group(1)
    if frame is None:
        return False
    path = Path(frame)
    if path.is_absolute():
        for base in (tree_dir, tree_dir.resolve()):
            try:
                path = path.relative_to(base)
                break
            except ValueError:
                continue
        else:
            return False
    return path.as_posix().startswith(source_root.rstrip("/") + "/")


def _called_name(called: str) -> str:
    """The name a call's parameters are filed under: the class for ``K.__init__``."""
    parts = called.split(".")
    if len(parts) > 1 and parts[-1] in ("__init__", "__new__"):
        return parts[-2]
    return parts[-1]


def missing_name(exception: str | None, message: str, head_names: Names,
                 parent_names: Names) -> bool:
    """Whether the error names something the head's source defines and the parent's lacks."""
    def added(check) -> bool:
        return check(head_names) and not check(parent_names)

    if exception == "TypeError":
        match = _KEYWORD.search(message)
        if not match:
            return False
        function, keyword = _called_name(match.group(1) or ""), match.group(2)

        def takes(names: Names) -> bool:
            if function in names.parameters:
                return keyword in names.parameters[function]
            return any(keyword in params for params in names.parameters.values())
        return added(takes)
    if exception not in MISSING_NAME_CLASSES:
        return False
    for pattern, check in (
            (_CANNOT_IMPORT, lambda m: lambda n: n.in_module(m.group(2), m.group(1))),
            (_NO_MODULE, lambda m: lambda n: m.group(1) in n.modules),
            (_MODULE_OBJECT_ATTR, lambda m: lambda n: n.in_module(m.group(1), m.group(2))),
            (_MODULE_ATTR, lambda m: lambda n: n.in_module(m.group(1), m.group(2))),
            (_MODULE_AT_ATTR, lambda m: lambda n: n.in_module(m.group(1), m.group(2))),
            (_ANY_ATTR, lambda m: lambda n: m.group(1) in n.class_level),
            (_NOT_DEFINED, lambda m: lambda n: n.anywhere(m.group(1)))):
        match = pattern.search(message)
        if match:
            return added(check(match))
    return False


def failure_kind(exception: str | None, phase: str, message: str, *, in_source: bool,
                 head_names: Names, parent_names: Names) -> str:
    """missing_api, assertion or crash, the first rule that applies.

    A missing module, attribute or name raised inside the source root is the
    parent's own failure, a crash; a keyword the parent's function lacks is
    a missing name wherever it is raised.
    """
    if (exception == "TypeError" or not in_source) \
            and missing_name(exception, message, head_names, parent_names):
        return "missing_api"
    if phase in ("call", "setup") and exception in ASSERTION_CLASSES and not in_source:
        return "assertion"
    return "crash"


def _case_id(file: str, classname: str, name: str) -> str:
    module = file[:-3].replace("/", ".") if file.endswith(".py") else ""
    cls = ""
    if module and (classname == module or classname.startswith(module + ".")):
        cls = classname[len(module):].lstrip(".")
    return "::".join([file, *(cls.split(".") if cls else []), name])


def _case_result(case, tree_dir: Path, source_root: str, head_names: Names,
                 parent_names: Names) -> TestResult:
    file = case.get("file") or ""
    name = case.get("name") or ""
    element = case.find("failure")
    element = element if element is not None else case.find("error")
    if element is not None and element.get("message") == "collection failure":
        file = file or name.replace(".", "/") + ".py"
        test_id = file
    else:
        test_id = _case_id(file, case.get("classname") or "", name)
    if element is None:
        outcome = "skipped" if case.find("skipped") is not None else "passed"
        return TestResult(test_id, outcome)
    exception, phase, message = exception_of(element)
    in_source = source_frame(element.text or "", tree_dir, source_root)
    kind = failure_kind(exception, phase, message, in_source=in_source,
                        head_names=head_names, parent_names=parent_names)
    outcome = "failed" if element.tag == "failure" else "error"
    return TestResult(test_id, outcome, phase, kind, exception, _first_line(message))


_SEVERITY = {"passed": 0, "skipped": 1, "error": 2, "failed": 3}


def read_report(cases, tree_dir: Path, *, expected: Sequence[TestResult] = (),
                head_names: Names, parent_names: Names, source_root: str,
                killed: str | None) -> tuple[TestResult, ...]:
    """The results of the test cases of a JUnit report, with the ids of ``expected`` it lacks.

    ``cases`` is None when the run wrote no report.

    An id of ``expected`` (the head run's results) missing from the report
    is ``not_run`` when its file did not import (with that error's kind) or
    when the run was killed (kind ``timeout`` or ``memory``), and ``absent``
    when it passed on the head and the run was not killed.
    """
    results: dict[str, TestResult] = {}
    collection: dict[str, TestResult] = {}
    for case in cases or ():
        result = _case_result(case, tree_dir, source_root, head_names, parent_names)
        if result.phase == "collection":
            collection[result.id] = result
        previous = results.get(result.id)
        if previous is None or _SEVERITY[result.outcome] > _SEVERITY.get(previous.outcome, 0):
            results[result.id] = result
    for wanted in expected:
        if "::" not in wanted.id or wanted.id in results:
            continue
        error = collection.get(wanted.id.split("::", 1)[0])
        if error is not None:
            results[wanted.id] = dataclasses.replace(error, id=wanted.id, outcome="not_run")
        elif killed is not None:
            results[wanted.id] = TestResult(wanted.id, "not_run", None, killed, None,
                                            f"the run was killed at its {killed} limit")
        elif wanted.outcome == "passed":
            results[wanted.id] = TestResult(wanted.id, "absent")
    return tuple(results[key] for key in sorted(results))
