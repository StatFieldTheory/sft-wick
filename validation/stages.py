"""Where a lowering failed: :func:`stage` marks an exception with the step
that raised it (``lower``, ``expand``, ``propagators``, ``evaluate``), so
that ``validation/worker.py`` can report it.  Imports no sft-wick code.
"""
from __future__ import annotations

import contextlib
from typing import Iterator

__all__ = ["STAGE_ATTR", "stage"]

#: The attribute that carries the stage on an exception.
STAGE_ATTR = "validation_stage"


@contextlib.contextmanager
def stage(name: str) -> Iterator[None]:
    """Mark an exception raised inside with ``name``, unless an inner stage
    already marked it."""
    try:
        yield
    except BaseException as exc:
        if not hasattr(exc, STAGE_ATTR):
            try:
                setattr(exc, STAGE_ATTR, name)
            except AttributeError:
                pass
        raise
