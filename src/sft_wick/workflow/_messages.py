"""Values in error messages."""

from __future__ import annotations

from typing import Any

#: Length past which an error message shows a stand-in for a value.
MAX_SHOWN_CHARS = 1000
#: Bits past which ``repr`` of an integer can raise: below the 640 digits of
#: the smallest limit ``sys.set_int_max_str_digits`` allows.
MAX_INT_BITS = 2000


def shown(value: Any) -> str:
    """``repr(value)`` for an error message, or a stand-in such as
    ``<list of length 2>`` past :data:`MAX_SHOWN_CHARS` characters.

    A value read from a YAML file can share nodes through aliases, and its
    ``repr`` expands every path to them, so a few lines can ask for more
    text than fits in memory.  The length is counted first, by a walk that
    stops at the limit, so the cost is bounded whatever the value holds.
    The count is the length of the ``repr`` of a scalar and a little above
    that of a container.  An integer past :data:`MAX_INT_BITS` bits gets the
    stand-in too, since its ``repr`` can raise in place of the message.
    """
    left, pending = MAX_SHOWN_CHARS, [value]
    while pending and left >= 0:
        item = pending.pop()
        if isinstance(item, dict):
            left -= 2 + 4 * len(item)
            pending.extend(item)
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            left -= 2 + 2 * len(item)
            pending.extend(item)
        elif isinstance(item, int) and item.bit_length() > MAX_INT_BITS:
            left = -1
        else:
            left -= len(repr(item))
    if left >= 0:
        return repr(value)
    length = f" of length {len(value)}" if hasattr(value, "__len__") else ""
    return f"<{type(value).__name__}{length}>"


def text_of(value: Any) -> str:
    """``str(value)`` for a value read from a file, where a list can stand
    in place of text.  Bounded as :func:`shown` is; ``str()`` of a list or an
    integer is its ``repr``, and every other type keeps its ``str()``.
    """
    return shown(value) if isinstance(value, (dict, list, tuple, int)) else str(value)
