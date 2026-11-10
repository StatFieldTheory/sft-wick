"""An oracle for sft-wick that shares no code with the package.

:mod:`validation.modelspec` describes a stochastic field model and an
observable as plain data.  :mod:`validation.hierarchy` compiles that
description into the exact moment hierarchy of a finite Markov embedding
(``examples/reference/ito_moments.py`` and ``hormander_moments.py``) and
returns the moments order by order, channel by channel.

Only ``validation/lower.py`` and ``validation/lower_v042.py`` may import
sft_wick.  ``validation/lowering_parts.py`` holds what they share and may
not, and no other module of the oracle may import it.  ``tests/`` checks
this with an AST walk.
"""
