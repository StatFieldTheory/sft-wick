"""An oracle for sft-wick that shares no code with the package.

:mod:`validation.modelspec` describes a stochastic field model and an
observable as plain data.  :mod:`validation.hierarchy` compiles that
description into the exact moment hierarchy of a finite Markov embedding
(``examples/reference/ito_moments.py`` and ``hormander_moments.py``) and
returns the moments order by order, channel by channel.

Only ``validation/lower*.py`` may import sft_wick; ``tests/`` checks this
with an AST walk.
"""
