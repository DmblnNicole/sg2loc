"""Vendored sgaligner networks (kept upstream-as-is).

Localized shim: sgaligner's own modules import each other as top-level `aligner.*` (e.g.
`from aligner.networks.base import BaseNetwork`). Put this package's `src` dir on sys.path
so those internal imports resolve without editing the vendored files. First-party code imports
these via the full path `sg2loc.models.sgaligner.src.aligner.networks.*`.
"""

import os
import sys

_SGALIGNER_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SGALIGNER_SRC not in sys.path:
    sys.path.insert(0, _SGALIGNER_SRC)
