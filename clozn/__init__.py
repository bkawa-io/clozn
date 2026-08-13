"""clozn -- the product Python backend.

The package is being reorganized around product domains: CLI, runs, receipts, replay, behavior, readouts,
memory, substrates, and server glue. See RUNTIME_SPLIT.md for how this relates to engine/.
"""

# The single source of version truth: root pyproject.toml reads this via
# `[tool.setuptools.dynamic] version = {attr = "clozn.__version__"}` instead of duplicating the string, so
# a release only ever needs one edit. `clozn version` (clozn/cli/commands/version.py) prints it verbatim.
__version__ = "0.1.0"

# One process-wide urllib seam covers product modules plus the bundled engine client. The wrapper is a
# no-op policy-wise unless local-only is enabled, but always appends privacy-safe attempt metadata.
from .network_policy import install_urllib_guard as _install_urllib_guard

_install_urllib_guard()
del _install_urllib_guard
