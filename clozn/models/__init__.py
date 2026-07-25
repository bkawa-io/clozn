"""clozn.models -- local-GGUF discovery shared by the CLI and the server.

See clozn.models.inventory for the actual scan/quant/inventory logic and why it lives in its own
top-level package rather than under clozn.cli (the server must never import clozn.cli -- see
clozn/server/app.py's route-registration comment; this package is the extraction point that lets both
sides call the exact same scan instead of the server reimplementing it).
"""
