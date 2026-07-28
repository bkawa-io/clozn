"""clozn.adopt -- discovering and reusing models from an existing local Ollama install.

This is the "adopt" direction, not the "compatibility endpoint" direction: clozn/server/routes/ollama.py
makes Clozn *answer* Ollama's wire protocol so Ollama-speaking clients can point at Clozn. This package
does the opposite -- it makes Clozn a (read-only) *client* of a real Ollama daemon/CLI/disk store, so a
user who already has Ollama can try Clozn on a model they already downloaded without redownloading or
deleting anything.

Hard rule, load-bearing across every module here: nothing in this package ever writes, deletes, renames,
or re-tags anything under Ollama's own storage or reachable through Ollama's own mutating API endpoints
(`/api/pull`, `/api/delete`, `/api/create`, `/api/copy`). Only read calls (`/api/version`, `/api/tags`,
`/api/show`) and read-only filesystem access are permitted from this package. New entries this package
creates live only under Clozn's own model directory (see clozn.models.inventory).
"""
