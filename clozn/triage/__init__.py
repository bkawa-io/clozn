"""Automatic regression triage (roadmap feature 05): an evidence ladder, not a generated explanation.

This package assembles a ``clozn.triage.v1`` artifact (see ``clozn/schemas/defs/clozn.triage.v1.json``)
from a baseline/candidate run pair. Every conclusion is traceable to a stored step; the summary is a pure
recomputation over those steps, never an independent narrative (roadmap rule 1: evidence before
narration).

    clozn.triage.status    the controlled evidence-status enum and the matched/mismatched ->
                            eliminated/observed derivation
    clozn.triage.steps     model-free comparison steps: identity diff (step 1) and context/rendered-
                            prompt diff (step 2)
    clozn.triage.rules     the rule engine: classify(steps) -> summary
    clozn.triage.artifact  assembles a complete, schema-validated clozn.triage.v1 document, including the
                            explicit not_run placeholders for every step this build does not implement
                            (controlled replay, quant/export, tool contract, internal localization)

Model-free by construction: nothing in this package boots an engine, runs a model, or makes a network
call. The GPU-touching steps (controlled replay and beyond) are deliberately out of scope for this slice
-- see notes/agent_roadmap/05-automatic-regression-triage.md and the plan this was built against.
"""
from __future__ import annotations
