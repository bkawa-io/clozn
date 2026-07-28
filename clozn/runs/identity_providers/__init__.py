"""Identity facet providers, discovered by clozn.runs.identity_ext.

Drop one module per facet here. Each must define a `NAME` string (the namespace its fields land under,
at `identity["ext"][NAME]`) and an `identity(context)` function returning a dict -- or `{}`/`None` when
the facet cannot be honestly measured, which omits the namespace entirely rather than null-padding it.

See clozn/runs/identity_ext.py for the full contract, including why a provider must be cheap (it runs on
the path that records every real run) and why it must never raise.
"""
