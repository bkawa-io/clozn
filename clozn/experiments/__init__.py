"""Shared execution/evidence kernel for the two v1 analyses.

Import concrete contracts from their owning modules.  Keeping this package
initializer empty prevents a normal product import from loading optional or
retired experiment products.
"""

__all__: tuple[str, ...] = ()
