"""bempp-cl backend: secondary validation solver (dense Numba assembly) used for
cross-checking NumCalc at low-to-mid frequencies via an independent Galerkin BEM.

Install the optional dependency group to use this backend:
    uv sync --group bempp

Then instantiate directly:
    from beamsim2.backends.bempp.adapter import BemppBackend
    backend = BemppBackend()
"""

from beamsim2.backends.bempp.adapter import BemppBackend

__all__ = ["BemppBackend"]
