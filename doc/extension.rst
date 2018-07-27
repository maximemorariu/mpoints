============
C extension
============

The crucial algorithms of this package (simulation, likelihood and gradient computations, computation of residuals)
are in fact implemented in C via Cython. The class :py:class:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp` wraps a C
extension that is compiled when the package is installed with `pip`.

Without the increase in computational speed provided by the C extension, the statistical inference of
state-dependent Hawkes processes would be impractical.

The Cython code that generated the C extension is available in the GitHub_ repository (the .pyx file).

.. _GitHub: https://github.com