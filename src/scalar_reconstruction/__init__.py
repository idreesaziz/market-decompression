"""Scalar transfer-field reconstruction package.

This add-on replaces pointwise simulator search with noisy scalar-field estimation:

    Y(theta, R, b, r) = F_b(theta; R) + eps(theta, R, b, r)

The package is intentionally field-observation centric.  It reuses the existing
`mde_search` simulator/model/training stack when executing expensive transfer
jobs, but keeps design, replication, noise, surrogate, and envelope logic here.
"""

__version__ = "0.1.0"
