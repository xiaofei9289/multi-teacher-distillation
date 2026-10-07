"""Prospectively frozen VG-MTD V5 components."""

from .vgmt_v5_contract import (
    CLAIM_LABELS,
    QUERY_LABELS,
    hierarchical_claim_distribution,
    hierarchical_targets_from_mass,
)

__all__ = [
    "CLAIM_LABELS",
    "QUERY_LABELS",
    "hierarchical_claim_distribution",
    "hierarchical_targets_from_mass",
]
