"""Public research interface for the Explicit Layout Transformer route.

The trainable architecture is deliberately decoupled from the canonical
candidate contract. Architecture variants should emit hypotheses through this
module so representation, replay, and evaluation remain comparable.
"""

from .candidate_interfaces import (
    EXPLICIT_POLICY,
    build_explicit_topology_candidate,
    replay_explicit_to_raw,
    select_explicit_candidate,
)

__all__ = [
    "EXPLICIT_POLICY",
    "build_explicit_topology_candidate",
    "replay_explicit_to_raw",
    "select_explicit_candidate",
]
