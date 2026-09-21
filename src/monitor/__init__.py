"""SentinelLoop Monitor Package — Real-time behavioral drift detection and multi-dimensional analysis."""

from src.monitor.classifier import (
    BehavioralVector,
    evaluate_conversation_history,
    compute_drift_velocity,
)

__all__ = [
    "BehavioralVector",
    "evaluate_conversation_history",
    "compute_drift_velocity",
]
