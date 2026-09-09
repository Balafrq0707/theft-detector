from dataclasses import dataclass


@dataclass
class TheftDetectorConfig:
    interaction_distance: float = 100.0
    parcel_missing_timeout: float = 5.0

    observation_timeout: float = 10.0

    minimum_interaction_duration: float = 1.0

    suspicious_risk_threshold: float = 0.6
    theft_risk_threshold: float = 0.85

    event_cooldown: float = 10.0