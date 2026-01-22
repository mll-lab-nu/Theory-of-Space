from dataclasses import dataclass
from vagen.env.base.base_service_config import BaseServiceConfig

@dataclass
class SpatialGymServiceConfig(BaseServiceConfig):
    """Configuration for SpatialGym service."""
    use_state_reward: bool = False