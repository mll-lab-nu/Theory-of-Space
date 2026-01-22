REGISTERED_ENV = {}

from .spatial import SpatialGym, SpatialGymConfig, SpatialGymService, SpatialGymServiceConfig
REGISTERED_ENV["spatial"] = {
    "env_cls": SpatialGym,
    "config_cls": SpatialGymConfig,
    "service_cls": SpatialGymService,
    "service_config_cls": SpatialGymServiceConfig
}
