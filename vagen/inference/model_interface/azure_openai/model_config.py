from dataclasses import dataclass
from typing import Optional, Dict, Any

from vagen.inference.model_interface.base_model_config import BaseModelConfig


@dataclass
class AzureOpenAIModelConfig(BaseModelConfig):
    """Configuration for Azure OpenAI API model interface."""

    # Azure OpenAI specific parameters
    api_key: Optional[str] = None  # If None, will use env AZURE_OPENAI_API_KEY
    azure_endpoint: Optional[str] = None  # e.g., https://<resource>.openai.azure.com/
    api_version: str = "2024-12-01-preview"
    deployment_name: Optional[str] = None  # Preferred; falls back to model_name

    # Retry/timeout knobs
    max_retries: int = 3
    timeout: int = 60

    # For o-series models on Azure we use max_completion_tokens
    max_completion_tokens: Optional[int] = None

    # Provider identifier
    provider: str = "azure_openai"

    def config_id(self) -> str:
        deployment = self.deployment_name or self.model_name
        return f"AzureOpenAIModelConfig({deployment},max_tokens={self.max_tokens},temp={self.temperature})"

    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        return {
            "description": "Azure-hosted OpenAI API",
            "supports_multimodal": True,
            # Keep permissive; actual availability depends on the user's Azure deployments
            "supported_models": [
                "gpt-5",
                "o3",
            ],
            "default_model": "gpt-5",
        }


