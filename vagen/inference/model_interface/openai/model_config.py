# vagen/mllm_agent/model_interface/openai/model_config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from vagen.inference.model_interface.base_model_config import BaseModelConfig

@dataclass
class OpenAIModelConfig(BaseModelConfig):
    """Configuration for OpenAI API model interface."""
    
    # OpenAI specific parameters
    api_key: Optional[str] = None  # If None, will use environment variable
    organization: Optional[str] = None # like google, intern, ... all use openai sdk
    base_url: Optional[str] = None  # For custom endpoints
    
    # Model parameters
    model_name: str = "gpt-4o"
    max_retries: int = 3
    max_retries_api: int = 2 
    timeout: int = 60
    
    # Generation parameters (inherited from base)
    # max_tokens, temperature already defined in base
    max_completion_tokens: Optional[int] = None # for o-series models
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    reasoning_effort: Optional[str] = None
    
    # Provider identifier
    provider: str = "openai"
    
    def config_id(self) -> str:
        """Generate unique identifier for this configuration."""
        return f"OpenAIModelConfig({self.model_name},max_tokens={self.max_tokens},temp={self.temperature})"
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about the OpenAI provider."""
        return {
            "description": "OpenAI API for GPT models",
            "supports_multimodal": True,
            "supported_models": [
                "gpt-5.1",
                "gpt-5",
                "gpt-5-mini",
                "o4-mini",
                "o3-mini",
                "gpt-4.1-nano",
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo", 
                "gpt-4-vision-preview",
                "gpt-3.5-turbo",
                "internvl3.5-241b-a28b",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-3-pro-preview",
                "Qwen/Qwen2.5-VL-72B-Instruct",
                "GLM-4.5V"

            ],
            "default_model": "gpt-4o"
        }