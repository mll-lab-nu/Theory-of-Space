from .openai import OpenAIModelInterface, OpenAIModelConfig
from .claude import ClaudeModelInterface, ClaudeModelConfig

REGISTERED_MODEL = {
    "openai": {
        "model_cls": OpenAIModelInterface,
        "config_cls": OpenAIModelConfig
    },
    "anthropic": {
        "model_cls": ClaudeModelInterface,
        "config_cls": ClaudeModelConfig
    },
}