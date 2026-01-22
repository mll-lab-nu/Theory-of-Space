from .vllm import VLLMModelInterface, VLLMModelConfig
from .openai import OpenAIModelInterface, OpenAIModelConfig
from .azure_openai import AzureOpenAIModelInterface, AzureOpenAIModelConfig
from .claude import ClaudeModelInterface, ClaudeModelConfig
# from .routerapi import RouterAPIModelInterface, RouterAPIModelConfig
from .together import TogetherModelInterface, TogetherModelConfig

REGISTERED_MODEL = {
    "vllm": {
        "model_cls": VLLMModelInterface,
        "config_cls": VLLMModelConfig,
    },
    "openai": {
        "model_cls": OpenAIModelInterface,
        "config_cls": OpenAIModelConfig
    },
    "azure_openai": {
        "model_cls": AzureOpenAIModelInterface,
        "config_cls": AzureOpenAIModelConfig
    },
    "anthropic": {
        "model_cls": ClaudeModelInterface,
        "config_cls": ClaudeModelConfig
    },
    # "routerapi": {
    #     "model_cls": RouterAPIModelInterface,
    #     "config_cls": RouterAPIModelConfig
    # },
    "together": {
        "model_cls": TogetherModelInterface,
        "config_cls": TogetherModelConfig
    }
}