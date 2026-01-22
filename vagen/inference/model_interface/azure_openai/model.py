import base64
import logging
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from openai import AzureOpenAI
from PIL import Image
import io

from vagen.inference.model_interface.base_model import BaseModelInterface
from .model_config import AzureOpenAIModelConfig
from vagen.utils.parallel_retry import run_parallel_with_retries


logger = logging.getLogger(__name__)


class AzureOpenAIModelInterface(BaseModelInterface):
    """Model interface for Azure OpenAI API with Qwen format compatibility."""

    def __init__(self, config: AzureOpenAIModelConfig):
        super().__init__(config)
        self.config = config

        # Resolve Azure credentials and endpoint
        api_key = config.api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        azure_endpoint = config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = config.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        if not azure_endpoint:
            raise ValueError("azure_endpoint is required for AzureOpenAIModelInterface")

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

        # Thread pool for batch processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        logger.info(
            f"Initialized Azure OpenAI interface with deployment {config.deployment_name or config.model_name}"
        )

    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using Azure OpenAI API with parallel retries and stable ordering."""
        formatted_requests: List[List[Dict[str, Any]]] = []
        for prompt in prompts:
            messages = self._convert_qwen_to_openai_format(prompt)
            formatted_requests.append(messages)

        def worker(messages: List[Dict]) -> Dict[str, Any]:
            return self._single_api_call(messages, **kwargs)

        return run_parallel_with_retries(
            formatted_requests,
            worker,
            max_workers=self.config.max_workers,
            max_attempt_rounds=self.config.max_retries,
        )

    def _convert_qwen_to_openai_format(self, prompt: List[Dict]) -> List[Dict]:
        """
        Convert Qwen format messages to OpenAI format (text + image_url parts).
        """
        openai_messages: List[Dict[str, Any]] = []
        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")
            openai_msg: Dict[str, Any] = {"role": role, "content": []}

            if "multi_modal_data" in message and "<image>" in content:
                images: List[Any] = []
                for key, values in message["multi_modal_data"].items():
                    if key == "<image>" or "image" in key.lower():
                        images.extend(values)

                parts = content.split("<image>")
                for i, part in enumerate(parts):
                    if part.strip():
                        openai_msg["content"].append({"type": "text", "text": part})
                    if i < len(parts) - 1 and i < len(images):
                        image_data = self._process_image_for_openai(images[i])
                        openai_msg["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            }
                        )
            else:
                openai_msg["content"].append({"type": "text", "text": content})

            openai_messages.append(openai_msg)

        return openai_messages

    def _process_image_for_openai(self, image: Any) -> str:
        """Convert image to base64 for Azure OpenAI API."""
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode()
        elif isinstance(image, dict) and "__pil_image__" in image:
            from vagen.server.serial import deserialize_pil_image

            pil_image = deserialize_pil_image(image)
            return self._process_image_for_openai(pil_image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _single_api_call(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Make a single API call to Azure OpenAI."""
        try:
            deployment = self.config.deployment_name or self.config.model_name
            msg_kwargs: Dict[str, Any] = {
                "model": deployment,  # On Azure, this is the deployment name
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "timeout": kwargs.get("timeout", self.config.timeout),
            }
            # o-series in Azure uses max_completion_tokens
            if deployment.startswith("o") or "gpt-5" in deployment:
                msg_kwargs["max_completion_tokens"] = kwargs.get(
                    "max_completion_tokens", self.config.max_completion_tokens
                )
            else:
                msg_kwargs["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)

            response = self.client.chat.completions.create(**msg_kwargs)

            response_text = response.choices[0].message.content
            return {
                "text": response_text,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
            }
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise

    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        return "\n".join(formatted)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "name": self.config.deployment_name or self.config.model_name,
                "type": "multimodal"
                if "vision" in (self.config.model_name or "").lower() or "4o" in (self.config.model_name or "")
                else "text",
                "supports_images": "vision" in (self.config.model_name or "").lower() or "4o" in (self.config.model_name or ""),
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "config_id": self.config.config_id(),
            }
        )
        return info


