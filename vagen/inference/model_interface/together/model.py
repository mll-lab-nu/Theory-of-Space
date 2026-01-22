import base64
import io
import logging
import os
from typing import Any, Dict, List

from PIL import Image
from together import Together

from vagen.inference.model_interface.base_model import BaseModelInterface
from .model_config import TogetherModelConfig
from vagen.utils.parallel_retry import run_parallel_with_retries, NonRetryableError

logger = logging.getLogger(__name__)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Small helper for dict-or-object SDK responses."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


class TogetherModelInterface(BaseModelInterface):
    """Together AI model interface (Together Python SDK), Qwen-style prompt compatible."""
    
    def __init__(self, config: TogetherModelConfig):
        super().__init__(config)
        self.config = config
        
        # Get API key from config or environment
        self.api_key = config.api_key or os.environ.get("TOGETHER_API_KEY")
        
        # Check if API key is available
        if not self.api_key:
            error_msg = "Together API key not set. Set TOGETHER_API_KEY or provide api_key in config."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Initialize Together client (docs: https://docs.together.ai/docs/inference-python)
        self.client = Together(
            api_key=self.api_key,
            timeout=self.config.timeout,
        )
        
        logger.info(f"Initialized Together AI interface with model {config.model_name}")
    
    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using Together AI API with parallel retries and stable ordering."""
        def worker(prompt: List[Dict]) -> Dict[str, Any]:
            return self._single_api_call(prompt, **kwargs)

        return run_parallel_with_retries(
            list(prompts),
            worker,
            max_workers=self.config.max_workers,
            max_attempt_rounds=self.config.max_retries,
        )
    
    def _prepare_together_request(self, prompt: List[Dict], **kwargs) -> Dict:
        """
        Convert Qwen format messages to Together AI request format.
        Together API supports OpenAI-compatible format.
        """
        messages = self._convert_qwen_to_together_format(prompt)
        
        return {
            "model": f"Qwen/{self.config.model_name}" if "Qwen" in self.config.model_name else self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
    
    def _convert_qwen_to_together_format(self, prompt: List[Dict]) -> List[Dict]:
        """
        Convert Qwen format messages to Together AI format.
        
        Qwen format: Text with <image> placeholders + separate multi_modal_data
        Together format: For multimodal models, uses OpenAI-compatible format
        """
        together_messages = []
        
        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Create Together AI message structure (OpenAI compatible)
            together_msg: Dict[str, Any] = {"role": role}
            
            # Handle multimodal content
            if ("multi_modal_data" in message or "images" in message) and "<image>" in content:
                images = []
                if "images" in message:
                    images.extend(message["images"])
                if "multi_modal_data" in message:
                    for key, values in message["multi_modal_data"].items():
                        if key == "<image>" or "image" in key.lower():
                            images.extend(values)
                
                # Split content by <image> placeholders
                parts = content.split("<image>")
                
                # Build content array alternating text and images
                content_array = []
                for i, part in enumerate(parts):
                    # Add text part if not empty
                    if part.strip():
                        content_array.append({
                            "type": "text",
                            "text": part
                        })
                    
                    # Add image if available (except for last part)
                    if i < len(parts) - 1 and i < len(images):
                        image_data = self._process_image_for_together(images[i])
                        content_array.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        })
                
                together_msg["content"] = content_array
            else:
                # Text-only message
                together_msg["content"] = content
            
            together_messages.append(together_msg)
        
        return together_messages
    
    def _process_image_for_together(self, image: Any) -> str:
        """Convert image to base64 (data URL payload)."""
        if isinstance(image, Image.Image):
            # Ensure RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize if too large to save tokens
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode()

        if isinstance(image, str):
            with Image.open(image) as img:
                return self._process_image_for_together(img)

        elif isinstance(image, dict) and "__pil_image__" in image:
            from vagen.server.serial import deserialize_pil_image
            pil_image = deserialize_pil_image(image)
            return self._process_image_for_together(pil_image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _single_api_call(self, prompt: List[Dict], **kwargs) -> Dict[str, Any]:
        """Make a single Together SDK call."""
        try:
            request_data = self._prepare_together_request(prompt, **kwargs)

            response = self.client.chat.completions.create(**request_data)

            choices = _get(response, "choices", [])
            first = choices[0] if choices else {}
            msg = _get(first, "message", {})
            text = _get(msg, "content", "")

            usage = _get(response, "usage", None)
            prompt_tokens = _get(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = _get(usage, "completion_tokens", 0) if usage else 0
            total_tokens = _get(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else (prompt_tokens + completion_tokens)

            return {
                "text": text,
                "usage": {
                    "prompt_tokens": prompt_tokens or 0,
                    "completion_tokens": completion_tokens or 0,
                    "total_tokens": total_tokens or 0,
                },
                "finish_reason": _get(first, "finish_reason", "unknown"),
            }
        except Exception as e:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None) or getattr(e, "status_code", None)
            msg = str(e)
            if status in (401, 403) or "401" in msg or "unauthorized" in msg.lower() or "api key" in msg.lower():
                raise NonRetryableError(f"Together auth error: {msg}") from e
            logger.error(f"Together API error: {e}")
            raise
    
    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format conversation messages into a prompt string.
        
        For Together AI, returns a string representation of the messages
        for logging/debugging purposes.
        """
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle different roles
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model."""
        info = super().get_model_info()
        
        # Determine multimodal support based on model name
        # This is a simplification - actual determination may require more logic
        multimodal_models = [
            "llava", "qwen-vl", "cogvlm", "bakllava", "internlm-xcomposer"
        ]
        is_multimodal = any(model_type in self.config.model_name.lower() for model_type in multimodal_models)
        
        info.update({
            "name": self.config.model_name,
            "type": "multimodal" if is_multimodal else "text",
            "supports_images": is_multimodal,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "config_id": self.config.config_id()
        })
        
        return info