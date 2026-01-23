# vagen/mllm_agent/model_interface/openai/model.py
import base64
import logging
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from PIL import Image
import io

from vagen.inference.model_interface.base_model import BaseModelInterface
from .model_config import OpenAIModelConfig
from vagen.utils.parallel_retry import run_parallel_with_retries

logger = logging.getLogger(__name__)

class OpenAIModelInterface(BaseModelInterface):
    """Model interface for OpenAI API with Qwen format compatibility."""
    
    def __init__(self, config: OpenAIModelConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize OpenAI client
        org = (config.organization or "").lower()
        if org == "intern":
            api_key = config.api_key or os.getenv("INTERN_API_KEY")
        elif org == "google":
            api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        elif org == "self-hosted":
            api_key = config.api_key
            if not api_key and config.base_url:
                api_key = "EMPTY"
        elif org == "openrouter":
            api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        else:
            api_key = config.api_key or os.getenv("OPENAI_API_KEY")

        if not api_key and config.base_url:
            api_key = "EMPTY"

        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            max_retries=config.max_retries_api if hasattr(config, 'max_retries_api') else 2,
            timeout=config.timeout,
        )
        
        # Thread pool for batch processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        logger.info(f"Initialized OpenAI interface with model {config.model_name}")
    
    def _prepare_api_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Prepare API payload from Qwen format messages."""
        openai_messages = self._convert_qwen_to_openai_format(messages)
        return self._prepare_request_kwargs(openai_messages, **kwargs)

    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using OpenAI API with parallel retries and stable ordering.
        All calls must succeed; otherwise an error is raised."""
        formatted_requests = prompts  # Pass raw Qwen prompts to worker

        def worker(messages: List[Dict]) -> Dict[str, Any]:
            return self._single_api_call(messages, **kwargs)

        return run_parallel_with_retries(
            formatted_requests,
            worker,
            max_workers=self.config.max_workers,
            max_attempt_rounds=self.config.max_retries,
        )
    @staticmethod
    def _convert_qwen_to_openai_format(prompt: List[Dict]) -> List[Dict]:
        """
        Convert Qwen format messages to OpenAI format.
        
        Qwen format: Text with <image> placeholders + separate multi_modal_data
        OpenAI format: Structured content array with text and image objects
        """
        openai_messages = []
        
        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Create OpenAI message structure
            openai_msg = {
                "role": role,
                "content": []
            }
            
            # Handle multimodal content
            if ("multi_modal_data" in message or "images" in message) and "<image>" in content:
                # Extract images from multi_modal_data
                images = []
                if "images" in message:
                    images.extend(message["images"])
                elif "multi_modal_data" in message:
                    for key, values in message["multi_modal_data"].items():
                        if key == "<image>" or "image" in key.lower():
                            images.extend(values)
                
                # Split content by <image> placeholders
                parts = content.split("<image>")
                
                # Build content array alternating text and images
                for i, part in enumerate(parts):
                    # Add text part if not empty
                    if part.strip():
                        openai_msg["content"].append({
                            "type": "text",
                            "text": part
                        })
                    
                    # Add image if available (except for last part)
                    if i < len(parts) - 1 and i < len(images):
                        image_data = OpenAIModelInterface._process_image_for_openai(images[i])
                        openai_msg["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        })
            else:
                # Text-only message
                openai_msg["content"].append({
                    "type": "text",
                    "text": content
                })
            
            openai_messages.append(openai_msg)
        
        return openai_messages
    
    @staticmethod
    def _process_image_for_openai(image: Any) -> str:
        """Convert image to base64 for OpenAI API."""
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
        elif isinstance(image, str):
            # Assume it's a file path
            with Image.open(image) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Resize if too large to save tokens
                max_size = 1024
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode()
        elif isinstance(image, dict) and "__pil_image__" in image:
            from vagen.server.serial import deserialize_pil_image
            pil_image = deserialize_pil_image(image)
            return OpenAIModelInterface._process_image_for_openai(pil_image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _prepare_request_kwargs(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Prepare arguments for OpenAI API call."""
        msg_kwargs = {
            "model": self.config.model_name if 'glm' not in self.config.model_name else 'z-ai/'+self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if self.config.max_completion_tokens is not None:
            msg_kwargs["max_completion_tokens"] = self.config.max_completion_tokens
        else:
            msg_kwargs["max_tokens"] = self.config.max_tokens
        if self.config.reasoning_effort:
            msg_kwargs['reasoning_effort'] = kwargs.get("reasoning_effort", self.config.reasoning_effort)
        return msg_kwargs

    def _single_api_call(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Make a single API call to OpenAI."""
        try:
            msg_kwargs = self._prepare_api_payload(messages, **kwargs)

            response = self.client.chat.completions.create(**msg_kwargs)
            response_text = response.choices[0].message.content

            return {
                "text": response_text,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format prompt for compatibility.
        
        Since OpenAI uses structured messages, this returns a string representation
        of the messages for logging/debugging purposes.
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle Qwen special tokens if present
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model."""
        info = super().get_model_info()
        
        model_name = self.config.model_name.lower()
        supports_images = any(key in model_name for key in ["vision", "vl", "4o"])

        info.update({
            "name": self.config.model_name,
            "type": "multimodal" if supports_images else "text",
            "supports_images": supports_images,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "config_id": self.config.config_id()
        })
        
        return info

    @staticmethod
    def _sanitize_messages_for_logging(messages: List[Dict]) -> List[Dict]:
        """Create a safe copy of messages for logging, truncating long base64 strings."""
        import copy
        sanitized = []
        for msg in messages:
            msg_copy = copy.deepcopy(msg)
            if "content" in msg_copy and isinstance(msg_copy["content"], list):
                for item in msg_copy["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image") and len(url) > 100:
                            item["image_url"]["url"] = url[:50] + "...[TRUNCATED]..." + url[-20:]
            sanitized.append(msg_copy)
        return sanitized