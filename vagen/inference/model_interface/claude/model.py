import base64
import logging
import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from anthropic import Anthropic
from PIL import Image
import io
import requests as http_requests

from vagen.inference.model_interface.base_model import BaseModelInterface
from .model_config import ClaudeModelConfig
from vagen.utils.parallel_retry import run_parallel_with_retries

logger = logging.getLogger(__name__)

class ClaudeModelInterface(BaseModelInterface):
    """Model interface for Claude API with Qwen format compatibility."""
    
    def __init__(self, config: ClaudeModelConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize Claude client
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Claude API key must be provided in config or ANTHROPIC_API_KEY environment variable")
        
        self.client = Anthropic(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        
        self.api_key = api_key
        
        # Thread pool for standard API calls
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        logger.info(f"Initialized Claude interface with model {config.model_name}")
    
    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using Claude API."""
        # Check if batch processing is requested
        use_batch = kwargs.get('use_batch', self.config.use_batch_api)
        
        if use_batch:
            return self._generate_batch(prompts, **kwargs)
        else:
            return self._generate_standard(prompts, **kwargs)
    
    def _prepare_api_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Prepare API payload from Qwen format messages."""
        claude_msgs, system = self._convert_qwen_to_claude_format(messages)
        return self._prepare_request_kwargs(claude_msgs, system, **kwargs)

    def _generate_standard(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using standard Claude API (realtime) with retries and stable ordering."""
        formatted_requests = prompts  # Pass raw Qwen prompts to worker

        def worker(messages: List[Dict]) -> Dict[str, Any]:
            return self._single_api_call(messages, **kwargs)

        return run_parallel_with_retries(
            formatted_requests,
            worker,
            max_workers=self.config.max_workers,
            max_attempt_rounds=self.config.max_retries,
        )
    
    def _generate_batch(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using Claude Batch API."""
        # Convert prompts to batch request format
        batch_requests = []
        
        for i, prompt in enumerate(prompts):
            params = self._prepare_api_payload(prompt, **kwargs)
            
            batch_requests.append({
                "custom_id": f"request-{i}",
                "params": params
            })
        
        # Submit batch
        batch_response = self._submit_batch(batch_requests)
        batch_id = batch_response["id"]
        
        # Poll for completion
        max_wait_time = kwargs.get('max_wait_time', 3600)  # 1 hour default
        poll_interval = kwargs.get('poll_interval', 5)  # 5 seconds default
        
        results = self._poll_batch_completion(batch_id, max_wait_time, poll_interval)
        
        # Map results back to original order; if any failed, raise
        ordered_results = []
        result_map = {r["custom_id"]: r for r in results}
        
        for i in range(len(prompts)):
            custom_id = f"request-{i}"
            if custom_id in result_map:
                result = result_map[custom_id]
                if result["result"]["type"] == "succeeded":
                    message_result = result["result"]["message"]
                    ordered_results.append({
                        "text": message_result["content"][0]["text"],
                        "usage": {
                            "prompt_tokens": message_result["usage"]["input_tokens"],
                            "completion_tokens": message_result["usage"]["output_tokens"],
                            "total_tokens": message_result["usage"]["input_tokens"] + message_result["usage"]["output_tokens"]
                        },
                        "finish_reason": message_result["stop_reason"]
                    })
                else:
                    error_msg = result["result"].get("error", {}).get("message", "Unknown error")
                    raise RuntimeError(f"Batch item {i} failed: {error_msg}")
            else:
                raise RuntimeError(f"Batch result not found for request index {i}")
        
        return ordered_results
    
    def _submit_batch(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submit a batch request to Claude API."""
        url = f"{self.config.base_url or 'https://api.anthropic.com'}/v1/messages/batches"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "requests": requests
        }
        
        response = http_requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()
    
    def _poll_batch_completion(self, batch_id: str, max_wait_time: int, poll_interval: int) -> List[Dict[str, Any]]:
        """Poll for batch completion and retrieve results."""
        url = f"{self.config.base_url or 'https://api.anthropic.com'}/v1/messages/batches/{batch_id}"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = http_requests.get(url, headers=headers)
            response.raise_for_status()
            
            batch_status = response.json()
            
            if batch_status["processing_status"] == "ended":
                # Retrieve results
                results_url = batch_status["results_url"]
                if results_url:
                    results_response = http_requests.get(results_url, headers=headers)
                    results_response.raise_for_status()
                    
                    # Parse JSONL format (JSON Lines)
                    results = []
                    for line in results_response.text.strip().split('\n'):
                        if line:
                            results.append(json.loads(line))
                    
                    return results
                else:
                    # If no results_url, the results might be included in the response
                    return batch_status.get("results", [])
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Batch {batch_id} did not complete within {max_wait_time} seconds")
    
    def _convert_qwen_to_claude_format(self, prompt: List[Dict]) -> Tuple[List[Dict], str]:
        """
        Convert Qwen format messages to Claude format.
        
        Qwen format: Text with <image> placeholders + separate multi_modal_data
        Claude format: Structured content array with text and image objects, 
                      plus separate system message
        
        Returns:
            Tuple of (messages, system_prompt)
        """
        claude_messages = []
        system_prompt = ""
        
        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Extract system prompt (Claude handles it separately)
            if role == "system":
                system_prompt = content
                continue
            
            # Convert role names
            claude_role = "user" if role == "user" else "assistant"
            
            # Create Claude message structure
            claude_msg = {
                "role": claude_role,
                "content": []
            }
            
            # Handle multimodal content
            if ("multi_modal_data" in message or "images" in message) and "<image>" in content:
                # Extract images from multi_modal_data or images field
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
                        claude_msg["content"].append({
                            "type": "text",
                            "text": part
                        })
                    
                    # Add image if available (except for last part)
                    if i < len(parts) - 1 and i < len(images):
                        image_data = self._process_image_for_claude(images[i])
                        claude_msg["content"].append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        })
            else:
                # Text-only message
                claude_msg["content"] = content
            
            claude_messages.append(claude_msg)
        
        return claude_messages, system_prompt
    
    def _process_image_for_claude(self, image: Any) -> str:
        """Convert image to base64 for Claude API."""
        if isinstance(image, Image.Image):
            # Ensure RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize if too large to save tokens
            max_size = 1568  # Claude's recommended maximum dimension
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
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
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image, dict) and "__pil_image__" in image:
            from vagen.server.serial import deserialize_pil_image
            pil_image = deserialize_pil_image(image)
            return self._process_image_for_claude(pil_image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _prepare_request_kwargs(self, messages: List[Dict], system_prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare arguments for Claude API call."""
        # Get max_tokens value
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        
        # # Create a copy of messages to avoid modifying the original and append token limit instruction
        # messages_with_limit = []
        # for i, msg in enumerate(messages):
        #     msg_copy = msg.copy()
        #     # If this is the last user message, append token limit instruction
        #     if i == len(messages) - 1 and msg_copy.get("role") == "user":
        #         if isinstance(msg_copy["content"], str):
        #             msg_copy["content"] += f"\n\nYour response should be within {max_tokens} tokens."
        #         elif isinstance(msg_copy["content"], list):
        #             # For multimodal content, append to the last text item
        #             for j in range(len(msg_copy["content"]) - 1, -1, -1):
        #                 if msg_copy["content"][j].get("type") == "text":
        #                     msg_copy["content"][j]["text"] += f"\n\nYour response should be within {max_tokens} tokens."
        #                     break
        #             else:
        #                 # If no text content found, add a new text item
        #                 msg_copy["content"].append({
        #                     "type": "text",
        #                     "text": f"Your response should be within {max_tokens} tokens."
        #                 })
        #     messages_with_limit.append(msg_copy)
        messages_with_limit = messages
        
        # Prepare parameters
        params = {
            "model": self.config.model_name,
            "messages": messages_with_limit,
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "stop_sequences": kwargs.get("stop_sequences", self.config.stop_sequences),
        }
        
        # Add thinking parameter if enabled and budget_tokens is set
        budget_tokens = kwargs.get("budget_tokens", self.config.budget_tokens)
        if self.config.thinking and budget_tokens:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens
            }
        
        # Add system prompt if provided
        if system_prompt:
            params["system"] = system_prompt
        
        # Add metadata if configured
        if self.config.metadata:
            params["metadata"] = self.config.metadata

        return params

    def _single_api_call(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Make a single API call to Claude."""
        try:
            params = self._prepare_api_payload(messages, **kwargs)
            
            # Make the API call
            response = self.client.messages.create(**params)
            
            # Extract text response
            response_text = next(block.text for block in response.content if block.type == "text")
            
            return {
                "text": response_text,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                "finish_reason": response.stop_reason
            }
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise
    
    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format prompt for compatibility.
        
        Since Claude uses structured messages, this returns a string representation
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
        
        info.update({
            "name": self.config.model_name,
            "type": "multimodal" if any(vision_model in self.config.model_name.lower() 
                                      for vision_model in ["vision", "claude-3"]) else "text",
            "supports_images": "claude-3" in self.config.model_name.lower(),
            "supports_batch": True,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "config_id": self.config.config_id()
        })
        
        return info