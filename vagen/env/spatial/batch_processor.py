import os
import json
import time
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import requests

from google import genai  # type: ignore
from google.genai import types  # type: ignore

from vagen.inference.model_interface.openai.model import OpenAIModelInterface
from vagen.inference.model_interface.openai.model_config import OpenAIModelConfig
from vagen.inference.model_interface.claude.model import ClaudeModelInterface
from vagen.inference.model_interface.claude.model_config import ClaudeModelConfig

class BaseBatchProcessor(ABC):
    """Base class for batch processing."""
    
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model_name = model_config.get("model_name", "")

    @abstractmethod
    def submit(self, messages_list: List[List[Dict[str, Any]]], metas: List[Dict[str, Any]]) -> List[str]:
        """Submit batch job and return list of batch_ids."""
        pass

    @abstractmethod
    def retrieve(self, batch_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve batch results."""
        pass
    
    def _save_jsonl(self, data: List[Dict], path: str):
        """Helper to save list of dicts to JSONL."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

class OpenAIBatchProcessor(BaseBatchProcessor):
    """Batch processor for OpenAI and compatible models (e.g. Gemini).

    Notes for Gemini (OpenAI compatibility layer):
    - Batch creation/status uses the OpenAI SDK pointed at Gemini's OpenAI-compatible base_url.
    - File upload/download for batch input/output must use the Google `genai` SDK.
      See: https://ai.google.dev/gemini-api/docs/openai
    """
    
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.cfg = OpenAIModelConfig(**model_config)
        self.interface = OpenAIModelInterface(self.cfg)
        self.client = self.interface.client
        self.is_gemini = (
            (self.cfg.organization or "").lower() == "google"
            or ("generativelanguage.googleapis.com" in (self.cfg.base_url or ""))
        )

        # Gemini OpenAI compatibility does not support OpenAI SDK file upload/download.
        # Use the Google GenAI SDK for those operations.
        self._genai_client = None
        self._genai_types = None
        if self.is_gemini:

            api_key = self.cfg.api_key or os.getenv("GOOGLE_API_KEY")
            self._genai_client = genai.Client(api_key=api_key) 
            self._genai_types = types

    def submit(self, messages_list, metas) -> List[str]:
        # OpenAI batch file size limit: 200MB
        MAX_FILE_SIZE_MB = 190  # Use 190MB to leave some buffer
        MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
        
        # Generate base path for batch files
        import tempfile
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        base_name = f"batch_requests_{timestamp}"
        
        lines = []
        for i, msgs in enumerate(messages_list):
            mid = (metas[i] or {}).get("message_id", f"req_{i}")
            body = self.interface._prepare_api_payload(msgs)
            lines.append({
                "custom_id": str(mid),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })
        
        # Split into batches based on file size
        batch_ids = []
        current_batch = []
        current_size = 0
        batch_num = 0
        
        for line in lines:
            line_str = json.dumps(line, ensure_ascii=False) + "\n"
            line_size = len(line_str.encode('utf-8'))
            
            if current_size + line_size > MAX_FILE_SIZE_BYTES and current_batch:
                batch_num += 1
                batch_path = os.path.join(temp_dir, f"{base_name}_part{batch_num}.jsonl")
                self._save_jsonl(current_batch, batch_path)
                batch_id = self._submit_single_file(batch_path)
                batch_ids.append(batch_id)
                print(f"Submitted batch {batch_num} with {len(current_batch)} requests, batch_id: {batch_id}", flush=True)
                current_batch = []
                current_size = 0
            
            current_batch.append(line)
            current_size += line_size
        
        if current_batch:
            batch_num += 1
            suffix = "" if batch_num == 1 else f"_part{batch_num}"
            batch_path = os.path.join(temp_dir, f"{base_name}{suffix}.jsonl")
            self._save_jsonl(current_batch, batch_path)
            batch_id = self._submit_single_file(batch_path)
            batch_ids.append(batch_id)
            print(f"Submitted batch {batch_num} with {len(current_batch)} requests, batch_id: {batch_id}", flush=True)
        
        return batch_ids
    
    def _submit_single_file(self, jsonl_path: str) -> str:
        """Submit a single batch file."""

        # Upload batch input file
        if self.is_gemini:
            assert self._genai_client is not None and self._genai_types is not None
            display_name = os.path.splitext(os.path.basename(jsonl_path))[0] or "batch_requests"
            uploaded_file = self._genai_client.files.upload(
                file=jsonl_path,
                config=self._genai_types.UploadFileConfig(
                    display_name=display_name,
                    mime_type="application/jsonl",
                ),
            )
            input_file_id = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "id", None)
            if not input_file_id:
                raise RuntimeError(f"Gemini file upload returned no file id: {uploaded_file!r}")
        else:
            with open(jsonl_path, "rb") as f:
                batch_input = self.client.files.create(file=f, purpose="batch")
            input_file_id = batch_input.id

        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    def retrieve(self, batch_ids: List[str]) -> List[Dict[str, Any]]:
        if isinstance(batch_ids, str):
            batch_ids = [batch_ids]
        
        # Wait for all batches to complete first
        for batch_id in batch_ids:
            self._wait_for_batch_completion(batch_id)
        
        # Then retrieve all results
        all_results = []
        for batch_id in batch_ids:
            print(f"Retrieving results for batch {batch_id}...", flush=True)
            all_results.extend(self._retrieve_batch_results(batch_id))
        return all_results
    
    def _wait_for_batch_completion(self, batch_id: str) -> None:
        """Wait for a single batch to complete."""
        total_wait_time = 0  # minutes
        loop_wait_time = 300  # seconds
        while True:
            b = self.client.batches.retrieve(batch_id)
            if b.status in ("failed", "canceled", "cancelled"):
                raise RuntimeError(f"Batch {batch_id} status={b.status} reason={b.errors}")
            if b.status in ("completed", "expired"):
                print(f"Batch {batch_id} status: {b.status}", flush=True)
                break
            print(f"Batch {batch_id}: Waiting another {loop_wait_time} seconds, Total wait time: {total_wait_time:.2f} minutes...", flush=True)
            time.sleep(loop_wait_time) 
            total_wait_time += loop_wait_time / 60.0
    
    def _retrieve_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Retrieve results from a completed batch."""
        b = self.client.batches.retrieve(batch_id)
        error_file_id = getattr(b, "error_file_id", None)
        if not b.output_file_id and not error_file_id:
            return []

        def _file_text(file_id: str) -> str:
            if self.is_gemini:
                assert self._genai_client is not None
                data = self._genai_client.files.download(file=file_id)
                if isinstance(data, bytes):
                    return data.decode("utf-8")
                if hasattr(data, "decode"):
                    return data.decode("utf-8")
                if isinstance(data, str):
                    return data
                if hasattr(data, "read"):
                    return data.read().decode("utf-8")
                return str(data)

            content = self.client.files.content(file_id)
            text = getattr(content, "text", None) or getattr(content, "content", None)
            if hasattr(text, "decode"):
                text = text.decode("utf-8")
            if not isinstance(text, str):
                text = content.read().decode("utf-8")
            return text

        results: List[Dict[str, Any]] = []
        if b.output_file_id:
            for line in _file_text(b.output_file_id).splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                body = ((obj.get("response") or {}).get("body") or {})
                choices = body.get("choices") or []
                llm_text = choices[0].get("message", {}).get("content", "") if choices else ""
                results.append({"message_id": obj.get("custom_id"), "text": llm_text, "usage": body.get("usage") or {}})

        if error_file_id:
            for line in _file_text(error_file_id).splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                print("Batch error:", obj.get("custom_id"), obj.get("error") or obj, flush=True)

        return results

class ClaudeBatchProcessor(BaseBatchProcessor):
    """Batch processor for Claude models."""
    
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.cfg = ClaudeModelConfig(**model_config)
        self.interface = ClaudeModelInterface(self.cfg)
        self.client = self.interface.client

    def submit(self, messages_list, metas) -> List[str]:
        # Claude batch file size limit: 256MB
        MAX_FILE_SIZE_MB = 250  # Use 250MB to leave some buffer
        MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
        
        # Generate base path for batch files
        import tempfile
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        base_name = f"batch_requests_{timestamp}"
        
        lines = []
        for i, msgs in enumerate(messages_list):
            mid = (metas[i] or {}).get("message_id", f"req_{i}")
            params = self.interface._prepare_api_payload(msgs)
            lines.append({
                "custom_id": str(mid),
                "params": params
            })
        
        # Split into batches based on file size
        batch_ids = []
        current_batch = []
        current_size = 0
        batch_num = 0
        
        for line in lines:
            line_str = json.dumps(line, ensure_ascii=False) + "\n"
            line_size = len(line_str.encode('utf-8'))
            
            if current_size + line_size > MAX_FILE_SIZE_BYTES and current_batch:
                batch_num += 1
                batch_path = os.path.join(temp_dir, f"{base_name}_part{batch_num}.jsonl")
                self._save_jsonl(current_batch, batch_path)
                batch_id = self._submit_single_file(current_batch)
                batch_ids.append(batch_id)
                print(f"Submitted batch {batch_num} with {len(current_batch)} requests, batch_id: {batch_id}", flush=True)
                current_batch = []
                current_size = 0
            
            current_batch.append(line)
            current_size += line_size
        
        if current_batch:
            batch_num += 1
            suffix = "" if batch_num == 1 else f"_part{batch_num}"
            batch_path = os.path.join(temp_dir, f"{base_name}{suffix}.jsonl")
            self._save_jsonl(current_batch, batch_path)
            batch_id = self._submit_single_file(current_batch)
            batch_ids.append(batch_id)
            print(f"Submitted batch {batch_num} with {len(current_batch)} requests, batch_id: {batch_id}", flush=True)
        
        return batch_ids
    
    def _submit_single_file(self, requests_data: List[Dict]) -> str:
        """Submit a single batch file."""
        batch = self.client.messages.batches.create(requests=requests_data)
        return batch.id

    def retrieve(self, batch_ids: List[str]) -> List[Dict[str, Any]]:
        if isinstance(batch_ids, str):
            batch_ids = [batch_ids]
        
        # Wait for all batches to complete first
        for batch_id in batch_ids:
            self._wait_for_batch_completion(batch_id)
        
        # Then retrieve all results
        all_results = []
        for batch_id in batch_ids:
            all_results.extend(self._retrieve_batch_results(batch_id))
        return all_results
    
    def _wait_for_batch_completion(self, batch_id: str) -> None:
        """Wait for a single batch to complete."""
        total_wait_time = 0  # minutes
        loop_wait_time = 60  # seconds
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                print(f"Batch {batch_id} completed", flush=True)
                break
            print(f"Batch {batch_id}: Waiting another {loop_wait_time} seconds, Total wait time: {total_wait_time:.2f} minutes...", flush=True)
            time.sleep(loop_wait_time)
            total_wait_time += loop_wait_time / 60.0
    
    def _retrieve_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Retrieve results from a completed batch."""
        b = self.client.messages.batches.retrieve(batch_id)
        
        headers = {"x-api-key": self.client.api_key, "anthropic-version": "2023-06-01"}
        results_url = getattr(b, "results_url", None)
        objs = None
        if not results_url:
            base_url = self.cfg.base_url or "https://api.anthropic.com"
            status = requests.get(f"{base_url}/v1/messages/batches/{batch_id}", headers=headers)
            status.raise_for_status()
            status_json = status.json()
            results_url = status_json.get("results_url")
            if not results_url:
                objs = status_json.get("results", [])
        if results_url:
            resp = requests.get(results_url, headers=headers)
            resp.raise_for_status()
            objs = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
        if not objs:
            return []
            
        results = []
        for obj in objs:
            custom_id = obj.get("custom_id")
            res = obj.get("result", {})
            if res.get("type") == "succeeded":
                msg = res.get("message", {})
                content = msg.get("content", [])
                text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
                usage = msg.get("usage", {})
                results.append({"message_id": custom_id, "text": text, "usage": usage})
            else:
                print("Batch error:", custom_id, res.get("error") or res, flush=True)
        return results

def get_batch_processor(model_config: dict) -> BaseBatchProcessor:
    """Factory to get appropriate batch processor."""
    if "anthropic" in model_config.get("provider", "").lower():
        return ClaudeBatchProcessor(model_config)
    elif "openai" in model_config.get("provider", "").lower():
        return OpenAIBatchProcessor(model_config)
    else:
        raise ValueError(f"Unsupported model: {model_config.get('model_name', '')}")
