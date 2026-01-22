<h1 align="center">THEORY OF SPACE: Can foundation models construct spatial beliefs through active perception?</h1>
<p align="center" style="font-size: 16px;">
  Pingyue Zhang*, Zihan Huang*, Yue Wang*, Jieyu Zhang*,Letian Xue, Zihan Wang, Qineng Wang, Keshigeyan Chandrasegaran, Ruohan Zhang, Yejin Choi, Ranjay Krishna, Jiajun Wu, Li Fei-Fei, Manling Li
</p>
<p align="center" style="font-size: 12px;"><i>(* equal contribution)</i></p>

This repository contains the official implementation of our paper, THEORY OF SPACE: Can foundation models construct spatial beliefs through active perception?

To build agents with spatial intelligence, we argue for evaluating not merely passive reasoning, but the active, self-directed construction of spatial belief from partial observations. We introduce **Theory of Space (ToS)**, a conceptual counterpart to Theory of Mind (ToM). While ToM models hidden mental states of others, ToS models uncertain, currently unobserved structure of space.

### Prepare the environment
```bash
pip install -e .
```

Add api keys (optional)
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
- SELF_HOSTED_API_KEY
- TOGETHER_API_KEY

The script supports multiple experiment types (active/passive), render modes (vision/text), and flexible seed ranges.

### Download the datasets

```bash
hf download yw12356/tos_dataset_0117_3room_100runs --repo-type dataset --local-dir data
```

## Quick Start

To add a new model, edit `scripts/SpatialGym/base_model_config.yaml` and add an entry under the `models` section. Each model requires specific parameters based on its provider:

### OpenAI Models
```yaml
models:
  gpt-4.1-mini:
    provider: openai
    model_name: gpt-4.1-mini  # Actual model identifier
    max_tokens: 8192
    temperature: 0.0
    max_workers: 64
    presence_penalty: 0.0
    frequency_penalty: 0.0
    max_retries: 3
    timeout: 60
```

### Google Models (via OpenAI API)
```yaml
  gemini-2.5-pro:
    provider: openai
    organization: google
    model_name: gemini-2.5-pro
    max_tokens: 32768
    temperature: 0
    max_workers: 32
    base_url: https://generativelanguage.googleapis.com/v1beta/openai/
    timeout: 500
    max_retries: 5
```

### Self-Hosted Models
```yaml
  your-custom-model:
    provider: openai
    organization: self-hosted
    model_name: your-model-name
    max_tokens: 16384
    temperature: 0
    max_workers: 64
    base_url: https://your-api-endpoint.com/v1
    timeout: 500
```

### VLLM Models
```yaml
  Qwen3-VL-2B-Instruct:
    provider: vllm
    model_name: models/Qwen3-VL-2B-Instruct
    max_tokens: 8192
    temperature: 0.0
    max_workers: 16
    gpu_memory_utilization: 0.7
```
  
### Common Parameters
- `provider`: API provider (`openai`, `anthropic`, `together`, `azure_openai`)
- `model_name`: Model identifier used by the API
- `max_tokens` or `max_completion_tokens`: Maximum tokens for response
- `temperature`: Sampling temperature (0 = deterministic)
- `max_workers`: Maximum parallel API calls
- `timeout`: Request timeout in seconds
- `max_retries`: Number of retry attempts on failure
- `base_url`: Custom API endpoint (for non-default providers)
- `organization`: Provider organization (for routing)

After adding your model, reproduce the result
```bash
python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name your-model-name \
  --num 25
  --data-dir data/ \
  --output-root result/ \
  --render-mode vision,text \
  --exp-type active,passive \
  --cogmap \
  --false-belief-exp \
```

## More Options
```bash
# passive
python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --exp-type passive \
  --num 25 \
  --output-root result/ \
  --data-dir vagen/env/spatial/room_data_3_room/  \
  --inference-mode batch \
  --render-mode text,vision \
  --proxy-agent scout 2>&1 | tee logs/passive_gpt-5.2.log

# active
python scripts/SpatialGym/spatial_run.py \
  --phase explore \
  --model-name gpt-5.2 \
  --exp-type active \
  --num 25 \
  --output-root result/ \
  --data-dir vagen/env/spatial/room_data_3_room/  \
  --inference-mode batch \
  --render-mode text,vision \
  --proxy-agent scout 2>&1 | tee logs/active_gpt-5.2.log

# cogmap (after exploration)
python scripts/SpatialGym/spatial_run.py \
  --phase cogmap \
  --model-name gpt-5.2 \
  --exp-type active \
  --num 25 \
  --output-root result/ \
  --data-dir vagen/env/spatial/room_data_3_room/  \
  --inference-mode batch \
  --render-mode text,vision \
  --proxy-agent scout 2>&1 | tee logs/cogmap_gpt-5.2.log

# active exploration + evaluation + cogmap
python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --exp-type active \
  --num 25 \
  --output-root result/ \
  --data-dir vagen/env/spatial/room_data_3_room/  \
  --inference-mode batch \
  --render-mode text,vision \
  --cogmap \
  --proxy-agent scout 2>&1 | tee logs/active_gpt-5.2.log

# false-belief-exp (after exploration)
python scripts/SpatialGym/spatial_run.py \
  --phase explore \
  --model-name gpt-5.2 \
  --exp-type active \
  --num 25 \
  --data-dir vagen/env/spatial/room_data_3_room/  \
  --output-root result/ \
  --render-mode text,vision \
  --false-belief-exp \
  --proxy-agent scout 2>&1 | tee logs/fb-exp_gpt-5.2.log
```


## Command Line Arguments

### Phase Selection
- `--phase`: Which phase to run (default: `all`)
  - `explore`: Dataset creation and exploration inference
  - `eval`: Evaluation inference on exploration results
  - `cogmap`: Cognitive map generation and evaluation
  - `reeval`: Re-run evaluation on existing exploration data
  - `cogmap_reeval`: Re-run cognitive map evaluation on existing data
  - `aggregate`: Aggregate logs and images
  - `all`: Run exploration + evaluation + aggregation (+ cogmap if `--cogmap` is set)

### Core Options
- `--exp-type`: Experiment type (default: `active`)
  - Single value: `active` or `passive`
  - Multiple values: `active,passive` (comma-separated)
- `--model-name`: Model identifier (default: `gpt-4o-mini`)
- `--render-mode`: Environment render mode (default: `vision`)
  - Single value: `vision` or `text`
  - Multiple values: `vision,text` (comma-separated)
- `--num`: Number of samples per task (default: `1`)
- `--seed-range`: Seed range in format `start-end` (e.g., `0-24`)
  - If not specified, uses `0` to `num-1`
- `--data-dir`: Data directory root (default: `data`)
- `--output-root`: Root directory for output (default: `results`)

### Thinking and Agent Options
- `--enable-think`: Enable/disable thinking mode (default: `1`)
  - `1`: Enable thinking
  - `0`: Disable thinking



## Output Structure

Results are organized as:
```
{output_root}/
  {model_name}/
    {room_hash}/
      {render_mode}/
        {exp_type}/
          {think|nothink}/
            [proxy_agent]/  # Only for passive exp_type
              config.json
              exploration.json
              evaluation.json
              iamges/
```
