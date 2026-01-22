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

### Quick start to reproduce the results
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

### Override Options
- `--all-override`: Override all history (delete entire sample path)
- `--eval-override`: Override evaluation history only
- `--cogmap-override`: Override cognitive map cache only
- `--cogmap`: Enable cognitive map phase in `all` mode

### Evaluation Options
- `--eval-task-counts`: JSON string specifying evaluation task counts
  - Example: `'{"dir": 1, "loc": 2}'`
  - If omitted, uses `eval_task_counts` from `inference_config.yaml`

### Inference Options
- `--inference-mode`: Inference execution mode (default: `direct`)
  - `direct`: Direct API calls
  - `batch`: OpenAI batch API

### Server Options
- `--no-server`: Don't start internal environment server (assume external server running)
- `--server-host`: Server host (default: `127.0.0.1`)
- `--server-port`: Server port (default: `5000`)
  - Automatically finds available port if specified port is in use

### Configuration Files
- `--base-env`: Path to base environment config (default: `base_env_config.yaml`)
- `--base-infer`: Path to inference config (default: `inference_config.yaml`)
- `--base-model`: Path to base model config (default: `base_model_config.yaml`)

## Workflow

### Phase: Exploration
1. Load base configurations and room config
2. Start environment server (if not disabled)
3. For each combination of `exp_type` and `render_mode`:
   - Generate temporary YAML configurations
   - Create dataset using `vagen.env.create_dataset`
   - Run exploration inference
4. Stop environment server

### Phase: Evaluation
1. Compute combo directory paths from previous exploration
2. Load evaluation task counts
3. Build evaluation messages for all combo directories
4. Run evaluation inference

### Phase: Cognitive Map
1. Compute combo directory paths from previous exploration
2. Build cognitive map messages for all combo directories
3. Run cognitive map inference

### Phase: Aggregation
1. Aggregate all logs and images using `SpatialEnvLogger`
2. Generate consolidated results

## Configuration Structure

### Environment Config (`base_env_config.yaml`)
Must contain `room_config` for spatial environment setup:
```yaml
room_config:
  n_objects: 9
  room_num: 1
  topology: "single"
  room_size: [10, 10]
```

### Inference Config (`inference_config.yaml`)
Must contain inference parameters and optional `eval_task_counts`:
```yaml
output_dir: "results"
eval_task_counts:
  dir: 1
  loc: 2
```

### Model Config (`base_model_config.yaml`)
Must contain a `models` section:
```yaml
models:
  gpt-4o-mini:
    model_name: "gpt-4o-mini"
    # ... other model parameters
```

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
