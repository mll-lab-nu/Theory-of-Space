## SpatialGym CLI

Command line arguments for `spatial_run.py`:

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
