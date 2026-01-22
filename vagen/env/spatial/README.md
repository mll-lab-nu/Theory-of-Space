# Spatial Gym

A spatial reasoning environment where agents explore rooms and answer questions about object relationships.

## Quick Start

```python
from vagen.env.spatial import SpatialGym, SpatialGymConfig

# Create environment
config = SpatialGymConfig(exp_type='active', max_exp_steps=10)
env = SpatialGym(config)
obs, info = env.reset(seed=42)

# Explore the room
action = "Movement: [JumpTo(table)]; Final: Observe()"
obs, reward, done, info = env.step(action)

# End exploration  
action = "Movement: []; Final: Term()"
obs, reward, done, info = env.step(action)

# Answer questions
answer = "north"
obs, reward, done, info = env.step(answer)
```

### Passive Exploration
You can generate passive exploration history by `python -m vagen.env.spatial.utils.generate_history`

## How It Works

The environment has two phases:

1. **Exploration**: Agent moves around and observes the room
2. **Evaluation**: Agent answers spatial reasoning questions

## Exploration Modes

- **Active**: Agent actively explores by moving and observing (limited steps)
- **Passive**: System provides exploration history automatically

## Actions

Format: `Movement: [action1, action2]; Final: final_action`

**Movement Actions:**
- `JumpTo(object_name)` - Jump to an object
- `Rotate(90)` - Rotate 90°, 180°, or 270°
- `Return()` - Return to start

**Final Actions:**
- `Observe()` - See objects in your field of view
- `Term()` - End exploration

**Examples:**
```
Movement: [JumpTo(table), Rotate(90)]; Final: Observe()
Movement: []; Final: Observe()
Movement: []; Final: Term()
```

## Configuration

```python
config = SpatialGymConfig(
    exp_type='active',        # 'active' or 'passive'
    max_exp_steps=10,         # Max exploration steps
    field_of_view=90,         # 90° or 180° vision
    eval_tasks=[              # Types of questions
        {"task_type": "dir", "task_kwargs": {}}
    ]
)
```

## Question Types

The environment includes various spatial reasoning tasks (`vagen/env/spatial/Base/tos_base/evaluation/task_types.py`):
- **all_pairs** (text-based): Understanding all pairs of objects
- **rot** (text-based): Mental Rotation
- **dual_rot** (text/vision): Given observed object sequence (text/images), determine the rotation of the object
- **pov**: Spatial relationships from different perspectives
- **rot_pov**: Mental rotation from a different perspective
- **e2a**: Ego-to-Allocentric
- **loc** (pov_dual): Given observation, determine where the agent is
- **false_belief** (vision): False Belief with object rotation
    - **Level I**: No occlusion, only false belif
    - **Level II**: Occlusion, only find grid to observe
    - **Level III**

## File Structure

```
spatial/
├── env.py              # Main environment
├── env_config.py       # Configuration  
├── room_data/          # Room layouts
├── utils/              # Helper functions
└── Base/               # Core framework (submodule)
```

## Git Submodule Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/your-org/VAGEN.git

# Or initialize after cloning
git submodule update --init --recursive
```

## Testing

```bash
cd VAGEN/vagen/env/spatial
python env.py  # Run tests
```

## Inference
1. run `python vagen/server/server.py`
2. run `bash scripts/SpatialGym/run.sh`

- If you want to modify config, use `scripts/SpatialGym/model_config.yaml` and refer to `vagen/env/spatial/env_config.py`
- if you want to modify model for evaluation, use `scripts/SpatialGym/model_config.yaml`, these models will be evaluated like a list

## Key Features

- Text and visual observations
- Configurable field of view
- Multiple evaluation tasks
- Exploration efficiency metrics
- Reward system for learning 