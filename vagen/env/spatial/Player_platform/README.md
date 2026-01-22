# SpatialGym Player Platform 

An interactive **human-in-the-loop interface** for the [`SpatialGym`](./vagen/env/spatial) environment.  
This platform allows human players to explore rooms, perform actions, and answer evaluation questions in a **chat-like interface**.

Players can:

- Use the Action Builder with dropdowns to compose actions.  
- Answer multiple-choice evaluation (MCQ) tasks with lettered options (`A`, `B`, `C`, `D`).  
- View images and text observations at each step, rendered like a chat app.  
- Save trajectories + analytics per user into structured JSON logs.  

## Running the Platform

Start the Streamlit app:
PYTHONPATH=. streamlit run vagen/env/spatial/Player_platform/app.py
For debug mode, run:
PLAYER_PLATFORM_DEBUG=1 PYTHONPATH=. streamlit run vagen/env/spatial/Player_platform/app.py
This will open a browser tab with the chat interface.
Enter an user_id to log trajectories and performance metrics.
Then go to `Play` tab to start.


## Logs & Analytics

Each episode is saved automatically to:
logs/{user_id}/episode_{n}.json
with fields:
trajectory: turn-by-turn records (actions, observations, rewards).
analytics: environment summaries (configs, tasks, metrics).
correct_answers: ground-truth for evaluation tasks.

## Configuration

Environment configs are managed by YAML files.
By default, the platform loads:
vagen/env/spatial/Player_platform/config.yaml