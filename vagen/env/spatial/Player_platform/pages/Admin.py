# pages/04_⚙️_Admin.py
import streamlit as st
from vagen.env.spatial.Player_platform.env_adapter import (
    SpatialEnvAdapter,
    bind_model_name_to_user,
    get_user_session_state,
    get_base_user_id,
    load_cfg_from_yaml,
    require_user_id,
    set_user_id,
    summarize_turn,
)

st.set_page_config(page_title="Admin", layout="wide", page_icon="⚙️")

st.title("⚙️ Admin")
CONFIG_PATH = "vagen/env/spatial/Player_platform/config.yaml"

session_id = require_user_id("Set your participant ID on the Home page before using Admin.")
base_id = get_base_user_id()
st.caption(f"Participant ID: {base_id} · Session ID: {session_id}")

state = get_user_session_state(session_id)
env_adapter = state.get("env")
if not env_adapter:
    st.warning("⚠️ No environment loaded yet. Go to Play first.")
    st.stop()

core_env = env_adapter.env
seed = st.number_input("Seed", min_value=0, value=int(state.get("current_seed", 0)), step=1)
if st.button("Reset Episode"):
    seed_value = int(seed)
    base_id = get_base_user_id()
    _, new_session_id = set_user_id(base_id, force_new=True)
    new_state = get_user_session_state(new_session_id)
    cfg = bind_model_name_to_user(load_cfg_from_yaml(CONFIG_PATH), new_session_id)
    env_adapter = SpatialEnvAdapter(cfg)
    obs, info = env_adapter.env.reset(seed=seed_value)
    init_obs = {"obs_str": obs} if isinstance(obs, str) else obs

    new_state.clear()
    new_state.update({
        "env": env_adapter,
        "loaded_config_path": CONFIG_PATH,
        "current_seed": seed_value,
        "force_reload": False,
        "last_obs": init_obs,
        "history": [
            summarize_turn(
                t=0,
                action="(system)",
                obs=init_obs,
                reward=0.0,
                done=False,
                info=info or {},
            )
        ],
        "episode_id": 1,
        "turn": 0,
        "action_buffer": ["Observe()"],
        "scroll_to_bottom": True,
    })

    st.success(f"Episode reset. New session ID: {new_session_id}")
    st.rerun()
