# pages/02_🧭_Trajectory.py
import streamlit as st
import pandas as pd
from vagen.env.spatial.Player_platform.env_adapter import (
    get_user_session_state,
    require_user_id,
    get_base_user_id,
)

st.set_page_config(page_title="Trajectory", layout="wide", page_icon="🧭")
st.title("🧭 Trajectory Log")

session_id = require_user_id("Set your participant ID on the Home page before viewing the trajectory.")
base_id = get_base_user_id()
st.caption(f"Participant ID: {base_id} · Session ID: {session_id}")

state = get_user_session_state(session_id)
history = state.get("history", [])
if not history:
    st.info("No turns yet. Go to **Play** and start interacting.")
else:
    data = [
        {
            "t": r.t,
            "action": r.action,
            "obs": r.obs_text,
            "reward": r.reward,
            "done": r.done,
        }
        for r in history
    ]
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, height=500)
