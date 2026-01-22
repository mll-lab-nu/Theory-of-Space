import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import re
import streamlit as st
from vagen.env.spatial.Player_platform.env_adapter import (
    get_user_session_state,
    set_user_id,
    get_base_user_id,
)

CONFIG_PATH = os.path.join(ROOT_DIR, "vagen/env/spatial/Player_platform/config.yaml")


def _read_render_mode() -> str:
    try:
        with open(CONFIG_PATH, "r") as f:
            for line in f:
                text = line.strip()
                if text.startswith("render_mode:"):
                    return text.split(":", 1)[1].split("#", 1)[0].strip() or "vision"
    except FileNotFoundError:
        pass
    return "vision"


def _write_render_mode(mode: str) -> None:
    try:
        with open(CONFIG_PATH, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return

    def repl(match: re.Match) -> str:
        tail = match.group(3) or ""
        return f"{match.group(1)}{mode}{tail}"

    new_content, count = re.subn(
        r"(^render_mode:\s*)(\S+)(.*)$", repl, content, flags=re.MULTILINE
    )
    if count and new_content != content:
        with open(CONFIG_PATH, "w") as f:
            f.write(new_content)


st.set_page_config(page_title="Spatial Exploration Dashboard", layout="wide", page_icon="🧭")

st.title("🧭 Spatial Exploration Dashboard")

if "participant_input" not in st.session_state:
    st.session_state.participant_input = ""

# Keep the raw input separate; the session ID adds the timestamp silently.
typed_id = st.text_input(
    "Enter your participant ID",
    value=st.session_state.participant_input or "",
    placeholder="e.g., user123"
)
st.session_state.participant_input = typed_id
base_id, session_id = set_user_id(typed_id)

if not session_id:
    st.warning("⚠️ Please enter your ID before proceeding to Play.")
else:
    state = get_user_session_state(session_id)
    current_mode = state.get("render_mode") or _read_render_mode()
    render_mode = st.radio(
        "Select render mode",
        ("vision", "text"),
        index=0 if current_mode != "text" else 1,
        horizontal=True,
    )
    if render_mode != state.get("render_mode"):
        state["render_mode"] = render_mode
    if render_mode != current_mode:
        _write_render_mode(render_mode)

    st.success(f"Using session ID: {session_id}")
    st.caption(f"Participant ID: {base_id}")

st.markdown("""
Welcome! Use the **Play** page to interact with the environment turn by turn.
- Type an action → press **Submit** (or hit Enter)
- See the new observation, reward, and whether the episode is done
- **Trajectory** page shows a turn-by-turn log
- **Analytics** gives quick summaries
- **Admin** lets you set seed, max turns, and reset
""")

st.markdown("---")
st.subheader("Quick Tips")
st.markdown("""
- You can reset the episode any time from the **Admin** page.
- Your session is isolated—no server-side persistence unless you add it.
""")
