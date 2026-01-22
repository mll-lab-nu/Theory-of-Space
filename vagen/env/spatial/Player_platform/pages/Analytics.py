# pages/03_📈_Analytics.py
import streamlit as st
import pandas as pd
import json
from vagen.env.spatial.Player_platform.env_adapter import (
    get_user_session_state,
    require_user_id,
    get_base_user_id,
)

st.set_page_config(page_title="Analytics", layout="wide", page_icon="📈")
st.title("📈 Analytics")

session_id = require_user_id("Set your participant ID on the Home page before viewing analytics.")
base_id = get_base_user_id()
st.caption(f"Participant ID: {base_id} · Session ID: {session_id}")

# Share the same per-user bucket the Play page uses.
state = get_user_session_state(session_id)
env = state.get("env")

export_payload = {}
if env:
    cfg = env.cfg.to_dict()
    summary = env.get_env_summary()

    export_payload["environment_config"] = cfg
    export_payload["exploration_summary"] = summary.get("exploration_summary")
    export_payload["evaluation_summary"] = summary.get("evaluation_summary")
    export_payload["turn_logs"] = summary.get("env_turn_logs")

    correct_answers = env.get_eval_answers()
    # Use env_adapter's evaluation_manager instead of core_env
    eval_mgr = env.evaluation_manager
    if eval_mgr and eval_mgr.turn_logs:
        per_task = {}
        for log in eval_mgr.turn_logs:
            task_type = log.get("task_type") if isinstance(log, dict) else getattr(log, "task_type", None)
            is_correct = log.get("is_correct") if isinstance(log, dict) else getattr(log, "is_correct", False)
            if task_type:
                total, correct = per_task.get(task_type, (0, 0))
                per_task[task_type] = (total + 1, correct + (1 if is_correct else 0))

        rows = [
            {
                "task": task,
                "accuracy": correct / total if total else 0.0,
                "correct": correct,
                "total": total,
            }
            for task, (total, correct) in sorted(per_task.items())
        ]
        export_payload["evaluation_accuracy_by_task"] = rows
        export_payload["evaluation_answers"] = correct_answers

st.download_button(
    label="⬇️ Download JSON",
    data=json.dumps(export_payload, indent=2),
    file_name=f"analytics_{session_id}.json",
    mime="application/json",
)

if not env:
    st.info("No environment yet.")
else:
    st.subheader("Environment Config")
    st.json(cfg)

    st.subheader("Exploration Summary")
    st.json(summary["exploration_summary"])

    st.subheader("Evaluation Summary")
    st.json(summary["evaluation_summary"])

    if "evaluation_accuracy_by_task" in export_payload:
        st.subheader("Evaluation Accuracy by Task")
        st.json(export_payload["evaluation_accuracy_by_task"])

    st.subheader("Turn Logs")
    st.json(summary["env_turn_logs"])
    st.subheader("Evaluation Answers")
    st.write(correct_answers)



hist = state.get("history", [])
if not hist:
    st.info("No data yet.")
else:
    df = pd.DataFrame([{"t": h.t, "reward": h.reward} for h in hist])
    st.subheader("Reward per turn")
    st.line_chart(df.set_index("t"))
    st.subheader("Cumulative reward")
    df["cum_reward"] = df["reward"].cumsum()
    st.line_chart(df.set_index("t")[["cum_reward"]])
