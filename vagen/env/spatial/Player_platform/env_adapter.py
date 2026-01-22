from __future__ import annotations
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
import json
import yaml
import re
import time
import numpy as np
from omegaconf import OmegaConf
try:
    import streamlit as st
except Exception:  # Streamlit is only available in the web app runtime.
    st = None
from vagen.env.spatial.Base.tos_base.utils.utils import format_llm_output
from vagen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType
from vagen.env.spatial.env import SpatialGym                  
from vagen.env.spatial.env_config import SpatialGymConfig
from vagen.env.spatial.Base.tos_base.utils.eval_utilities import evaluate_task_answer


_USER_STATE_KEY = "_player_sessions"
_USER_BASE_FIELD = "_player_id_base"
_USER_FULL_FIELD = "_player_id_full"


@dataclass
class EvalTask:
    """Represents a single evaluation task with question and answer."""
    task_type: str
    class_name: str
    question: str
    answer: str
    eval_data: Any = None


class PlayerEvaluationManager:
    """Manages evaluation tasks for the player platform using EvalTaskType.create_task()."""
    
    def __init__(self, eval_task_counts: Dict[str, int], room, agent, seed: int, image_dir: str = None, render_mode: str = "text"):
        self.eval_task_counts = eval_task_counts or {}
        self.room = room
        self.agent = agent
        self.seed = seed
        self.image_dir = image_dir
        self.render_mode = render_mode
        self.tasks: List[EvalTask] = []
        self.current_index = 0
        self.turn_logs: List[Dict] = []
        self._generate_tasks()
    
    def _generate_tasks(self):
        """Generate all evaluation tasks based on eval_task_counts."""
        np_random = np.random.default_rng(self.seed)
        
        for task_short, count in self.eval_task_counts.items():
            # Skip false_belief_exp as it requires running a full environment
            if task_short == 'false_belief_exp':
                continue
            
            is_vision_question = 'vision' in task_short
            if is_vision_question and self.render_mode == "text":
                continue  # Skip vision questions in text mode
            
            config = {"image_dir": self.image_dir if is_vision_question else None}
            
            for _ in range(count):
                try:
                    task = EvalTaskType.create_task(
                        task_short, np_random, self.room, self.agent, config, None
                    )
                    question_text = task.generate_question()
                    task_type = next(t for t in EvalTaskType if t.short_name == task_short)
                    class_name = task_type.class_name
                    eval_task = EvalTask(
                        task_type=task_short,
                        class_name=class_name,
                        question=question_text,
                        answer=task.eval_data.answer if hasattr(task.eval_data, 'answer') else str(task.eval_data),
                        eval_data=task.eval_data,
                    )
                    self.tasks.append(eval_task)
                except Exception as e:
                    print(f"Failed to generate task {task_short}: {e}")
    
    def get_current_task(self) -> Optional[EvalTask]:
        """Get the current evaluation task."""
        if self.current_index < len(self.tasks):
            return self.tasks[self.current_index]
        return None
    
    def submit_answer(self, answer: str) -> Tuple[bool, float]:
        """Submit an answer for the current task and move to next."""
        current_task = self.get_current_task()
        if current_task is None:
            return False, 0.0
        
        score, info = evaluate_task_answer(
            task_type=current_task.eval_data.task_type,
            pred=answer,
            answer=current_task.eval_data.answer,
            choices=getattr(current_task.eval_data, "choices", None),
        )

        is_correct = bool(score >= 0.999)

        self.turn_logs.append({
            "task_type": current_task.task_type,
            "question": current_task.question,
            "user_answer": answer,
            "correct_answer": current_task.eval_data.answer,
            "score": score,
            "is_correct": is_correct,
            "eval_info": info,
        })

        self.current_index += 1
        return is_correct, float(score)

    
    def is_complete(self) -> bool:
        """Check if all tasks are complete."""
        return self.current_index >= len(self.tasks)
    
    def get_answers(self) -> List[str]:
        """Get all correct answers."""
        return [task.answer for task in self.tasks]


def _require_streamlit():
    if not st:
        raise RuntimeError("Streamlit is required for session helpers.")
    return st


def _session_store() -> Dict[str, Dict[str, Any]]:
    """Return the shared session bucket that holds per-user dictionaries."""
    _require_streamlit()
    return st.session_state.setdefault(_USER_STATE_KEY, {})


def _sanitize_user_id(value: Optional[str]) -> str:
    """Normalize IDs so every helper speaks the same language."""
    value = (value or "").strip()
    return value or ""


def set_user_id(raw_id: str, force_new: bool = False) -> Tuple[str, str]:
    """
    Persist both the human-entered ID and its unique timestamped variant.
    Returns (base_id, session_id).
    """
    _require_streamlit()
    base = _sanitize_user_id(raw_id)
    if not base:
        st.session_state.pop(_USER_BASE_FIELD, None)
        st.session_state.pop(_USER_FULL_FIELD, None)
        st.session_state.pop("user_id", None)
        return "", ""

    stored_base = st.session_state.get(_USER_BASE_FIELD)
    session_id = st.session_state.get(_USER_FULL_FIELD)
    if force_new or base != stored_base or not session_id:
        session_id = f"{base}-{int(time.time() * 1000)}"

    st.session_state[_USER_BASE_FIELD] = base
    st.session_state[_USER_FULL_FIELD] = session_id
    st.session_state["user_id"] = session_id  # main ID now includes timestamp
    return base, session_id


def require_user_id(message: str = "Set your participant ID on the Home page.") -> str:
    """
    Stop the page early unless the participant has typed an ID.
    Returns the unique session ID (base + timestamp).
    """
    _require_streamlit()
    session_id = get_full_user_id("")
    if not session_id:
        st.warning(message)
        st.stop()
    return session_id


def get_user_session_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Keep user-scoped objects (env, history, etc.) under one key so
    concurrent participants never overwrite one another.
    """
    store = _session_store()
    key = (session_id or get_full_user_id("")).strip()
    if not key:
        raise RuntimeError("No active participant session. Call set_user_id first.")
    return store.setdefault(key, {})


def get_base_user_id(default: str = "") -> str:
    """Return the human-entered participant ID."""
    if not st:
        return default
    return (st.session_state.get(_USER_BASE_FIELD) or "").strip() or default


def get_full_user_id(default: str = "anon") -> str:
    """Return the unique ID (base+timestamp) used for storage and logging."""
    if not st:
        return default
    return (st.session_state.get(_USER_FULL_FIELD) or "").strip() or default


def bind_model_name_to_user(cfg: SpatialGymConfig, user_id: str) -> SpatialGymConfig:
    """
    Ensure model_name (used for history directories) contains a user-specific suffix.
    """
    user_tag = re.sub(r"[^A-Za-z0-9._-]", "_", _sanitize_user_id(user_id)) or "anon"
    cfg.kwargs = cfg.kwargs or {}
    model_cfg = dict(cfg.kwargs.get("model_config") or {})
    base_name = model_cfg.get("model_name") or "human_player"
    model_cfg["model_name"] = f"{base_name}-{user_tag}"
    cfg.kwargs["model_config"] = model_cfg
    return cfg
@dataclass
class TurnRecord:
    t: int
    action: str
    obs_text: str
    reward: float
    done: bool
    info: Dict[str, Any]
    obs_raw: Dict[str, Any] = None 

def load_cfg_from_yaml(path: str) -> SpatialGymConfig:
    """
    Load SpatialGymConfig from a YAML file.
    Converts eval_task_counts to eval_tasks format.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    del raw['seed-range']
    # Convert eval_task_counts to eval_tasks if present
    if "eval_task_counts" in raw and not raw.get("eval_tasks"):
        eval_tasks = []
        for task_type, count in raw["eval_task_counts"].items():
            eval_tasks.append({
                "task_type": task_type,
                "num": int(count)
            })
        raw["eval_tasks"] = eval_tasks
        # Remove eval_task_counts to avoid confusion
        del raw["eval_task_counts"]

    # OmegaConf to ensure compatibility with ListConfig, etc.
    conf = OmegaConf.create(raw)

    # Convert into dataclass
    return SpatialGymConfig(**OmegaConf.to_container(conf, resolve=True))

def _wrap_user_action_for_env(user_text: str, enable_think: bool = True) -> str:
    """
    Convert human input into the expected LLM-style format (header-based).
    """
    action = user_text.strip()
    think_content = "Placeholder" if enable_think else ""
    # Case 1: single letter answer (A, B, C, etc.)
    if re.fullmatch(r"[A-Za-z]", action):
        return format_llm_output(think_content, action, enable_think=enable_think)

    # Case 2: action sequence (contains "()")
    if "(" in action and ")" in action:
        return format_llm_output(think_content, f"Actions: [{action}]", enable_think=enable_think)
    return format_llm_output(think_content, action, enable_think=enable_think)

def save_episode(user_id: str, episode_id: int, trajectory: list, analytics: dict, correct_answers: dict = None):
    log_dir = os.path.join("vagen/env/spatial/Player_platform/logs", user_id)
    os.makedirs(log_dir, exist_ok=True)

    out_path = os.path.join(log_dir, f"episode_{episode_id}.json")

    clean_traj = []
    for r in trajectory:
        # If it's a dataclass or object with __dict__, copy it
        if hasattr(r, "__dict__"):
            d = r.__dict__.copy()
        else:
            d = dict(r)  # already a dict

        #Drop obs_raw (contains PIL images)
        d.pop("obs_raw", None)
        clean_traj.append(d)

    payload = {
        "user_id": user_id,
        "episode_id": episode_id,
        "trajectory": clean_traj,
        "analytics": analytics,
        "correct_answers": correct_answers,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path


class SpatialEnvAdapter:
    """
    Thin wrapper exposing a stable API to the Streamlit pages.
    """
    def __init__(
        self,
        cfg: SpatialGymConfig,
        debug: Optional[bool] = None, 
    ):
        self.cfg = cfg
        self.env = SpatialGym(cfg)
        self.evaluation_manager: Optional[PlayerEvaluationManager] = None
        self.is_exploration_phase = True

        if debug is None:
            debug_str = os.getenv("PLAYER_PLATFORM_DEBUG", "").strip().lower()
            debug = debug_str in {"1", "true", "yes", "y", "on"}
        self.debug = bool(debug)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        obs, info = self.env.reset(seed=seed)
        if isinstance(obs, str):
            obs = {"obs_str": obs}
        self.is_exploration_phase = True
        self.evaluation_manager = None
        return obs

    def _init_evaluation_manager(self, seed: int):
        """Initialize the evaluation manager after exploration is complete."""
        eval_task_counts = {}
        # Get eval_task_counts from config
        for task in (self.cfg.eval_tasks or []):
            task_type = task.get("task_type") if isinstance(task, dict) else getattr(task, "task_type", None)
            num = task.get("num", 1) if isinstance(task, dict) else getattr(task, "num", 1)
            if task_type:
                eval_task_counts[task_type] = num
        
        if not eval_task_counts:
            return
        
        room = self.env.initial_room
        # Use the final exploration pose for bwd_nav_rev; other tasks will reset to init.
        agent = (self.env.exploration_manager.agent.copy() if self.env.exploration_manager else self.env.initial_agent)
        image_dir = getattr(self.env.image_handler, 'image_dir', None) if hasattr(self.env, 'image_handler') else None
        render_mode = self.cfg.render_mode or "text"
        
        self.evaluation_manager = PlayerEvaluationManager(
            eval_task_counts=eval_task_counts,
            room=room,
            agent=agent,
            seed=seed,
            image_dir=image_dir,
            render_mode=render_mode,
        )
    def _attach_eval_image(self, obs: Dict[str, Any], task: Any) -> None:
        """
        Attach evaluation image in the exact format Play.py expects:
        obs["multi_modal_data"] = {"<image>": [ ... ]}
        """
        if task is None:
            return

        task_short = getattr(task, "task_type", "") or ""
        if "vision" not in task_short:
            return

        # needs cached image handler
        if not (hasattr(self.env, "image_handler") and self.env.image_handler):
            return

        eval_data = getattr(task, "eval_data", None)
        ans = getattr(eval_data, "answer", None) or {}
        final_pos = ans.get("final_pos")
        final_ori = ans.get("final_ori")
        object_positions = ans.get("object_positions", {}) or {}

        object_name = None
        if final_pos is not None:
            fp = tuple(map(int, final_pos))
            for name, pos in object_positions.items():
                try:
                    if tuple(map(int, pos)) == fp:
                        object_name = name
                        break
                except Exception:
                    pass

        img_path = self.env.image_handler.get_image_path(object_name or final_pos, final_ori)

        # Match Streamlit UI contract
        obs["multi_modal_data"] = {"<image>": [img_path]}

    def step(self, user_action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Streamlit calls this with human action text. We wrap it for the env.
        Your env.step returns: (obs: dict, reward: float, done: bool, step_info: dict)
        """
        # Handle evaluation phase
        if not self.is_exploration_phase and self.evaluation_manager is not None:
            return self._step_evaluation(user_action)
        
        enable_think = bool(self.cfg.prompt_config.get("enable_think", True))
        llm_response = _wrap_user_action_for_env(user_action, enable_think=enable_think)
        
        obs, reward, done, step_info = self.env.step(llm_response)
        # Check if exploration is done (Term action was sent)
        if done or "Term" in user_action:
            # Transition to evaluation phase
            self.is_exploration_phase = False
            self._init_evaluation_manager(self.env.current_seed or 0)
            
            # If we have evaluation tasks, prepare the first question
            if self.evaluation_manager and self.evaluation_manager.tasks:
                current_task = self.evaluation_manager.get_current_task()
                if current_task:
                    obs["obs_str"] = obs.get("obs_str", "") + "\n\n" + current_task.question
                    self._attach_eval_image(obs, current_task)
                    done = False  # Continue with evaluation
        
        return obs, reward, done, step_info
    
    def _step_evaluation(self, user_action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Handle evaluation phase step (with image output)."""
        is_correct, reward = self.evaluation_manager.submit_answer(user_action)

        obs: Dict[str, Any] = {}
        step_info = {
            "is_correct": is_correct,
            "phase": "evaluation",
        }

        # If finished
        if self.evaluation_manager.is_complete():
            if self.debug:
                obs["obs_str"] = f"Evaluation complete! Your answer was {'correct' if is_correct else 'incorrect'}."
            else:
                obs["obs_str"] = "Evaluation complete!"
            return obs, reward, True, step_info

        # Otherwise show next question
        current_task = self.evaluation_manager.get_current_task()

        if self.debug:
            obs["obs_str"] = f"Your answer was {'correct' if is_correct else 'incorrect'}.\n\n{current_task.question}"
        else:
            obs["obs_str"] = current_task.question

        try:
            self._attach_eval_image(obs, current_task)
        except Exception as e:
            # Don't let image failure break evaluation
            print(f"[Eval image attach failed] {e}")


        return obs, reward, False, step_info

    def get_eval_answers(self):
        answers = []
        if self.evaluation_manager is not None:
            return self.evaluation_manager.get_answers()
        return answers

    def max_turn(self) -> int:
        return int(self.cfg.max_exp_steps or 20)

    # ---- optional helpers surfaced to UI pages ----
    def get_env_summary(self) -> Dict[str, Any]:
        env_summary = self.env.get_env_summary()
        env_summary["exploration_summary"] = self.env.get_exp_summary()
        # Get evaluation summary from PlayerEvaluationManager if available
        if self.evaluation_manager is not None:
            env_summary["evaluation_summary"] = {
                "total_questions": len(self.evaluation_manager.tasks),
                "current_index": self.evaluation_manager.current_index,
                "turn_logs": self.evaluation_manager.turn_logs,
            }
        else:
            env_summary["evaluation_summary"] = {}
        return env_summary

    def render_cache(self) -> Dict[str, Any]:
        return self.env.render() 

def format_obs(obs: Dict[str, Any]) -> str:
    """Pick the text to display to the user (your env places FORMAT_PROMPT at the end already)."""
    return obs.get("obs_str", "").strip()

def summarize_turn(t: int, action: str, obs: Dict[str, Any], reward: float, done: bool, info: Dict[str, Any]) -> TurnRecord:
    return TurnRecord(
        t=t,
        action=action,
        obs_text=format_obs(obs),   
        reward=reward,
        done=done,
        info=info,
        obs_raw=obs                 
    )
