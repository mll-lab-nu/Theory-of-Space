import os
import json
import re
from typing import List, Dict, Tuple, Any
from vagen.env.spatial.Base.tos_base.managers.history_manager import HistoryManager


# ========================= Filenames / Constants =========================

MESSAGES_BASENAME = "messages.json"
EXPLORATION_LOG_BASENAME = "exploration_turn_logs.json"
CONFIG_BASENAME = "config.json"
STATE_BASENAME = "history_state.json"
BUILT_DIRNAME = "built_messages"

EVAL_MESSAGES_FILENAME = "eval_messages.jsonl"
EVAL_META_FILENAME = "eval_meta.jsonl"
COGMAP_MESSAGES_FILENAME = "cogmap_messages.jsonl"
COGMAP_META_FILENAME = "cogmap_meta.jsonl"


# ========================= Simple IO =========================

def read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_messages_and_meta_jsonl(messages_path: str, meta_path: str) -> Tuple[List[List[Dict]], List[Dict]]:
    messages_list: List[List[Dict]] = []
    meta_list: List[Dict] = []
    with open(messages_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            messages_list.append(obj.get("messages", []))
    with open(meta_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            meta_list.append(json.loads(line))
    assert len(messages_list) == len(meta_list)
    return messages_list, meta_list


# ========================= Path Helpers =========================
def resolve_built_root(root_dir: str, out_dir: str | None) -> str:
    """Return a single built output directory under root.

    If out_dir is absolute, use it directly; if relative, place under root_dir;
    otherwise default to <root_dir>/built_messages.
    """
    root_dir = os.path.abspath(root_dir)
    if out_dir:
        return out_dir if os.path.isabs(out_dir) else os.path.join(root_dir, out_dir)
    return os.path.join(root_dir, BUILT_DIRNAME)


def paths_for_mode(built_root: str, mode: str) -> Tuple[str, str]:
    """Return (messages_path, meta_path) files inside the single built root dir."""
    if mode == "eval":
        return (
            os.path.join(built_root, EVAL_MESSAGES_FILENAME),
            os.path.join(built_root, EVAL_META_FILENAME),
        )
    return (
        os.path.join(built_root, COGMAP_MESSAGES_FILENAME),
        os.path.join(built_root, COGMAP_META_FILENAME),
    )

def list_built_pairs(built_root: str) -> List[Tuple[str, str]]:
    """Return available (messages, meta) pairs inside the single built root dir."""
    pairs: List[Tuple[str, str]] = []
    eval_msgs = os.path.join(built_root, EVAL_MESSAGES_FILENAME)
    eval_meta = os.path.join(built_root, EVAL_META_FILENAME)
    cog_msgs = os.path.join(built_root, COGMAP_MESSAGES_FILENAME)
    cog_meta = os.path.join(built_root, COGMAP_META_FILENAME)
    if os.path.exists(eval_msgs) and os.path.exists(eval_meta):
        pairs.append((eval_msgs, eval_meta))
    if os.path.exists(cog_msgs) and os.path.exists(cog_meta):
        pairs.append((cog_msgs, cog_meta))
    return pairs


def _sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name))


def responses_jsonl_path(built_dir: str, model_name: str) -> str:
    safe = _sanitize_model_name(model_name)
    return os.path.join(built_dir, f"responses_{safe}.jsonl")


# ========================= Discovery =========================

def iter_combo_dirs(root_dir: str) -> List[str]:
    """Return all directories under root_dir that contain a messages.json file."""
    root_dir = os.path.abspath(root_dir)
    hits: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if MESSAGES_BASENAME in filenames:
            hits.append(dirpath)
    return sorted(hits)


# ========================= History Manager Loader =========================

def load_history_manager(combo_dir: str, eval_override: bool = False, all_tasks: List = None, image_dir: str = None, eval_mode: str = "default") -> HistoryManager:
    """Load HistoryManager from a combo directory using its saved state file.
    
    Args:
        combo_dir: Path to combo directory
        eval_override: If True, ignore existing evaluation history
        image_dir: Optional override for image directory
        eval_mode: Evaluation mode to use (default, prompt_cogmap, use_gt_cogmap, use_model_cogmap)
    """
    return HistoryManager.load_from_dir(combo_dir, eval_override=eval_override, all_tasks=all_tasks, image_dir=image_dir, eval_mode=eval_mode)


