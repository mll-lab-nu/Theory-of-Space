from typing import List, Dict, Tuple, Any, Optional, Union
import os
import json
import re
import numpy as np
import argparse
import copy
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


# Reuse existing components
from vagen.env.spatial.Base.tos_base import Room, Agent
from vagen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType
from vagen.env.spatial.Base.tos_base.prompts.cogmap_prompts import get_cogmap_prompt
from vagen.env.spatial.Base.tos_base.utils.utils import hash, numpy_to_python, THINK_LABEL, ANSWER_LABEL
from vagen.env.spatial.Base.tos_base.utils.image_handler import ImageHandler
from vagen.env.spatial.Base.tos_base.utils.visualization.annotate_point import load_mapping_from_meta, draw_point
# Shared common utilities/constants
from vagen.env.spatial.common import (
    load_history_manager,
)


def _detect_exp_type(combo_dir: str) -> str:
    parts = os.path.abspath(combo_dir).split(os.sep)
    return "active" if "active" in parts else "passive"


def _get_turn_number(seq: List[Dict]) -> int:
    return len([m for m in seq if m.get('role') == 'user' and (seq[0].get('role') != 'system' or True)])


def _iter_user_indices(messages: List[Dict]) -> List[int]:
    idxs: List[int] = []
    for i, m in enumerate(messages):
        if i == 0 and m.get("role") == "system":
            continue
        if m.get("role") == "user":
            idxs.append(i)
    return idxs


def _clone_until_inclusive(messages: List[Dict], end_idx: int) -> List[Dict]:
    return [m.copy() for m in messages[: end_idx + 1]]


"""Builder utilities for evaluation and cogmap message lists."""


def _add_message(out_msgs: List[List[Dict]], out_meta: List[Dict], msgs: List[Dict], meta: Dict[str, Any]) -> None:
    out_msgs.append(msgs)
    out_meta.append(meta)

def _evaluation_format_footer(enable_think: bool) -> str:
    answer_hint = "[your answer (only required answer, no extra text, notes, formatting or anything else)]"
    if enable_think:
        return f"## Output Format\n{THINK_LABEL}\n[Your thoughts on the question]\n{ANSWER_LABEL}\n{answer_hint}"
    return f"## Output Format\n{ANSWER_LABEL}\n{answer_hint}"


def _evaluation_cogmap_format_footer(enable_think: bool) -> str:
    answer_hint = "<cogmap>\n[JSON map]\n</cogmap>\n<answer>\n[your answer]\n</answer>"
    if enable_think:
        return (
            f"## Output Format\n{THINK_LABEL}\n"
            "[Your thoughts on cognitive map, then on the question]\n"
            f"{ANSWER_LABEL}\n{answer_hint}"
        )
    return f"## Output Format\n{ANSWER_LABEL}\n{answer_hint}"


def _strip_cogmap_format_rules(prompt: str) -> str:
    marker = "!!! IMPORTANT OUTPUT RULES !!!"
    if marker in prompt:
        return prompt.split(marker, 1)[0].rstrip()
    return prompt.rstrip()


def _strip_steps_left(content: str) -> str:
    return re.sub(r"You have a maximum of\s*\d+\s*exploration steps left.*", "", content, flags=re.DOTALL)


def _format_cogmap_json(cogmap_json: Dict[str, Any]) -> str:
    """Format a cogmap JSON dict into a readable string for appending to prompts."""
    if not cogmap_json:
        return ""
    return "```json\n" + json.dumps(cogmap_json, indent=2, ensure_ascii=False) + "\n```"


def _get_gt_cogmap_json(room: Room, agent: Agent) -> Dict[str, Any]:
    """Generate ground truth global cogmap JSON from room and agent."""
    from vagen.env.spatial.Base.tos_base.managers.cognitive_map_manager import CognitiveMapManager
    
    cm = CognitiveMapManager(cogmap_type="standard", pos_allow_scale=False, scope="all")
    # Build ground truth global baseroom using observed items (all objects)
    all_item_names = [o.name for o in room.all_objects]
    
    # Use the same logic as evaluate_cogmap_type for global map
    observed_set = set(all_item_names)
    gt_global_br = cm._build_gt_global_baseroom(room, agent, observed_set)
    gt_json = cm.baseroom_to_json(gt_global_br, include_gates=True)
    
    return gt_json


def _get_last_global_cogmap_response(hm) -> Optional[str]:
    for t in reversed(hm.exploration_turn_logs or []):
        global_log = (t.get("cogmap_log") or {}).get("global") or {}
        if not isinstance(global_log, dict):
            continue
        return global_log["original_response"]
        # pred_json = global_log.get("pred_json")
        # if pred_json:
        #     return _format_cogmap_json(pred_json)
    return None


def build_evaluation_from_combo(
    combo_dir: str,
    eval_task_counts: Dict[str, int],
    eval_override: bool = False,
    image_dir: str = None,
    mode: str = "default",
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Create evaluation message lists from exploration history for one sample combo dir.

    Returns (messages_list, meta_list) with meta including sample_id, task_type, question_id, message_id.

    Args:
        combo_dir: Directory containing exploration history
        eval_task_counts: Dict mapping task types to count
        seed: Seed for task generation
        eval_override: If True, ignore existing evaluation history and regenerate all questions
        mode: Evaluation mode controlling cogmap handling:
            - "default": No cogmap, normal evaluation
            - "prompt_cogmap": Ask model to output cogmap before answering
            - "use_gt_cogmap": Provide ground truth cogmap in prompt
            - "use_model_cogmap": Provide model's last global cogmap in prompt
    """
    # Load history manager with eval_override flag and mode
    hm = load_history_manager(combo_dir, eval_override=eval_override, all_tasks=list(eval_task_counts.keys()), image_dir=image_dir, eval_mode=mode)
    
    # Use messages from hm (which may have corrected paths)
    base_msgs = copy.deepcopy(hm.messages)

    enable_think = hm.get_enable_think()
    out_msgs: List[List[Dict]] = []
    meta: List[Dict] = []

    # Get existing eval counts (will be empty if eval_override=True)
    existing_ids = hm.get_eval_ids()

    room = Room.from_dict(hm.room_dict).copy()
    agent = Agent.from_dict(hm.agent_dict).copy()
    agent_init = agent.copy()
    agent_init.pos = agent_init.init_pos.copy()
    agent_init.ori = agent_init.init_ori.copy()
    if agent_init.init_room_id is not None:
        agent_init.room_id = agent_init.init_room_id
    
    # Use override image_dir from history manager (handles base dir override logic)
    image_dir = hm.image_dir
    
    image_handler = ImageHandler(image_dir=image_dir) if image_dir else None
    # Track message_ids to ensure uniqueness
    seen_message_ids = set()

    for task_short, count in (eval_task_counts or {}).items():
        # Skip false_belief_exp as it requires running a full environment
        if task_short == 'false_belief_exp':
            continue
            
        is_vision_question = False
        # filter vision question in text mode
        if 'vision' in task_short:
            if hm.observation_config['render_mode'] == "text":
                continue
            else:
                is_vision_question = True

        # All evaluation tasks start from initial state except bwd_nav_rev (starts from final pose).
        base_agent = agent if task_short == "bwd_nav_rev" else agent_init
        task = EvalTaskType.create_task(task_short, np.random.default_rng(hm.seed), room, base_agent, {"image_dir": image_dir if is_vision_question else None}, None)
        task_class_name = task.__class__.__name__

        # Calculate how many questions still needed
        existing_id_for_task = existing_ids.get(task_class_name, [])
        for i in range(count - len(existing_id_for_task)):
            # retry
            q_text = task.generate_question()
            retry = 20
            while task.eval_data.id in existing_id_for_task and retry:
                q_text = task.generate_question()
                retry -= 1
            
            if task.eval_data.id in existing_id_for_task:
                base_id = task.eval_data.id
                dup_cnt = 0
                while f"{base_id}_dup{dup_cnt}" in existing_id_for_task:
                    dup_cnt += 1
                new_id = f"{base_id}_dup{dup_cnt}"
                print(f"Warning: Duplicate question generated for {task_short} in {combo_dir} (id={base_id}). Renaming to {new_id}")
                task.eval_data.id = new_id
            
            existing_id_for_task.append(task.eval_data.id)
            assert base_msgs[-1]["role"] == "user"
            new_list = copy.deepcopy(base_msgs)
            
            # Build question content based on mode
            prefix = ""
            
            if mode == "prompt_cogmap":
                # Mode 1: Ask model to first output cogmap, then answer
                cogmap_prompt = _strip_cogmap_format_rules(get_cogmap_prompt("global", enable_think))
                prefix = (
                    cogmap_prompt
                    + "\n\nFirst output the cognitive map, then answer the following question:\n\n"
                )
            
            elif mode == "use_gt_cogmap":
                # Mode 2: Provide ground truth cogmap
                gt_cogmap = _get_gt_cogmap_json(room, base_agent)
                if gt_cogmap:
                    cogmap_str = _format_cogmap_json(gt_cogmap)
                    prefix = "\n## Reference Cognitive Map\nHere is the ground truth cognitive map of the environment:\n" + cogmap_str + "\n\n"
            
            elif mode == "use_model_cogmap":
                # Mode 3: Append last global cogmap prompt + response
                cogmap_resp = _get_last_global_cogmap_response(hm)
                if cogmap_resp:
                    base_user = _strip_steps_left(new_list[-1]["content"])
                    new_list[-1]["content"] = base_user + get_cogmap_prompt("global", enable_think)
                    new_list.append({"role": "assistant", "content": cogmap_resp})
                    new_list.append({"role": "user", "content": ""})
            
            footer = (
                _evaluation_cogmap_format_footer(enable_think)
                if mode == "prompt_cogmap"
                else _evaluation_format_footer(enable_think)
            )
            new_list[-1]['content'] = new_list[-1]['content'] + prefix + q_text + "\n\n" + footer
            
            if is_vision_question:
                if "images" not in new_list[-1]:
                    new_list[-1]["images"] = []
                object_name = None
                for name, pos in task.eval_data.answer.get('object_positions').items():
                    # Compare positions (handle both int and float tuples)
                    if tuple(map(int, pos)) == tuple(map(int, task.eval_data.answer['final_pos'])):
                        object_name = name
                        break
                new_list[-1]["images"] += [image_handler.get_image_path(object_name or task.eval_data.answer['final_pos'], task.eval_data.answer.get('final_ori'))]
            
            meta_obj = {
                "type": "evaluation",
                "task_type": task_short,
                "task_class": task.__class__.__name__,
                "question_id": task.eval_data.id,
                "combo_dir": os.path.abspath(combo_dir),
                "message_images": new_list[-1].get("images", []),
                "evaluation_data": task.eval_data.to_dict(),
                "eval_mode": mode,
            }
            meta_obj["message_id"] = hash(json.dumps(meta_obj, sort_keys=True, default=numpy_to_python))
            if meta_obj["message_id"] in seen_message_ids:
                raise ValueError(f"Duplicate message_id detected: {meta_obj['message_id']} for combo_dir={combo_dir}, task={task_short}, question_id={task.eval_data.id}")
            seen_message_ids.add(meta_obj["message_id"])
            _add_message(out_msgs, meta, new_list, meta_obj)

    return out_msgs, meta


def _generate_annotated_cogmap(cogmap_dir: str, image_dir: str, abs_candidates: List[Tuple[int, int]], agent_pos: Tuple[int, int], t_idx: int) -> str:
    """Generate annotated top_down image with candidate positions marked.
    
    Args:
        cogmap_dir: Path to the cogmap directory (combo_dir / 'cogmap')
        image_dir: Path to the image directory containing top_down_empty.png and meta_data.json
        abs_candidates: List of absolute candidate coordinates to mark on the map
        agent_pos: Agent's current position to mark on the map (as a dot)
        t_idx: Turn index to distinguish different output images
    
    Returns:
        Path to the generated annotated image, or empty string if generation failed
    """
    if not image_dir or not abs_candidates:
        return ""
    
    image_dir_path = Path(image_dir)
    cogmap_dir_path = Path(cogmap_dir)
    
    # Source files
    top_down_img = image_dir_path / "top_down_empty.png"
    meta_data_json = image_dir_path / "meta_data.json"
    
    # Output file with turn index
    os.makedirs(cogmap_dir_path, exist_ok=True)
    out_img = cogmap_dir_path / f"top_down_candidates_t{t_idx}.png"
    
    # Check if source files exist
    if not top_down_img.exists():
        raise FileNotFoundError(f"top_down_empty.png not found at {top_down_img}")
    if not meta_data_json.exists():
        raise FileNotFoundError(f"meta_data.json not found at {meta_data_json}")
    
    # Load mapping from meta_data.json
    mapping, rows, cols = load_mapping_from_meta(meta_data_json)
    
    # Create label dict: assign letters A, B, C, ... to candidates in order
    label_dict = {}
    letter_idx = 0
    for coord in abs_candidates:
        if coord in mapping:
            label_dict[coord] = chr(ord('A') + letter_idx)
            letter_idx += 1
    
    # Add agent position as a dot
    agent_pos_tuple = (int(agent_pos[0]), int(agent_pos[1]))
    if agent_pos_tuple in mapping:
        label_dict[agent_pos_tuple] = None
    
    if not label_dict:
        raise ValueError(f"No valid candidate coordinates found in mapping; cannot generate annotated cogmap. cogmap_dir: {cogmap_dir}, abs_candidates: {abs_candidates}, agent_pos: {agent_pos}")
    
    # Generate annotated image
    draw_point(top_down_img, out_img, mapping, label_dict, rows, cols, agent_pos)
    return str(out_img)

def build_cogmap_from_combo(
    combo_dir: str,
    cogmap_override: bool = False,
    image_dir: str = None,
    last_global_only: bool = False,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Create cogmap message lists strictly following cog_utils logic (local/global only).

    Args:
        combo_dir: Directory containing exploration history
        cogmap_override: If True, regenerate all cogmaps; if False, skip turns with existing cogmaps
    """
    hm = load_history_manager(combo_dir, image_dir=image_dir)
    messages = hm.messages
    turn_logs = hm.exploration_turn_logs
    
    # Derive sample_id from combo_dir path instead of image_dir
    # Format: room_hash (parent of vision/text directory)
    combo_abs = os.path.abspath(combo_dir)
    parts = combo_abs.split(os.sep)
    try:
        render_idx = max(i for i, p in enumerate(parts) if p in ("vision", "text"))
        room_hash_idx = render_idx - 1
        sample_id = parts[room_hash_idx]
    except (ValueError, IndexError):
        # Fallback if path structure is unexpected
        sample_id = os.path.basename(hm.image_dir or "sample")
    
    enable_think = hm.get_enable_think()
    exp_type = getattr(hm, "exp_type", _detect_exp_type(combo_dir))

    # Load room for determining observed room from agent position/orientation
    room = Room.from_dict(hm.room_dict).copy() if hm.room_dict else None

    # Use override image_dir from history manager (handles base dir override logic)
    image_dir = hm.image_dir

    out_msgs: List[List[Dict]] = []
    meta: List[Dict] = []

    user_idxs = _iter_user_indices(messages)

    if exp_type == "active":
        # For each turn after the first action, use previous turn index for decision
        for t_idx in range(1, len(turn_logs) + 1):
            if last_global_only and t_idx != len(turn_logs):
                continue
            if t_idx == len(turn_logs):
                # After termination, only probe global map.
                types = ["global"]
            else:
                visible = (turn_logs[t_idx - 1].get("exploration_log", {}) or {}).get("visible_objects")
                types = ["local", "global", "fog_probe"] if visible else ["global", "fog_probe"]

            if not cogmap_override:
                existing_cogmap = hm.get_cogmap(t_idx - 1) or {}
                # Filter types to only those missing from existing cogmap
                types = [t for t in types if t not in existing_cogmap]
                if not types:
                    print(f"Skipping turn {t_idx} in {combo_dir}: all cogmap types already exist")
                    continue

            # Use the next user message to attach the probe.
            # For the final turn this is the "Task finished" message after Term().
            try:
                end_idx = user_idxs[t_idx]
            except IndexError:
                print(f'sample_id: {sample_id}, t_idx: {t_idx}, user_idxs: {user_idxs}, turn_logs: {turn_logs}')
                raise IndexError
            seq = _clone_until_inclusive(messages, end_idx)
            assert seq[-1]["role"] == "user"
            base_user = _strip_steps_left(seq[-1]["content"])
            mod_seq = [m.copy() for m in seq]

            for mtype in types:
                if mtype == "fog_probe":
                    all_candidate_coords_raw = turn_logs[t_idx-1].get("exploration_log", {}).get("all_candidate_coords", [])
                    # Convert from serialized format [[x,y],...] to list of tuples
                    all_candidate_coords = [(int(pt[0]), int(pt[1])) for pt in all_candidate_coords_raw] if all_candidate_coords_raw else None
                    
                    if all_candidate_coords:
                        # Candidates are already absolute
                        abs_candidates = all_candidate_coords
                        
                        use_vision = (hm.observation_config['render_mode'] == "vision")

                        mod_seq[-1]["content"] = base_user + get_cogmap_prompt(
                            mtype, 
                            enable_think, 
                            abs_candidates, 
                            use_vision=use_vision, 
                            room=room, 
                            agent=Agent.from_dict(turn_logs[t_idx-1]['agent_state'])
                        )
                        
                        if use_vision:
                            # Get agent's current position
                            agent_current = Agent.from_dict(turn_logs[t_idx-1]['agent_state'])
                            agent_pos = (int(agent_current.pos[0]), int(agent_current.pos[1]))
                            
                            # Generate annotated cogmap image
                            cogmap_dir = os.path.join(combo_dir, 'cogmap')
                            # image_dir is already set correctly at start of function
                            annotated_img_path = _generate_annotated_cogmap(cogmap_dir, image_dir, abs_candidates, agent_pos, t_idx)
                            if annotated_img_path:
                                # Ensure we work with a fresh list for images
                                current_imgs = mod_seq[-1].get("images") or []
                                mod_seq[-1]["images"] = list(current_imgs) + [annotated_img_path]
                    else:
                        continue
                else:
                    mod_seq[-1]["content"] = base_user + get_cogmap_prompt(mtype, enable_think)
                meta_obj = {
                    "type": "cogmap",
                    "sample_id": sample_id,
                    "turn_number": t_idx,
                    "map_type": mtype,
                    "combo_dir": os.path.abspath(combo_dir),
                    "message_images": mod_seq[-1].get("images", []),
                }
                meta_obj["message_id"] = hash(json.dumps(meta_obj, sort_keys=True, default=numpy_to_python))
                _add_message(out_msgs, meta, copy.deepcopy(mod_seq), meta_obj)

    else:  # passive
        if user_idxs:
            # Check if cogmap already exists (unless override)
            if not cogmap_override:
                existing_cogmap = hm.get_cogmap(0)
                if existing_cogmap:
                    print(f"Skipping passive cogmap in {combo_dir}: cogmap already exists")
                    return out_msgs, meta
            
            end_idx = user_idxs[0]
            seq = _clone_until_inclusive(messages, end_idx)
            base_user = re.sub(r"You have a maximum of\s*\d+\s*exploration steps left.*", "", seq[-1]["content"], flags=re.DOTALL)
            mod_seq = [m.copy() for m in seq]
            mod_seq[-1]["content"] = base_user + get_cogmap_prompt("global", enable_think)
            meta_obj = {
                "type": "cogmap",
                "sample_id": sample_id,
                "turn_number": _get_turn_number(mod_seq),
                "map_type": "global",
                "combo_dir": os.path.abspath(combo_dir),
            }
            meta_obj["message_id"] = hash(json.dumps(meta_obj, sort_keys=True, default=numpy_to_python))
            _add_message(out_msgs, meta, mod_seq, meta_obj)

    return out_msgs, meta


def build_cogmap_fb_from_combo(
    combo_dir: str,
    cogmap_fb_override: bool = False,
    image_dir: str = None,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Create cogmap message lists from false belief phase logs.

    Args:
        combo_dir: Directory containing false belief history
        cogmap_fb_override: If True, regenerate all cogmaps; if False, skip turns with existing cogmaps
        image_dir: Base image directory
    """
    hm = load_history_manager(combo_dir, image_dir=image_dir)
    fb_logs = hm.false_belief_turn_logs
    
    if not fb_logs:
        print(f"No false belief logs found in {combo_dir}")
        return [], []
    
    # Derive sample_id from combo_dir path
    combo_abs = os.path.abspath(combo_dir)
    parts = combo_abs.split(os.sep)
    try:
        render_idx = max(i for i, p in enumerate(parts) if p in ("vision", "text"))
        room_hash_idx = render_idx - 1
        sample_id = parts[room_hash_idx]
    except (ValueError, IndexError):
        sample_id = os.path.basename(combo_dir)
    
    enable_think = hm.get_enable_think()
    
    out_msgs: List[List[Dict]] = []
    meta: List[Dict] = []
    
    # Build base message sequence from exploration phase
    # This includes all exploration turns
    base_messages = hm.messages.copy()
    if base_messages[-1]['role'] == 'user':
        # Remove last user message (prompt for false belief phase)
        base_messages = base_messages[:-1]
    
    # Process each false belief turn (only the last turn should have completed phase)
    for fb_idx, fb_log in enumerate(fb_logs[:-1]):
        # Only process final turn (where changes are reported)
        fb_log_data = fb_log.get("false_belief_log", {})
        if fb_log_data.get("reported_changes") is None:
            continue
        
        # Check if cogmap already exists (unless override)
        if not cogmap_fb_override:
            existing_cogmap = fb_log.get("false_belief_log", {}).get("cogmap_log")
            if existing_cogmap:
                print(f"Skipping FB turn {fb_idx} in {combo_dir}: cogmap already exists")
                continue
        
        # Build message sequence: base_messages + false belief messages up to this turn
        mod_seq = base_messages.copy()
        
        # Add false belief turn messages from the logs
        # Each false belief turn has user_message and assistant_raw_message
        for i in range(fb_idx + 1):
            fb_turn = fb_logs[i]
            user_msg = fb_turn.get("user_message", "")
            assistant_msg = fb_turn.get("assistant_raw_message", "")
            assert user_msg and assistant_msg, f"Missing messages in FB turn {i} of {combo_dir}"
            mod_seq.append({"role": "user", "content": user_msg,
                            "images": fb_logs[i-1].get("message_images", []) if i > 0 else []})
            mod_seq.append({"role": "assistant", "content": assistant_msg})
        
        
        next_fb_turn = fb_logs[fb_idx + 1]
        next_user_msg = next_fb_turn.get("user_message", "")
        
        # Remove step counter from the user message
        base_user = re.sub(r"You have a maximum of\s*\d+\s*exploration steps left.*", "", next_user_msg, flags=re.DOTALL)
        # Append false-belief cogmap prompt (global + include ALL objects)
        cogmap_prompt = get_cogmap_prompt("global_fb", enable_think)
        mod_seq.append({"role": "user", "content": base_user + cogmap_prompt,
                        "images": fb_logs[fb_idx].get("message_images", [])})
        
        turn_num = fb_log.get("turn_number", fb_idx + 1)
        
        meta_obj = {
            "type": "cogmap_fb",
            "sample_id": sample_id,
            "turn_number": turn_num,
            "fb_turn_index": fb_idx,
            "map_type": "global",
            "combo_dir": os.path.abspath(combo_dir),
            "message_images": [],
        }
        meta_obj["message_id"] = hash(json.dumps(meta_obj, sort_keys=True, default=numpy_to_python))
        _add_message(out_msgs, meta, copy.deepcopy(mod_seq), meta_obj)
    
    return out_msgs, meta



# ========================= Root-level Builders =========================

def build_all_for_combo_dirs(
    combo_dirs: List[str],
    mode: str = "eval",
    eval_task_counts: Dict[str, int] | None = None,
    eval_override: bool = False,
    cogmap_override: bool = False,
    cogmap_fb_override: bool = False,
    image_dir: str = None,
    last_global_only: bool = False,
    eval_mode: str = "default",
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Build messages/meta for a specific list of combo directories.
    
    Args:
        combo_dirs: List of combo directory paths to process
        mode: 'eval', 'cogmap', or 'cogmap_fb'
        eval_task_counts: Dict mapping task types to count (for eval mode)
        eval_override: If True, ignore existing evaluation history and regenerate all
        cogmap_override: If True, regenerate all cogmaps; if False, skip existing cogmaps
        cogmap_fb_override: If True, regenerate all false belief cogmaps
        eval_mode: Evaluation mode for cogmap handling (only used when mode='eval')
            - "default": No cogmap, normal evaluation
            - "prompt_cogmap": Ask model to output cogmap before answering
            - "use_gt_cogmap": Provide ground truth cogmap in prompt
            - "use_model_cogmap": Provide model's last global cogmap in prompt
    
    Returns:
        Tuple of (messages_list, meta_list)
    """
    all_msgs: List[List[Dict]] = []
    all_meta: List[Dict] = []
    
    for combo in combo_dirs:
        if mode == "eval":
            assert eval_task_counts is not None, "eval_task_counts must be provided for eval mode"
            msgs, meta = build_evaluation_from_combo(
                combo, eval_task_counts, 
                eval_override=eval_override,
                image_dir=image_dir,
                mode=eval_mode,
            )
        elif mode == "cogmap_fb":
            msgs, meta = build_cogmap_fb_from_combo(combo, cogmap_fb_override=cogmap_fb_override, image_dir=image_dir)
        else:
            msgs, meta = build_cogmap_from_combo(
                combo,
                cogmap_override=cogmap_override,
                image_dir=image_dir,
                last_global_only=last_global_only,
            )
        all_msgs.extend(msgs)
        all_meta.extend(meta)
    
    return all_msgs, all_meta
