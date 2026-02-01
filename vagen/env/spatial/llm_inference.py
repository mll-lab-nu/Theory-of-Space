import os
import json
import time
from typing import List, Dict, Tuple, Any
import argparse

from vagen.env.spatial.Base.tos_base.managers.cognitive_map_manager import CognitiveMapManager
from vagen.env.spatial.Base.tos_base.utils.cog_utils import _evaluate_cogmaps
from vagen.env.spatial.Base.tos_base.evaluation.tasks import evaluate_from_dict
from vagen.env.spatial.Base.tos_base.utils.eval_utilities import evaluate_task_answer_with_cogmap
from vagen.inference.model_interface.factory_model import ModelFactory
from vagen.env.spatial.batch_processor import get_batch_processor
from vagen.env.spatial.Base.tos_base.utils.utils import parse_llm_response
import dotenv
dotenv.load_dotenv()

from vagen.env.spatial.common import (
    resolve_built_root,
    paths_for_mode,
    list_built_pairs,
    load_messages_and_meta_jsonl as common_load_messages_and_meta_jsonl,
    responses_jsonl_path,
    iter_combo_dirs,
    load_history_manager,
)


"""Root-only inference runner: reads built files once and maps responses back via HistoryManager.load_from_dir."""





# ========================= Direct Generate via Model Interface =========================

def generate_with_model_interface(
    model_config: dict,
    messages_list: List[List[Dict[str, Any]]],
    metas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Use ModelFactory to create interface based on provider field
    interface = ModelFactory.create(model_config)
    
    results = interface.generate(messages_list)
    outputs: List[Dict[str, Any]] = []
    for i, r in enumerate(results):
        if r is None:
            continue
        outputs.append({
            "message_id": (metas[i] or {}).get("message_id"),
            "text": r.get("text", ""),
            "usage": r.get("usage", {}),
        })
    return outputs


# ========================= Output Utilities =========================

def index_meta_by_id(metas: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(m.get("message_id")): m for m in metas}


def save_outputs_jsonl(outputs: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def group_and_map_outputs(outputs: List[Dict[str, Any]], metas: List[Dict[str, Any]]) -> tuple:
    """
    Group outputs by `combo_dir` using the provided `metas` and map them
    back into histories via `map_llm_responses`.

    Returns a tuple (processed_count, set_of_combo_dirs_updated).
    """
    meta_by_id = index_meta_by_id(metas)
    combo_data: Dict[str, Dict[str, List]] = {}

    for out in outputs:
        mid = str(out.get("message_id"))
        m = meta_by_id.get(mid)
        if not m:
            continue
        cdir = m.get("combo_dir")
        combo_data.setdefault(cdir, {"outputs": [], "metas": []})
        combo_data[cdir]["outputs"].append(out)
        combo_data[cdir]["metas"].append(m)

    combos_updated = set()
    processed = 0
    for cdir, data in combo_data.items():
        try:
            map_llm_responses(cdir, data["metas"], data["outputs"])
            combos_updated.add(cdir)
            processed += len(data["outputs"])
        except Exception as e:
            print(f"Error mapping responses for combo {cdir}: {e}", flush=True)
            continue

    return processed, combos_updated


# ========================= Input Loading (prebuilt) =========================
def load_prebuilt_inputs(built_root: str) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Load prebuilt inputs from built root directory."""
    pairs = list_built_pairs(built_root)
    messages_list: List[List[Dict[str, Any]]] = []
    metas: List[Dict[str, Any]] = []
    for msgs_path, meta_path in pairs:
        msgs, meta = common_load_messages_and_meta_jsonl(msgs_path, meta_path)
        messages_list.extend(msgs)
        metas.extend(meta)
    return messages_list, metas


def load_prebuilt_inputs_under_root(root_dir: str, out_dir: str | None) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Load prebuilt inputs from root directory."""
    built_root = resolve_built_root(root_dir, out_dir)
    return load_prebuilt_inputs(built_root)


def get_done_ids(responses_path: str) -> set:
    if not os.path.exists(responses_path):
        return set()
    done = set()
    with open(responses_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                mid = str(obj.get("message_id"))
                if mid:
                    done.add(mid)
            except Exception:
                continue
    return done


# ========================= Mapping LLM Responses back to History =========================

def map_llm_responses(
    combo_dir: str,
    metas: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    cogmap_config: Dict[str, Any] | None = None,
) -> None:
    """
    Map responses (aligned to metas) into the correct history files.
    - evaluation: write per-question via HistoryManager.update_eval_turn_log
    - cogmap: evaluate and write via HistoryManager.update_cogmap
    """
    # Extract eval_mode from first evaluation meta (all should have same mode)
    eval_mode = "default"
    for meta in metas:
        if (meta.get("type") or "").lower() == "evaluation":
            eval_mode = meta.get("eval_mode", "default")
            break
    
    history = load_history_manager(combo_dir, eval_mode=eval_mode)
    sample_cfg = json.load(open(history.state_path))

    meta_by_id = index_meta_by_id(metas)

    # Group for cogmap: (turn_idx) -> {map_type: text}
    cogmap_groups: Dict[int, Dict[str, str]] = {}
    cogmap_message_images: Dict[int, Dict[str, List[str]]] = {}
    # Group for cogmap_fb: (fb_turn_index) -> text
    cogmap_fb_groups: Dict[int, str] = {}

    def _to_rel_imgs(imgs: Any) -> List[str]:
        out: List[str] = []
        for p in (imgs or []):
            if not isinstance(p, str) or not p:
                continue
            out.append(os.path.relpath(p, history.model_path) if os.path.isabs(p) else p)
        return out

    for out in outputs:
        mid = str(out.get("message_id"))
        meta = meta_by_id.get(mid)
        if not meta:
            continue
        text = out.get("text", "")
        if (meta.get("type") or "").lower() == "evaluation":
            qid = meta["question_id"]
            if history.has_question(qid):
                print("question repeated:", qid)
                continue  # Skip existing
            eval_data = (meta.get("evaluation_data") or {})
            # Evaluate using same logic as in env runtime
            _, answer, _ = parse_llm_response(text)
            
            if eval_mode == "prompt_cogmap":
                score, info, cogmap = evaluate_task_answer_with_cogmap(
                    task_type=eval_data.get("task_type", ""),
                    pred=answer,
                    answer=eval_data.get("answer"),
                    choices=eval_data.get("choices")
                )
            else:
                score, info = evaluate_from_dict(eval_data, answer)
            task_class = meta.get("task_class") or meta.get("task_type")
            turn_log = {
                "is_exploration_phase": False,
                "evaluation_log": {
                    "task_type": task_class,
                    "user_answer": text,
                    "score": score,
                    "evaluation_info": info or {},
                    "evaluation_data": eval_data,
                },
                "assistant_raw_message": text,
                "room_state": sample_cfg["room_dict"],
                "agent_state": sample_cfg["agent_dict"],
                "message_images": meta.get("message_images", []),
                "turn_number": 1,
            }
            history.update_eval_turn_log(turn_log)

        elif (meta.get("type") or "").lower() == "cogmap":
            tnum = int(meta.get("turn_number", 1))
            t_idx = tnum - 1
            mtype = str(meta.get("map_type", "global") or "global")
            cogmap_groups.setdefault(t_idx, {})[mtype] = text
            imgs = _to_rel_imgs(meta.get("message_images"))  # for fog probe (vision)
            if imgs:
                cogmap_message_images.setdefault(t_idx, {})[mtype] = imgs
        
        elif (meta.get("type") or "").lower() == "cogmap_fb": # for false belief cogmap
            fb_turn_idx = int(meta.get("fb_turn_index", 0))
            cogmap_fb_groups[fb_turn_idx] = text

    # Process cogmap groups
    if cogmap_groups:
        for idx, turn_log in enumerate(history.exploration_turn_logs):
            existing_cogmap = turn_log.get("cogmap_log", {}) or {}
            # Extract original responses saved previously
            existing_responses_by_type: Dict[str, str] = {}
            for map_type, map_data in existing_cogmap.items():
                if isinstance(map_data, dict) and map_data.get("original_response"):
                    existing_responses_by_type[map_type] = map_data["original_response"]

            if existing_responses_by_type:
                if idx not in cogmap_groups:
                    cogmap_groups[idx] = dict(existing_responses_by_type)
                else:
                    # fill missing map types with existing originals
                    for k, v in existing_responses_by_type.items():
                        cogmap_groups[idx].setdefault(k, v)

        cm_cfg = cogmap_config or {"cogmap_type": "standard", "pos_allow_scale": False, "scope": "all"}
        cm = CognitiveMapManager(**cm_cfg)

        for t_idx, resp_by_type in cogmap_groups.items():
            if not (0 <= t_idx < len(history.exploration_turn_logs)):
                continue
            turn_log = history.exploration_turn_logs[t_idx]
            try:
                cogmap_log = _evaluate_cogmaps(cm, resp_by_type, turn_log)
                result = cogmap_log.to_dict() if cogmap_log else {}
            except Exception as e:
                print(f"Error evaluating cogmap for turn {t_idx} in {combo_dir}: {e}")
                result = {k: {"original_response": v} for k, v in resp_by_type.items()}
            # Attach prompt images (e.g., fog-probe annotated top-down candidates) for visualization.
            for mtype, imgs in (cogmap_message_images.get(t_idx) or {}).items():
                if not imgs:
                    continue
                result.setdefault(mtype, {})
                if isinstance(result.get(mtype), dict):
                    result[mtype]["message_images"] = imgs
            history.update_cogmap({
                "is_exploration_phase": True,
                "turn_number": t_idx + 1,
                "cogmap_log": result,
            })
    
    # Process cogmap_fb groups
    if cogmap_fb_groups:
        cm_cfg = cogmap_config or {"cogmap_type": "standard", "pos_allow_scale": False, "scope": "all"}
        cm = CognitiveMapManager(**cm_cfg)
        # Get last exploration turn for pos_norm_L computation and changed_exp metric
        last_exp = next((t for t in reversed(history.exploration_turn_logs) if t.get('cogmap_log')), None)
        exp_room = (last_exp or {}).get('room_state')
        exp_agent = (last_exp or {}).get('agent_state')
        last_exp_cogmap = (last_exp or {}).get('cogmap_log')
        
        for fb_idx, response_text in cogmap_fb_groups.items():
            if not (0 <= fb_idx < len(history.false_belief_turn_logs)):
                continue
            fb_turn_log = history.false_belief_turn_logs[fb_idx]
            fb_turn_log.setdefault('false_belief_log', {})['cogmap_log'] = cm.evaluate_false_belief_cogmap(
                response_text, fb_turn_log, exp_room, exp_agent, last_exp_cogmap)
            fb_turn_log.pop('cogmap_log', None)
        
        # Save updated false belief logs
        history.save_false_belief()

    history.save()


# ========================= Re-evaluation of existing answers =========================

def reevaluate_combo_dir(combo_dir: str) -> int:
    """Re-evaluate all existing evaluation answers in a combo directory.
    
    Args:
        combo_dir: Path to combo directory
        
    Returns:
        Number of answers re-evaluated
    """
    history = load_history_manager(combo_dir)
    
    # Get all evaluation turn logs
    eval_logs = history.evaluation_turn_logs
    if not eval_logs:
        print(f"No evaluation logs found in {combo_dir}")
        return 0
    
    count = 0
    for questions in eval_logs.values():
        for question in questions.values():
            eval_log = question.get("evaluation_log", {})
            if not eval_log:
                continue

            # Get existing data
            eval_data = eval_log.get("evaluation_data")
            user_answer_raw = eval_log.get("user_answer", "")
            
            if not eval_data:
                continue
            
            # Parse answer from raw message (same as in map_llm_responses)
            _, answer, _ = parse_llm_response(user_answer_raw)
            
            # Re-evaluate
            score, info = evaluate_from_dict(eval_data, answer)
            
            # Update the log
            eval_log["score"] = score
            eval_log["evaluation_info"] = info or {}
            
            count += 1
    
    # Save updated history
    history.save()
    print(f"Re-evaluated {count} answers in {combo_dir}")
    return count


def reevaluate_combo_dirs(combo_dirs: List[str]) -> None:
    """Re-evaluate all existing evaluation answers in multiple combo directories.

    Args:
        combo_dirs: List of combo directory paths
    """
    total_count = 0
    for combo_dir in combo_dirs:
        try:
            count = reevaluate_combo_dir(combo_dir)
            total_count += count
        except Exception as e:
            print(f"Error re-evaluating {combo_dir}: {e}")
            continue

    print(f"\nRe-evaluation completed: {total_count} answers re-evaluated across {len(combo_dirs)} combo directories.")


def reevaluate_cogmaps_combo_dir(combo_dir: str) -> int:
    """Re-evaluate all existing cognitive maps in a combo directory.

    This function reads existing cogmap responses from history and re-evaluates
    them using the current evaluation logic, without regenerating the responses.

    Args:
        combo_dir: Path to combo directory

    Returns:
        Number of cognitive maps re-evaluated
    """
    from vagen.env.spatial.Base.tos_base.managers.cognitive_map_manager import CognitiveMapManager

    history = load_history_manager(combo_dir)

    # Initialize cognitive map manager with default config
    cm = CognitiveMapManager(cogmap_type="standard", pos_allow_scale=False, scope="all")

    count = 0
    # Re-evaluate exploration turn cogmaps
    for turn_idx, turn_log in enumerate(history.exploration_turn_logs):
        cogmap = turn_log.get("cogmap_log", {})
        if not cogmap:
            continue

        # Extract original responses and preserve message_images
        responses_by_type = {}
        existing_message_images: Dict[str, List[str]] = {}
        for map_type, map_data in cogmap.items():
            if isinstance(map_data, dict) and map_data.get("original_response"):
                responses_by_type[map_type] = map_data["original_response"]
                if map_data.get("message_images"):
                    existing_message_images[map_type] = map_data["message_images"]

        if not responses_by_type:
            continue

        # Re-evaluate using current turn log's ground truth
        try:
            cogmap_log = _evaluate_cogmaps(cm, responses_by_type, turn_log)
            result = cogmap_log.to_dict() if cogmap_log else {}
            # Restore message_images (e.g., fog_probe annotated images)
            for mtype, imgs in existing_message_images.items():
                if mtype in result and isinstance(result[mtype], dict):
                    result[mtype]["message_images"] = imgs
            turn_log["cogmap_log"] = result
            history.update_cogmap({
                "is_exploration_phase": True,
                "turn_number": turn_idx + 1,
                "cogmap_log": turn_log["cogmap_log"],
            })
            count += 1
        except Exception as e:
            print(f"Error re-evaluating cogmap for turn {turn_idx} in {combo_dir}: {e}")
            continue

    # Save updated history
    history.save()
    print(f"Re-evaluated {count} cognitive maps in {combo_dir}")
    return count


def reevaluate_cogmaps_combo_dirs(combo_dirs: List[str]) -> None:
    """Re-evaluate all existing cognitive maps in multiple combo directories.

    Args:
        combo_dirs: List of combo directory paths
    """
    total_count = 0
    for combo_dir in combo_dirs:
        try:
            count = reevaluate_cogmaps_combo_dir(combo_dir)
            total_count += count
        except Exception as e:
            print(f"Error re-evaluating cogmaps in {combo_dir}: {e}")
            continue

    print(f"\nCogmap re-evaluation completed: {total_count} cognitive maps re-evaluated across {len(combo_dirs)} combo directories.")


def reevaluate_cogmap_fb_combo_dir(combo_dir: str) -> int:
    """Re-evaluate all existing false belief cognitive maps in a combo directory.

    This function reads existing false belief cogmap responses from history and 
    re-evaluates them using the current evaluation logic with separate metrics 
    for changed and unchanged objects.

    Args:
        combo_dir: Path to combo directory

    Returns:
        Number of false belief cognitive maps re-evaluated
    """
    from vagen.env.spatial.Base.tos_base.managers.cognitive_map_manager import CognitiveMapManager

    history = load_history_manager(combo_dir)

    # Initialize cognitive map manager with default config
    cm = CognitiveMapManager(cogmap_type="standard", pos_allow_scale=False, scope="all")
    # Get last exploration turn for pos_norm_L computation and changed_exp metric
    last_exp = next((t for t in reversed(history.exploration_turn_logs) if t.get('cogmap_log')), None)
    exp_room = (last_exp or {}).get('room_state')
    exp_agent = (last_exp or {}).get('agent_state')
    last_exp_cogmap = (last_exp or {}).get('cogmap_log')

    count = 0
    # Re-evaluate false belief turn cogmaps
    for fb_idx, fb_turn_log in enumerate(history.false_belief_turn_logs):
        # Check both locations: inside false_belief_log (preferred) or top-level (legacy)
        cogmap_log = (fb_turn_log.get("false_belief_log") or {}).get("cogmap_log")
        if not cogmap_log:
            cogmap_log = fb_turn_log.get("cogmap_log")
            
        if not cogmap_log:
            continue

        # Extract original response
        original_response = cogmap_log.get("original_response")
        if not original_response:
            continue

        # Re-evaluate with separate metrics for changed and unchanged objects
        try:
            cogmap_result = cm.evaluate_false_belief_cogmap(original_response, fb_turn_log, exp_room, exp_agent, last_exp_cogmap)
            fb_turn_log.setdefault("false_belief_log", {})["cogmap_log"] = cogmap_result
            fb_turn_log.pop("cogmap_log", None)
            count += 1
        except Exception as e:
            print(f"Error re-evaluating false belief cogmap for FB turn {fb_idx} in {combo_dir}: {e}")
            continue

    # Save updated false belief logs
    if count > 0:
        history.save_false_belief()
    
    print(f"Re-evaluated {count} false belief cognitive maps in {combo_dir}")
    return count


def reevaluate_cogmap_fb_combo_dirs(combo_dirs: List[str]) -> None:
    """Re-evaluate all existing false belief cognitive maps in multiple combo directories.

    Args:
        combo_dirs: List of combo directory paths
    """
    total_count = 0
    for combo_dir in combo_dirs:
        try:
            count = reevaluate_cogmap_fb_combo_dir(combo_dir)
            total_count += count
        except Exception as e:
            print(f"Error re-evaluating false belief cogmaps in {combo_dir}: {e}")
            continue

    print(f"\nFalse belief cogmap re-evaluation completed: {total_count} false belief cognitive maps re-evaluated across {len(combo_dirs)} combo directories.")


# ========================= Combo-level inference =========================

def run_inference_for_combo_dirs(
    combo_dirs: List[str],
    model_config: dict,
    mode: str = "eval",
    eval_task_counts: Dict[str, int] | None = None,
    inference_mode: str = "direct",
    eval_override: bool = False,
    cogmap_override: bool = False,
    cogmap_fb_override: bool = False,
    image_dir: str = None,
    last_global_only: bool = False,
    eval_mode: str = "default",
) -> None:
    """Run inference for a specific list of combo directories.

    Args:
        combo_dirs: List of combo directory paths
        model_name: Model name for inference
        mode: 'eval', 'cogmap', or 'cogmap_fb'
        eval_task_counts: Evaluation task counts (for eval mode)
        seed: Seed for task generation (for eval mode)
        inference_mode: 'batch' or 'direct'
        eval_override: If True, ignore existing evaluation history and regenerate all
        cogmap_override: If True, regenerate all cogmaps; if False, skip existing cogmaps
        cogmap_fb_override: If True, regenerate all false belief cogmaps
        eval_mode: Evaluation mode for cogmap handling (only used when mode='eval')
            - "default": No cogmap, normal evaluation
            - "prompt_cogmap": Ask model to output cogmap before answering
            - "use_gt_cogmap": Provide ground truth cogmap in prompt
            - "use_model_cogmap": Provide model's last global cogmap in prompt
    """
    from vagen.env.spatial.message_list_builder import build_all_for_combo_dirs
    
    # Build messages for the specified combos (override logic handled in builder)
    all_msgs, all_meta = build_all_for_combo_dirs(
        combo_dirs,
        mode=mode,
        eval_task_counts=eval_task_counts,
        eval_override=eval_override,
        cogmap_override=cogmap_override,     
        cogmap_fb_override=cogmap_fb_override,
        image_dir=image_dir,
        last_global_only=last_global_only,
        eval_mode=eval_mode,
    )
    
    if not all_msgs or not all_meta:
        print(f"No messages generated for {mode} mode", flush=True)
        return
    
    # Run inference
    if inference_mode == "batch":
        processor = get_batch_processor(model_config)
        batch_ids = processor.submit(all_msgs, all_meta)
        print(f"Submitted batch: {batch_ids}, Messages: {len(all_msgs)}", flush=True)
        outputs = processor.retrieve(batch_ids)

        processed, combos = group_and_map_outputs(outputs, all_meta)
        print(f"Completed {mode} inference for {len(combos)} combos, processed {processed} responses.", flush=True)

    else:
        # Direct mode: chunk into batches to avoid losing all progress on errors.
        CHUNK_SIZE = 1024
        total_processed = 0
        combos_updated = set()

        # Process in chunks and map/save results after each chunk
        for start in range(0, len(all_msgs), CHUNK_SIZE):
            chunk_msgs = all_msgs[start : start + CHUNK_SIZE]
            chunk_meta = all_meta[start : start + CHUNK_SIZE]
            try:
                chunk_outputs = generate_with_model_interface(model_config, chunk_msgs, chunk_meta) or []
            except Exception as e:
                print(f"Error during direct inference chunk {start}-{start+len(chunk_msgs)}: {e}", flush=True)
                # continue to next chunk (do not abort entire run)
                continue

            processed, combos = group_and_map_outputs(chunk_outputs, chunk_meta)
            total_processed += processed
            combos_updated.update(combos)

        print(f"Completed {mode} direct inference in chunks; processed {total_processed} responses across {len(combos_updated)} combos.", flush=True)


# ========================= __main__ demos =========================

def main_infer() -> None:
    parser = argparse.ArgumentParser(description="Run LLM inference on prebuilt messages and map results by message_id")
    parser.add_argument("--root-dir", required=True, help="Root dir to scan for combo dirs")
    parser.add_argument("--model-name", default="gpt-4o-mini")
    parser.add_argument("--mode", choices=["batch", "direct"], default="batch")
    parser.add_argument("--out-dir", default=None, help="Built output dir under root (single folder)")
    parser.add_argument("--override", action="store_true", help="If set, resend all and overwrite responses")
    args = parser.parse_args()

    # Gather inputs
    all_msgs, all_meta = load_prebuilt_inputs_under_root(args.root_dir, args.out_dir)
    built_dir = resolve_built_root(args.root_dir, args.out_dir)

    os.makedirs(built_dir, exist_ok=True)
    responses_path = responses_jsonl_path(built_dir, args.model_name)

    assert all_msgs and all_meta and len(all_msgs) == len(all_meta), "No inputs loaded or mismatched lengths"

    # Use message_id from builder meta directly
    metas_with_ids: List[Dict[str, Any]] = []
    for meta in all_meta:
        meta = dict(meta)
        assert "message_id" in meta, "Message ID is required in meta"
        metas_with_ids.append(meta)

    # Skip already processed unless override
    if not args.override:
        done_ids = get_done_ids(responses_path)
        filtered = [(msg, meta) for msg, meta in zip(all_msgs, metas_with_ids) if str(meta["message_id"]) not in done_ids]
        if filtered:
            all_msgs, metas_with_ids = zip(*filtered)
            all_msgs, metas_with_ids = list(all_msgs), list(metas_with_ids)
        else:
            all_msgs, metas_with_ids = [], []

    if not all_msgs or not metas_with_ids:
        print("No new messages to process. Skipping.")
        raise SystemExit(0)

    # Run inference
    if args.mode == "batch":
        # Need to construct model_config from args or minimal defaults
        # Since main_infer is a demo, we might not have full config. 
        # But BatchProcessor expects a dict.
        model_config = {"model_name": args.model_name}
        processor = get_batch_processor(model_config)
 
        batch_ids = processor.submit(all_msgs, metas_with_ids)
        print(f"Submitted batch: {batch_ids}")
        outputs = processor.retrieve(batch_ids)
    else:
        model_config = {"model_name": args.model_name}
        outputs = generate_with_model_interface(model_config, all_msgs, metas_with_ids)

    # Append responses
    save_outputs_jsonl(outputs, responses_path)

    # Map back to histories per combo
    # Group outputs by combo_dir
    meta_by_id = index_meta_by_id(metas_with_ids)
    combo_to_outputs: Dict[str, List[Dict[str, Any]]] = {}
    combo_to_metas: Dict[str, List[Dict[str, Any]]] = {}
    for out in outputs:
        mid = str(out.get("message_id"))
        m = meta_by_id.get(mid)
        if not m:
            continue
        cdir = m.get("combo_dir")
        combo_to_outputs.setdefault(cdir, []).append(out)
        combo_to_metas.setdefault(cdir, []).append(m)

    for cdir, outs in combo_to_outputs.items():
        metas = combo_to_metas.get(cdir, [])
        if metas:
            map_llm_responses(cdir, metas, outs)

    print(f"Appended {len(outputs)} responses to {responses_path} and updated histories for {len(combo_to_outputs)} combos.")


if __name__ == "__main__":
    main_infer()


