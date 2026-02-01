from typing import Optional, List, Dict
import os
import shutil
import json

from ..utils.cogmap.correlation import compute_correlation_metrics
from ..utils.utils import hash, get_model_name
from ..utils.room_utils import RoomPlotter
from ..evaluation.task_types import EvalTaskType
from .. import (
    Agent,
    Room,
    EvaluationManager,
    ExplorationManager,
    CognitiveMapManager,
)

# -------- Filenames (constants) --------
EXPLORATION_LOG_BASENAME = "exploration_turn_logs.json"
FALSE_BELIEF_LOG_BASENAME = "false_belief_turn_logs.json"
EVALUATION_LOG_BASENAME = "evaluation_turn_logs.json"
CONFIG_BASENAME = "config.json"
METRICS_BASENAME = "metrics.json"
IMAGES_DIRNAME = "images"
MESSAGES_BASENAME = "messages.json"
STATE_BASENAME = "history_state.json"
def get_evaluation_log_basename(eval_mode: str = "default") -> str:
    """Get evaluation log filename based on eval_mode."""
    if eval_mode == "default":
        return "evaluation_turn_logs.json"
    return f"evaluation_turn_logs_{eval_mode}.json"

def _discover_eval_modes(combo_path: str) -> List[str]:
    modes: List[str] = []
    default_path = os.path.join(combo_path, EVALUATION_LOG_BASENAME)
    if os.path.exists(default_path):
        modes.append("default")
    for name in os.listdir(combo_path):
        if not name.startswith("evaluation_turn_logs_") or not name.endswith(".json"):
            continue
        mode = name[len("evaluation_turn_logs_"):-len(".json")]
        if mode and mode not in modes:
            modes.append(mode)
    return sorted(modes)
class HistoryManager:
    """Simple conversation history manager, one history manager for one run.
    Store only env turn logs in a single JSON file
    Directory structure: model_name/room_hash_key/vision_or_text/active_or_passive/
    Example: gpt-4o/1d54fa/vision/active/
    """

    def __init__(self, observation_config: Dict, model_config: Dict , room_dict: Dict, agent_dict: Dict, output_dir:str, seed: int,
                 image_dir:str = None, eval_override: bool = False, all_override: bool = False, false_belief_override: bool = False, all_tasks: List = None, eval_mode: str = "default"):
        # only explore turn logs are saved
        self.exploration_turn_logs: List[Dict] = []
        self.false_belief_turn_logs: List[Dict] = []
        self.evaluation_turn_logs: Dict[str, Dict[str, Dict]] = {}
        self.messages: List[Dict] = []
        self.seed: int  = seed
        self.exp_type = observation_config['exp_type']
        self.enable_think = bool(((observation_config or {}).get('prompt_config') or {}).get('enable_think', False))
        self.eval_mode = eval_mode
        self.model_path = HistoryManager.get_model_dir(output_dir, model_config['model_name'])
        self.output_dir = os.path.abspath(os.path.join(
            self.model_path, self._generate_room_key(room_dict, agent_dict),
            observation_config['render_mode'],
            observation_config['exp_type'],
            "think" if observation_config['prompt_config']["enable_think"] else "nothink",
        ))
        self.room_dict = room_dict
        self.agent_dict = agent_dict
        # Agent initial pose must exist; it must NOT be derived from possibly-final pos/ori.
        if "init_pos" not in self.agent_dict or "init_ori" not in self.agent_dict:
            raise ValueError("agent_dict must include init_pos/init_ori (initial state).")
        self.image_dir = image_dir
        self.observation_config = observation_config
        if observation_config['exp_type'] == 'passive':
            self.output_dir = os.path.join(self.output_dir, observation_config["proxy_agent"])
        self.model_config_path = os.path.join(self.model_path, CONFIG_BASENAME)
        self.exploration_path = os.path.join(self.output_dir, EXPLORATION_LOG_BASENAME)
        self.false_belief_path = os.path.join(self.output_dir, FALSE_BELIEF_LOG_BASENAME)
        self.evaluation_path = os.path.join(self.output_dir, get_evaluation_log_basename(self.eval_mode))
        self.metrics_path = os.path.join(self.output_dir, METRICS_BASENAME)
        self.messages_path = os.path.join(self.output_dir, MESSAGES_BASENAME)
        self.state_path = os.path.join(self.output_dir, STATE_BASENAME)

        # Apply granular overrides
        if all_override:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)

        self._load()
        os.makedirs(self.output_dir, exist_ok=True)
        if eval_override:
            task_map = EvalTaskType.get_task_map()
            for task_type in all_tasks:
                mapped_task = task_map.get(task_type).__name__
                if mapped_task in self.evaluation_turn_logs:
                    self.evaluation_turn_logs[mapped_task] = {}
            self.save_evaluation()
        if false_belief_override:
            if os.path.exists(self.false_belief_path):
                os.remove(self.false_belief_path)
            self.false_belief_turn_logs = []
            
        os.makedirs(os.path.join(self.output_dir, IMAGES_DIRNAME), exist_ok=True)
        if not os.path.exists(self.model_config_path):
            with open(self.model_config_path, "w") as f:
                json.dump(model_config, f, ensure_ascii=False, indent=2)

    def replace_paths(self, old_base: str, new_base: str) -> None:
        def _recursive_replace(obj):
            if isinstance(obj, str):
                return obj.replace(old_base, new_base)
            if isinstance(obj, list):
                return [_recursive_replace(item) for item in obj]
            if isinstance(obj, dict):
                return {k: _recursive_replace(v) for k, v in obj.items()}
            return obj
            
        self.exploration_turn_logs = _recursive_replace(self.exploration_turn_logs)
        self.false_belief_turn_logs = _recursive_replace(self.false_belief_turn_logs)
        self.evaluation_turn_logs = _recursive_replace(self.evaluation_turn_logs)
        self.messages = _recursive_replace(self.messages)

    def has_exploration(self, index):
        return 0 <= index < len(self.exploration_turn_logs)

    def has_false_belief(self, index):
        return 0 <= index < len(self.false_belief_turn_logs)

    def _generate_room_key(self, room_dict, agent_dict):
        # Do NOT mutate the input dict: we still need pos/ori for saving state.
        a = dict(agent_dict or {})
        a.pop("pos", None)
        a.pop("ori", None)
        room_str = json.dumps({**room_dict, **a}, sort_keys=True)
        return hash(room_str)
        
    def _load(self):
        """Load env turn logs from JSON file"""
        if os.path.exists(self.exploration_path):
            with open(self.exploration_path, "r") as f:
                self.exploration_turn_logs = json.load(f)
        if os.path.exists(self.false_belief_path):
            with open(self.false_belief_path, "r") as f:
                self.false_belief_turn_logs = json.load(f)
        if os.path.exists(self.evaluation_path):
            with open(self.evaluation_path, "r") as f:
                self.evaluation_turn_logs = json.load(f)
            self.evaluation_turn_logs = self._migrate_task_names(self.evaluation_turn_logs)

        if os.path.exists(self.messages_path):
            with open(self.messages_path, "r") as f:
                self.messages = json.load(f)
                
    @staticmethod
    def _migrate_task_names(logs: Dict[str, Dict]) -> Dict[str, Dict]:
        """Migrate old task names to new ones using EvalTaskType."""
        new_logs = {}
        for task_name, content in logs.items():
            new_name = EvalTaskType.migrate_legacy_name(task_name)
            if new_name in new_logs:
                new_logs[new_name].update(content)
            else:
                new_logs[new_name] = content
        return new_logs

    def save_exploration(self) -> None:
        """Save env turn logs to JSON file"""
        if self.exploration_turn_logs:
            with open(self.exploration_path, "w") as f:
                json.dump(self.exploration_turn_logs, f, ensure_ascii=False, indent=2)
    def save_false_belief(self) -> None:
        """Save false belief turn logs to JSON file"""
        if self.false_belief_turn_logs:
            with open(self.false_belief_path, "w") as f:
                json.dump(self.false_belief_turn_logs, f, ensure_ascii=False, indent=2)
    def save_evaluation(self) -> None:
        """Save evaluation turn logs to JSON file"""
        with open(self.evaluation_path, "w") as f:
            json.dump(self.evaluation_turn_logs, f, ensure_ascii=False, indent=2)

    def save(self) -> None:
        """Save env turn logs to JSON file"""
        self.save_exploration()
        self.save_false_belief()
        self.save_evaluation()
        # Also compute and save metrics for this sample
        metrics = self._compute_sample_metrics()
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        self.save_state()

    # -------- Messages (conversation list) --------
    def _last_conv_role(self) -> Optional[str]:
        for m in reversed(self.messages):
            r = m.get('role')
            if r in ("system", "user", "assistant"):
                return r
        return None

    def _expected_next_role(self) -> str:
        last = self._last_conv_role()
        if last is None:
            return "system"
        if last == "system":
            return "user"
        if last == "user":
            return "assistant"
        return "user"

    def _check_message_order(self, role: str) -> None:
        if role in ("system", "user", "assistant"):
            expected = self._expected_next_role()
            assert role == expected, f"Message order violation: expected {expected}, got {role}"

    def init_messages(self, system_prompt: str) -> None:
        self.messages = [{"role": "system", "content": system_prompt}]

    def append_assistant_message(self, assistant_raw: str) -> None:
        self._check_message_order("assistant")
        self.messages.append({"role": "assistant", "content": assistant_raw})

    def append_env_feedback(self, obs_str: str, image_paths: List[str]) -> None:
        self._check_message_order("user")
        entry: Dict = {"role": "user", "content": obs_str}
        if image_paths:
            entry["images"] = list(image_paths)
        self.messages.append(entry)

    def save_messages(self, agent_loc = None) -> None:
        with open(self.messages_path, "w") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        self.save_state(agent_loc)


    
    def _attach_room_image(self, turn_log: Dict, filename: str) -> None:
        if turn_log['room_state'] and turn_log['agent_state']:
            img_path = os.path.join(self.output_dir, IMAGES_DIRNAME, filename)
            if not os.path.exists(img_path):
                RoomPlotter.plot(
                    Room.from_dict(turn_log['room_state']),
                    Agent.from_dict(turn_log['agent_state']),
                    mode='img', save_path=img_path,
                )
            turn_log['room_image'] = os.path.relpath(img_path, self.model_path)

    def update_exp_turn_log(self, turn_log: Dict, replay: bool = False) -> None:
        assert turn_log['is_exploration_phase']
        turn_idx = turn_log['turn_number'] - 1
        self._attach_room_image(turn_log, f"room_turn_{turn_log['turn_number']}.png") # fix the bug caused by cogmap override
        
        if replay:
            # Override mode: replace existing turn log at the given index
            assert 0 <= turn_idx < len(self.exploration_turn_logs), f"Invalid turn index {turn_idx} for replay (max: {len(self.exploration_turn_logs)-1}) for output directory {self.output_dir}"
            turn_log['cogmap_log'] = self.exploration_turn_logs[turn_idx].get('cogmap_log')
            self.exploration_turn_logs[turn_idx] = turn_log
        else:
            # Append mode: add new turn log
            assert not self.has_exploration(turn_idx), f"Turn {turn_log['turn_number']} already exists"
            self.exploration_turn_logs.append(turn_log)

    def update_false_belief_turn_log(self, turn_log: Dict, replay: bool = False) -> None:
        assert not turn_log['is_exploration_phase']
        assert turn_log.get('false_belief_log')
        self._attach_room_image(turn_log, f"room_false_belief_{turn_log['turn_number']}.png")
        fb_step = (turn_log.get('false_belief_log') or {}).get('step')
        idx = int(fb_step) - 1 if isinstance(fb_step, (int, float)) else None
        if replay:
            assert idx is not None and 0 <= idx < len(self.false_belief_turn_logs), (
                f"Invalid false belief step {fb_step} for replay (max: {len(self.false_belief_turn_logs)}), output directory: {self.output_dir}"
            )
            turn_log['false_belief_log']['cogmap_log'] = (self.false_belief_turn_logs[idx].get('false_belief_log') or {}).get('cogmap_log')
            self.false_belief_turn_logs[idx] = turn_log
        else:
            assert idx is not None and idx >= len(self.false_belief_turn_logs), (
                f"False belief step {fb_step} already exists for output directory {self.output_dir}"
            )
            self.false_belief_turn_logs.append(turn_log)

    def update_eval_turn_log(self, turn_log: Dict) -> None:
        assert not turn_log['is_exploration_phase']
        assert turn_log['evaluation_log']
        assert turn_log['room_state'] and turn_log['agent_state']

        task_type = turn_log['evaluation_log']['task_type']
        question_id = turn_log['evaluation_log']['evaluation_data']['id']

        self._attach_room_image(turn_log, f"room_{task_type}_{question_id}.png")

        if task_type not in self.evaluation_turn_logs:
            self.evaluation_turn_logs[task_type] = {}
        self.evaluation_turn_logs[task_type][question_id] = turn_log

    def update_turn_log(self, turn_log: Dict, replay: bool = False) -> None:
        """Dispatch to specific update functions (kept for compatibility)."""
        if turn_log.get('false_belief_log'):
            self.update_false_belief_turn_log(turn_log, replay=replay)
        elif turn_log['is_exploration_phase']:
            self.update_exp_turn_log(turn_log, replay=replay)
        else:
            self.update_eval_turn_log(turn_log)

    def get_responses(self) -> List[Dict]:
        msgs = [log.get('assistant_raw_message') for log in self.exploration_turn_logs if log.get('assistant_raw_message') is not None]
        if self.false_belief_turn_logs:
            msgs.extend([log.get('assistant_raw_message') for log in self.false_belief_turn_logs if log.get('assistant_raw_message') is not None])
        return msgs


    def update_cogmap(self, turn_log: Dict) -> None:
        """Update cognitive map response for a specific turn"""
        if turn_log['is_exploration_phase']:
            turn_idx = turn_log['turn_number'] - 1
            assert 0 <= turn_idx < len(self.exploration_turn_logs)
            self.exploration_turn_logs[turn_idx]['cogmap_log'] = turn_log['cogmap_log']
        else:
            assert turn_log['evaluation_log']
            assert self.exp_type == 'active'
            task_type = turn_log['evaluation_log']['task_type']
            question_id = turn_log['evaluation_log']['evaluation_data']['id']

            assert task_type in self.evaluation_turn_logs
            assert question_id in self.evaluation_turn_logs[task_type]

            self.evaluation_turn_logs[task_type][question_id]['cogmap_log'] = turn_log['cogmap_log']

    def get_cogmap(self, turn_idx) -> Optional[Dict]:
        """Get cognitive map response for a specific turn (0-indexed)"""
        if not (0 <= turn_idx < len(self.exploration_turn_logs)):
            return None
        return self.exploration_turn_logs[turn_idx].get('cogmap_log')

    def has_question(self, question_id: str) -> bool:
        """Check if a question with the given ID already exists in evaluation logs"""
        return any(question_id in questions for questions in self.evaluation_turn_logs.values())

    def get_eval_ids(self) -> Dict[str, int]:
        """Return list of completed eval question IDs per task class name."""
        return {task_type: [question["evaluation_log"]["evaluation_data"]['id'] for question in questions.values()] for task_type, questions in self.evaluation_turn_logs.items()}

    def get_eval_counts(self) -> Dict[str, int]:
        """Return number of completed eval questions per task class name."""
        return {task_type: len(questions or {}) for task_type, questions in self.evaluation_turn_logs.items()}

    def get_existing_question_ids(self, task_type: str) -> List[str]:
        """Return list of existing question ids for a given task class name."""
        return list((self.evaluation_turn_logs.get(task_type) or {}).keys())
    
    @staticmethod
    def get_model_dir(output_dir: str, model_name: str) -> str:
        """Generate a unique directory name for the model configuration"""
        model_name = get_model_name(model_name)
        return os.path.join(output_dir, model_name)
    
    @staticmethod
    def aggregate_from_directories(model_dir: str, save_images: bool = True) -> Dict:
        """
        Aggregate data from new directory structure:
        base_dir/model_name/hash_value/vision_or_text/active_or_passive/

        Returns:
            Aggregated data dictionary with config_groups organized by text/vision + active/passive combinations
        """
        assert os.path.exists(model_dir), f"Model directory {model_dir} does not exist"

        samples = {}
        all_config_keys = set()

        # Scan all sample directories (room keys)
        sample_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

        # Sequential numbering only for valid samples (with valid subdirs)
        sample_counter = 0
        for sample_dir in sample_dirs:
            sample_path = os.path.join(model_dir, sample_dir)

            # Collect subdirectories containing log files
            subdirs: List[str] = []
            for root, _, files in os.walk(sample_path):
                if EXPLORATION_LOG_BASENAME in files or EVALUATION_LOG_BASENAME in files or FALSE_BELIEF_LOG_BASENAME in files:
                    subdirs.append(root)

            # Skip if subdirs is empty (invalid sample)
            if not subdirs:
                continue
            with open(os.path.join(subdirs[0], STATE_BASENAME), 'r') as f:
                sample_cfg = json.load(f)
            if sample_cfg.get("image_dir") is None:
                sample_key = f"sample_{sample_counter}"
                sample_counter += 1
            else:
                sample_key = f"sample_{os.path.basename(sample_cfg['image_dir'])}"
            assert sample_key not in samples, f"Duplicate sample key {sample_key}"
            samples[sample_key] = {}

            for combo_path in subdirs:
                rel = os.path.relpath(combo_path, sample_path)
                if rel in (".", ""):
                    continue
                config_key = rel.replace(os.sep, "_")

                sample_data = HistoryManager._load_sample_data(
                    combo_path=combo_path,
                    sample_key=sample_key,
                    save_images=save_images,
                    model_dir=model_dir,
                )
                if sample_data:
                    # Store the hash directory name in sample_data for use in visualization
                    sample_data["config"]["_hash_dir"] = sample_dir
                    # Store the full sample subdirectory path for plots
                    sample_data["config"]["_sample_subdir"] = os.path.relpath(combo_path, model_dir)
                    samples[sample_key][config_key] = sample_data
                    all_config_keys.add(config_key)

        # Initialize result structure with samples
        result = {
            "samples": samples,
            "exp_summary": {"group_performance": {}},
            "eval_summary": {"group_performance": {}, "group_performance_by_mode": {}},
            "cogmap_summary": {"group_performance": {}},
            "correlation": {"group_performance": {}},
        }

        # Aggregate performance for each config combination across all samples
        for config_name in sorted(all_config_keys):
            env_data_list = []
            for sample_data_dict in samples.values(): # config_key -> sample_data
                if config_name in sample_data_dict and sample_data_dict[config_name] is not None:
                    env_data_list.append(sample_data_dict[config_name])

            if env_data_list:
                result["exp_summary"]["group_performance"][config_name] = ExplorationManager.aggregate_group_performance(env_data_list)
                result["eval_summary"]["group_performance"][config_name] = EvaluationManager.aggregate_group_performance(env_data_list)
                modes: List[str] = []
                for entry in env_data_list:
                    if entry.get("evaluation_tasks"):
                        modes.append("default")
                    for k in entry.keys():
                        if k.startswith("evaluation_tasks_") and entry.get(k):
                            modes.append(k.split("evaluation_tasks_", 1)[1] or "default")
                for mode in sorted(set(modes)):
                    mode_env = []
                    for entry in env_data_list:
                        tasks = entry.get("evaluation_tasks") if mode == "default" else entry.get(f"evaluation_tasks_{mode}")
                        if tasks:
                            mode_env.append({"evaluation_tasks": tasks})
                    if mode_env:
                        result["eval_summary"]["group_performance_by_mode"].setdefault(mode, {})[config_name] = (
                            EvaluationManager.aggregate_group_performance(mode_env)
                        )
                # Provide both exploration and evaluation cogmap summaries
                exp_type = "active" if "active" in config_name else "passive"
                result["cogmap_summary"]["group_performance"][config_name] = CognitiveMapManager.aggregate_group_performance(env_data_list, exp_type=exp_type)
                result["correlation"]["group_performance"][config_name] = compute_correlation_metrics(env_data_list, exp_type=exp_type)
        return result

    @staticmethod
    def _load_sample_data(combo_path: str, sample_key: str, save_images: bool, model_dir: str) -> Optional[Dict]:
        """Load data from a single sample's combination directory"""
        exploration_file = os.path.join(combo_path, EXPLORATION_LOG_BASENAME)
        false_belief_file = os.path.join(combo_path, FALSE_BELIEF_LOG_BASENAME)
        config_file = os.path.join(combo_path, STATE_BASENAME)
        metrics_file = os.path.join(combo_path, METRICS_BASENAME)

        sample_data = {
            "sample_id": sample_key,
            "env_turn_logs": [],  # Only exploration turn logs
            "false_belief_turn_logs": [],
            "evaluation_tasks": {},  # Separate storage for evaluation tasks
            "config": {},
            "metrics": {},
        }

        # Load exploration turn logs
        if os.path.exists(exploration_file):
            with open(exploration_file, 'r') as f:
                exploration_logs = json.load(f)
            sample_data["env_turn_logs"] = exploration_logs if exploration_logs else [] # Only exploration logs

        # Load false belief turn logs
        if os.path.exists(false_belief_file):
            with open(false_belief_file, 'r') as f:
                false_belief_logs = json.load(f)
            sample_data["false_belief_turn_logs"] = false_belief_logs if false_belief_logs else []

        # Load evaluation turn logs - store each task separately
        # Try loading all possible eval_mode files
        for eval_mode in _discover_eval_modes(combo_path):
            eval_file = os.path.join(combo_path, get_evaluation_log_basename(eval_mode))
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    evaluation_logs = json.load(f)
                # Migrate task names
                evaluation_logs = HistoryManager._migrate_task_names(evaluation_logs)
                # Store each evaluation task separately with mode suffix if not default
                if eval_mode == "default":
                    sample_data["evaluation_tasks"] = evaluation_logs if evaluation_logs else {}
                else:
                    sample_data[f"evaluation_tasks_{eval_mode}"] = evaluation_logs if evaluation_logs else {}
        
        # Keep backward compatibility: if default mode exists, use it as primary
        if "evaluation_tasks" not in sample_data:
            sample_data["evaluation_tasks"] = {}

        # Load config and metrics if present
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                sample_data["config"] = json.load(f)
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                sample_data["metrics"] = json.load(f)

        # Ensure key metrics are derived from logs (incl. false-belief stats).
        env_data = {
            "env_turn_logs": sample_data.get("env_turn_logs") or [],
            "false_belief_turn_logs": sample_data.get("false_belief_turn_logs") or [],
            "evaluation_tasks": sample_data.get("evaluation_tasks") or {},
        }
        if not isinstance(sample_data.get("metrics"), dict):
            sample_data["metrics"] = {}
        try:
            sample_data["metrics"]["exploration"] = ExplorationManager.aggregate_per_sample(env_data)
        except Exception:
            pass
        try:
            sample_data["metrics"]["evaluation"] = EvaluationManager.aggregate_per_sample(env_data)
            for k, v in sample_data.items():
                if k.startswith("evaluation_tasks_"):
                    mode = k.split("evaluation_tasks_", 1)[1] or "default"
                    mode_metrics = EvaluationManager.aggregate_per_sample({"evaluation_tasks": v})
                    sample_data["metrics"][f"evaluation_{mode}"] = mode_metrics
        except Exception:
            pass
        try:
            exp_type = ((sample_data.get("config") or {}).get("observation_config") or {}).get("exp_type")
            sample_data["metrics"]["cogmap"] = CognitiveMapManager.aggregate_per_sample(env_data, exp_type=exp_type)
        except Exception:
            pass

        # Process image paths if save_images is enabled
        if save_images:
            def _process_cogmap_imgs(cogmap_log: dict) -> None:
                """Process message_images inside cogmap_log."""
                for map_data in cogmap_log.values():
                    if isinstance(map_data, dict) and map_data.get('message_images'):
                        map_data['message_images'] = [os.path.relpath(p, model_dir) for p in map_data['message_images']]

            # Process exploration turn logs
            for turn_log in sample_data["env_turn_logs"]:
                if turn_log.get('message_images'):
                    turn_log['message_images'] = [os.path.relpath(img_path, model_dir) for img_path in turn_log['message_images']]
                if turn_log.get('cogmap_log'):
                    _process_cogmap_imgs(turn_log['cogmap_log'])

            # Process false belief turn logs
            for turn_log in sample_data["false_belief_turn_logs"]:
                if turn_log.get('message_images'):
                    turn_log['message_images'] = [os.path.relpath(img_path, model_dir) for img_path in turn_log['message_images']]
                fb_log = turn_log.get('false_belief_log') or {}
                if fb_log.get('cogmap_log'):
                    _process_cogmap_imgs(fb_log['cogmap_log'])


            # Process evaluation tasks (all modes)
            eval_sets = {"evaluation_tasks": sample_data.get("evaluation_tasks") or {}}
            for k, v in sample_data.items():
                if k.startswith("evaluation_tasks_") and v:
                    eval_sets[k] = v
            for task_set in eval_sets.values():
                for task in task_set.values():
                    for question_data in task.values():
                        if question_data.get('message_images'):
                            question_data['message_images'] = [os.path.relpath(img_path, model_dir) for img_path in question_data['message_images']]

        return sample_data if sample_data["env_turn_logs"] or sample_data["evaluation_tasks"] or sample_data["false_belief_turn_logs"] else None        

    def _compute_sample_metrics(self) -> Dict:
        env_data = {
            "env_turn_logs": self.exploration_turn_logs,
            "false_belief_turn_logs": self.false_belief_turn_logs,
            "evaluation_tasks": self.evaluation_turn_logs,
        }
        return {
            "exploration": ExplorationManager.aggregate_per_sample(env_data),
            "evaluation": EvaluationManager.aggregate_per_sample(env_data),
            "cogmap": CognitiveMapManager.aggregate_per_sample(env_data, exp_type=self.exp_type),
        }

    # -------- Simple persist/restore for reuse --------
    def save_state(self, agent_loc = None) -> None:
        """Persist minimal state to reload this HistoryManager later without reconstruction."""
        if  agent_loc is not None:
            self.agent_dict['pos'] = agent_loc[0]
            self.agent_dict['ori'] = agent_loc[1]
        state = {
            "observation_config": {
                "render_mode": self.observation_config['render_mode'],
                "exp_type": self.exp_type,
                "prompt_config": {"enable_think": bool(self.enable_think)},
                "proxy_agent": self.observation_config['proxy_agent'] if self.exp_type == 'passive' else None,
            },
            "model_config": json.load(open(self.model_config_path)) if os.path.exists(self.model_config_path) else {},
            "room_dict": self.room_dict,
            "agent_dict": self.agent_dict,
            "image_dir": self.image_dir,
            "seed": self.seed,
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_from_dir(combo_dir: str, eval_override: bool, all_tasks: List = None, image_dir: str = None, eval_mode: str = "default") -> "HistoryManager":
        """Load a HistoryManager using state saved in combo_dir/history_state.json."""
        combo_dir = os.path.abspath(combo_dir)
        state_file = os.path.join(combo_dir, STATE_BASENAME)
        assert os.path.exists(state_file), f"Missing state file: {state_file}"
        with open(state_file, "r") as f:
            s = json.load(f)
        model_name = get_model_name(s.get("model_config", {}).get("model_name", ""))
        output_dir = combo_dir.split(model_name)[0]
        assert model_name and model_name not in output_dir, f"Failed to infer output_dir from combo_dir: {combo_dir}"
        
        saved_image_dir = s.get("image_dir")
        final_image_dir = saved_image_dir
        if image_dir is not None and saved_image_dir:
            final_image_dir = os.path.join(image_dir, os.path.basename(saved_image_dir.rstrip(os.sep)))
        
        hm = HistoryManager(
            observation_config=s.get("observation_config", {}),
            model_config=s.get("model_config", {}),
            room_dict=s.get("room_dict", {}),
            agent_dict=s.get("agent_dict", {}),
            output_dir=output_dir,
            seed=s.get("seed", 0),
            image_dir=final_image_dir,
            eval_override=eval_override,
            false_belief_override=False,
            all_tasks=all_tasks,
            eval_mode=eval_mode,
        )
        if final_image_dir and saved_image_dir and final_image_dir != saved_image_dir:
            old_base = os.path.dirname(saved_image_dir)
            new_base = os.path.dirname(final_image_dir)
            if old_base != new_base:
                hm.replace_paths(old_base, new_base)
        return hm


    # -------- Accessors for builder/inference --------
    def get_enable_think(self) -> bool:
        return bool(self.enable_think)
