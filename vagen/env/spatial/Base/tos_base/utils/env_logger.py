import os
import json
from typing import Dict, List, Optional
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from .. import (
    HistoryManager,
    EvaluationTurnLog,
    Room,
    ExplorationTurnLog,
    CognitiveMapTurnLog,
    Agent
)
from .visualization.visualization import HTMLGenerator


@dataclass
class FBLog:
    """Log data for a single false belief exploration turn."""
    step: int
    room_state: Optional['Room'] = None
    agent_state: Optional['Agent'] = None
    
    # Final step info
    correctly_identified_changes: Optional[float] = None
    f1_overall: Optional[float] = None
    f1_position: Optional[float] = None
    f1_facing: Optional[float] = None
    action_cost: Optional[int] = None
    action_cost_after_seen: Optional[int] = None
    ground_truth_changes: Optional[List[Any]] = None # List[ChangedObject]
    reported_changes: Optional[List[Dict]] = None
    cogmap_log: Optional[Dict] = None
    newly_observed_changed_objects: List[str] = field(default_factory=list)
    newly_observed_unchanged_objects: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "step": self.step,
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {},
            "correctly_identified_changes": self.correctly_identified_changes,
            "f1_overall": self.f1_overall,
            "f1_position": self.f1_position,
            "f1_facing": self.f1_facing,
            "action_cost": self.action_cost,
            "action_cost_after_seen": self.action_cost_after_seen,
            "ground_truth_changes": [c.to_dict() for c in self.ground_truth_changes] if self.ground_truth_changes else [],
            "reported_changes": [c.to_dict() for c in self.reported_changes] if self.reported_changes else [],
            "cogmap_log": self.cogmap_log.to_dict() if hasattr(self.cogmap_log, 'to_dict') else (self.cogmap_log or {}),
            "newly_observed_changed_objects": self.newly_observed_changed_objects,
            "newly_observed_unchanged_objects": self.newly_observed_unchanged_objects,
        }

@dataclass
class EnvTurnLog:
    """Log data for a single environment turn."""
    turn_number: int
    user_message: str = ""  # Environment observation
    assistant_raw_message: str = ""  # Raw assistant input
    assistant_think_message: str = ""  # Think part of assistant message
    assistant_parsed_message: str = ""  # Parsed assistant action
    is_exploration_phase: bool = False
    is_last_exp: bool = False
    exploration_log: Optional["ExplorationTurnLog"] = None
    false_belief_log: Optional["FBLog"] = None
    evaluation_log: Optional["EvaluationTurnLog"] = None
    cogmap_log: Optional["CognitiveMapTurnLog"] = None
    room_state: Optional["Room"] = None
    agent_state: Optional["Agent"] = None
    room_image: Optional[str] = None
    message_images: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    
    def to_dict(self):
        return {
            "turn_number": self.turn_number,
            "user_message": self.user_message,
            "assistant_raw_message": self.assistant_raw_message,
            "assistant_think_message": self.assistant_think_message,
            "assistant_parsed_message": self.assistant_parsed_message,
            "is_exploration_phase": self.is_exploration_phase,
            "is_last_exp": self.is_last_exp,
            "exploration_log": self.exploration_log.to_dict() if self.exploration_log else {},
            "false_belief_log": self.false_belief_log.to_dict() if self.false_belief_log else {},
            "evaluation_log": self.evaluation_log.to_dict() if self.evaluation_log else {},
            "cogmap_log": self.cogmap_log.to_dict() if self.cogmap_log else {},
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {},
            "room_image": self.room_image,
            "message_images": self.message_images,
            "info": self.info
        }

class SpatialEnvLogger:
    """Logger for spatial environment data aggregation and visualization."""
    
    @staticmethod
    def _convert_omegaconf_to_python(obj):
        """Recursively convert OmegaConf objects to standard Python types for JSON serialization."""
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
        elif isinstance(obj, dict):
            return {key: SpatialEnvLogger._convert_omegaconf_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [SpatialEnvLogger._convert_omegaconf_to_python(item) for item in obj]
        else:
            return obj

    @staticmethod
    def _save_data(aggregated_data: Dict, output_dir: str, model_name: str):
        """Save aggregated data to JSON and generate HTML dashboard."""
        # Sort samples by run number before saving
        samples = aggregated_data.get('samples', {})
        def get_sample_number(sample_key: str) -> float:
            """Extract numeric part from sample key for sorting."""
            if sample_key.startswith('sample_run'):
                # Extract number from 'sample_run59' -> 59
                num_part = sample_key.replace('sample_run', '')
                return int(num_part) if num_part.isdigit() else float('inf')
            return float('inf')
        
        sorted_samples = dict(sorted(samples.items(), key=lambda x: get_sample_number(x[0])))
        
        saved_data = {
            'meta_info': {
                'model_name': model_name,
                'n_envs': len(samples),
            },
            **{k: v for k, v in aggregated_data.items() if k != 'samples'},
            'samples': sorted_samples,
        }

        # Convert OmegaConf objects to standard Python types
        saved_data = SpatialEnvLogger._convert_omegaconf_to_python(saved_data)
        
        # os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(os.path.join(output_dir, "env_data.json"), "w") as f:
            json.dump(saved_data, f, indent=2)

        # Generate HTML dashboard
        html_path = os.path.join(output_dir, "env_data.html")
        viz = HTMLGenerator(saved_data, html_path, True)
        dashboard_path = viz.generate_html()
        
        print(f"Environment data logged to {os.path.abspath(output_dir)}")
        print(f"Dashboard written to {os.path.abspath(dashboard_path)}")
        return output_dir
    

    @staticmethod
    def log_each_env_info(output_dir: str, model_name, save_images: bool = True):
        """Logs detailed information for each environment and overall performance metrics.

        New implementation that reads from directory structure:
        model_name/hash_value/vision_or_text/active_or_passive/

        Note: env_summaries and messages parameters are no longer used in this implementation,
        but kept for backwards compatibility.
        """

        # Use new directory-based aggregation instead of env_summaries
        # output_dir is results/debug, model_name from kwargs
        model_dir = HistoryManager.get_model_dir(output_dir, model_name)
        aggregated_data = HistoryManager.aggregate_from_directories(
            model_dir=model_dir,
            save_images=save_images,
        )

        return SpatialEnvLogger._save_data(aggregated_data, model_dir, model_name)