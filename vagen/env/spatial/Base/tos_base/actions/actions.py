from typing import Optional, List, Dict, Any
import re
import numpy as np

from .base import BaseAction, ActionResult
from ..core.object import Gate
from ..core.relationship import PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship, RelationTriple, OrientationRel, DegreeRel
from vagen.env.spatial.Base.tos_base.utils.room_modifier import ChangedObject

"""
Specific action implementations for spatial exploration.
Contains all concrete action classes and the ActionSequence parser.
"""


from ..utils.utils import ANSWER_LABEL

ACTION_INSTRUCTION = """\
You can jump to objects within and across rooms, turn, and observe.
When you are at a door, you can see objects from both connected rooms (within FOV).

Available Actions:
{actions}

Action Grammar (HARD CONSTRAINT):
Actions: [ <M>* <F> ]
<M> = "JumpTo(OBJ)" | "Rotate(DEG)"
<F> = "Observe()" | "Query(OBJ)" | "Term()"
Constraints:
- Zero, one or more <M>. No JumpTo at first step.
- Exactly one <F>, and it must be the final action.
- No more than one Observe().
- Term() may appear only alone.
- Any violation is invalid.

Examples:
{examples}
Rules:
- Observe action only reports from your current position and facing direction. If you jump multiple times, the final Observe() action gives the view only from your last position.
- Actions execute in order. Field of view: {field_of_view}°.

Observe and Query action have costs:
{costs}
"""



class MoveAction(BaseAction):
    """Jump to a target object"""
    
    format_desc = "JumpTo(OBJ)"
    description = (
        "Jump to the same position as the object or door. "
        "Your orientation does NOT change. "
        "The object you jump to MUST be in your field of view and previously observed. Use object or door names only. NO numbers or directions or others. "
        "Invalid: JumpTo(left), JumpTo(1)."
    )
    example = "JumpTo(table)"
    format_pattern = r"^JumpTo\(([A-Za-z0-9_ -]+)\)$"
    cost = 0
    
    def __init__(self, target: str):
        super().__init__(target)
        self.target = target.replace('_', ' ')
    
    def success_message(self, **kwargs) -> str:
        extra = kwargs.get('extra')
        return f"You jumped to {self.target}." + (f" {extra}" if extra else "")
    
    def error_message(self, error_type: str) -> str:
        errors = {"not_found": "object not found", "not_visible": "object not visible", "not_observed": "object not observed yet"}
        return f"Cannot jump to '{self.target}': {errors.get(error_type, 'execution failed')}."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute jump action on room state."""
        if not room.has_object(self.target):
            return ActionResult(False, self.get_feedback(False, "not_found"), str(self), 'move', {'target_name': self.target})
        
        target_obj = room.get_object_by_name(self.target)
        observed_items = set(kwargs.get('observed_items', [])) # if None, all objects are observed
        if observed_items is not None and self.target not in observed_items:
            return ActionResult(False, self.get_feedback(False, "not_observed"), str(self), 'move', {'target_name': self.target})
        if not kwargs.get('move_anyway', False) and not self._is_visible(agent, target_obj):
            return ActionResult(False, self.get_feedback(False, "not_visible"), str(self), 'move', {'target_name': self.target})        
        
        # apply move and room membership
        agent.pos, agent.room_id = target_obj.pos, target_obj.room_id
        
        return ActionResult(True, self.get_feedback(True), str(self), 'move', {'target_name': self.target})
    
    def __repr__(self):
        return f"JumpTo({self.target})"

class RotateAction(BaseAction):
    """Rotate by specified degrees"""
    
    format_desc = "Rotate(DEG)"
    description = ("Rotate relative to your current orientation. "
                   "Positive = clockwise, negative = counterclockwise. "
                   "Valid: -270, -180, -90, 0, 90, 180, 270. "
                   "You must rotate by these specified degrees; otherwise your action will be invalid.")
    example = "Rotate(-90)"
    format_pattern = r"^Rotate\(([0-9-]+)\)$"
    VALID_DEGREES = [0, 90, 180, 270, -90, -180, -270]
    
    def __init__(self, degrees: int):
        super().__init__(degrees)
        self.degrees = int(degrees)
        
    def success_message(self, **kwargs) -> str:
        if self.degrees == 0:
            return "You rotated 0°."
        direction = 'clockwise' if self.degrees > 0 else 'counterclockwise'
        return f"You rotated {direction} {abs(self.degrees)}°."
    
    def error_message(self, error_type: str) -> str:
        if error_type == "invalid_degree":
            return f"Cannot rotate by {self.degrees}°: only {self.VALID_DEGREES} allowed."
        return f"Cannot rotate by {self.degrees}°: execution failed."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute rotate action on room state."""
        if self.degrees is None or self.degrees not in self.VALID_DEGREES:
            return ActionResult(False, self.get_feedback(False, "invalid_degree"), str(self), 'rotate', {'degrees': self.degrees})
        agent.ori = agent.ori @ self._get_rotation_matrix(self.degrees)  
        return ActionResult(True, self.get_feedback(True), str(self), 'rotate', {'degrees': self.degrees})
    
    def __repr__(self):
        return f"Rotate({self.degrees})"


class ObserveBase(BaseAction):
    """Base observe implementation (internal)."""
    
    format_desc = "Observe()"
    description = (
        "Observe spatial relationships of all objects in the field of view relative to your current position. "
        "You can only observe objects that are within your field of view."
    )
    example = "Observe()"
    format_pattern = r"^Observe\(\)$"
    cost = 1
    
    def __init__(self):
        super().__init__()
    
    
    def _collect_obj_observations(self, agent, visible_objects, anchor_name: str, discrete: bool = False):
        relationships: List[str] = []
        relation_triples: List[RelationTriple] = []
        for obj in visible_objects:
            if discrete:
                rel = PairwiseRelationshipDiscrete.relationship(tuple(obj.pos), tuple(agent.pos), anchor_ori=tuple(agent.ori))
            else:
                rel = PairwiseRelationship.relationship(tuple(obj.pos), tuple(agent.pos), anchor_ori=tuple(agent.ori), full=True)
            pairwise_str = rel.to_string()

            if hasattr(obj, 'has_orientation') and not obj.has_orientation:
                answer_str = f"{obj.name}: {pairwise_str}"
            else:
                if isinstance(obj, Gate):
                    rid = agent.room_id
                    if isinstance(rid, (list, tuple)):
                        rid = list(set(agent.room_id) & set(obj.room_id))
                        assert len(rid) == 1, f"intersection of room ids is not unique: {rid}"
                        rid = rid[0]
                    gate_ori = obj.get_ori_for_room(int(rid)) if rid is not None else obj.ori
                    ori_pair = OrientationRel.get_relative_orientation(tuple(gate_ori), tuple(agent.ori))
                    ori_str = OrientationRel.to_string(ori_pair, 'ego', 'orientation', if_gate=True)
                else:
                    ori_pair = OrientationRel.get_relative_orientation(tuple(obj.ori), tuple(agent.ori))
                    ori_str = OrientationRel.to_string(ori_pair, 'ego', 'orientation')
                answer_str = f"{obj.name}: {pairwise_str}, {ori_str}"
            relationships.append(answer_str)
            relation_triples.append(RelationTriple(subject=obj.name, anchor=anchor_name, relation=rel, orientation=tuple(agent.ori)))
        final_answer = "\n" + "\n".join(f"• {rel}" for rel in relationships)
        return final_answer, relationships, relation_triples
    
    def success_message(self, **kwargs) -> str:
        return f"You observe: {kwargs.get('answer', 'nothing')}."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot observe: execution failed."

    

class ObserveAction(ObserveBase):
    """Observe with approximate relations and local (near) pair descriptions"""
    format_desc = "Observe()"
    description = ("Report objects (including doors) and their spatial relationships from your current position in your FOV. "
                   "Also reports relations between mutually close objects in your FOV, using your current facing direction as north (a relative reference frame, not true north)."
                   "Use exactly one Observe() per step and make it the last action. "
                   "Never call Term() after Observe().")
    example = "Observe()"
    format_pattern = r"^Observe\(\)$"
    cost = 1
    @staticmethod
    def is_final() -> bool:
        return True

    def __repr__(self):
        return "Observe()"


    
    def _collect_local_relationships(self, agent, visible_objects, anchor_name: str):
        # proximity-based pair relations using discrete relationship binning
        relationships, relation_triples = [], []
        n = len(visible_objects)
        for i in range(n):
            for j in range(i + 1, n):
                a_obj, b_obj = visible_objects[i], visible_objects[j]
                # NOTE always use agent's orientation for orientation
                prox_rel = ProximityRelationship.from_positions(tuple(a_obj.pos), tuple(b_obj.pos), tuple(agent.ori))
                if prox_rel is not None:
                    relationships.append(prox_rel.to_string(a_obj.name, b_obj.name))
                    relation_triples.append(RelationTriple(subject=a_obj.name, anchor=b_obj.name, relation=prox_rel, orientation=tuple(agent.ori)))
        final_answer = "\n".join(f"• {rel}" for rel in relationships)
        return final_answer, relationships, relation_triples

    def execute(self, room, agent, **kwargs) -> ActionResult:
        neglect_objects = kwargs.get('neglect_objects', []) + [obj.name for obj in room.all_objects if np.allclose(obj.pos, agent.pos)]
        visible_objects = [obj for obj in room.all_objects if self._is_visible(agent, obj) and obj.name not in neglect_objects]
        if not visible_objects:
            answer = "No objects in field of view."
            return ActionResult(True, self.get_feedback(True, answer=answer), str(self), 'observe', {
                'answer': answer, 'visible_objects': [], 'relationships': [], 'local_relationships': []
            })

        anchor_name = self.get_anchor_name(room, agent) if not kwargs.get('free_position', False) else 'free_position'
        pairwise_answer, relationships, pairwise_relation_triples = self._collect_obj_observations(agent=agent, visible_objects=visible_objects, anchor_name=anchor_name, discrete=True)
        local_answer, local_relationships, local_relation_triples = self._collect_local_relationships(agent, visible_objects, anchor_name)

        final_answer = pairwise_answer
        if local_answer:
            final_answer += (
                f"\nAssume your current facing direction is called \"north\":\n{local_answer}"
            )
        return ActionResult(True, self.get_feedback(True, answer=final_answer), str(self), 'observe', {
            'answer': final_answer,
            'visible_objects': [obj.name for obj in visible_objects],
            'relationships': relationships,
            'local_relationships': local_relationships,
            'relation_triples': pairwise_relation_triples + local_relation_triples
        })

class TermAction(BaseAction):
    """Terminate exploration"""
    
    format_desc = "Term()"
    description = ("Terminate the exploration phase. "
                   "Term() must be alone with no movement actions. "
                   "You MUST ONLY use it in the last turn and no other turns. Otherwise your action sequence will be invalid.")
    example = "Term()"
    format_pattern = r"^Term\(\)$"
    cost = 0
    def success_message(self, **kwargs) -> str:
        return "Exploration terminated."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot terminate exploration: execution failed."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute term action on room state."""
        return ActionResult(True, self.get_feedback(True), str(self), 'term', {'terminated': True})
    
    @staticmethod
    def is_final() -> bool:
        return True
    
    @staticmethod
    def is_term() -> bool:
        return True
    
    def __repr__(self):
        return "Term()"


# Internal-only forced terminate (not exposed to parser/registry)
class ForcedTermAction(TermAction):
    format_desc = "ForcedTerm()"
    description = "Forced termination when exploration steps are exhausted."
    example = ""
    format_pattern = r"^ForcedTerm\(\)$"
    cost = 0
    def success_message(self, **kwargs) -> str:
        return "Exploration ended. No further exploration actions are allowed."
    def execute(self, room, agent, **kwargs) -> ActionResult:
        return ActionResult(True, self.get_feedback(True), str(self), 'forced_term', {'terminated': True, 'internal': True})
    def __repr__(self): return "ForcedTerm()"


class QueryBase(BaseAction):
    """Base class for query actions."""
    format_desc = "Query(obj)"
    example = "Query(table)"
    format_pattern = r"^Query\(([A-Za-z0-9_ -]+)\)$"
    cost = 2
    def __init__(self, obj: str):
        super().__init__(obj)
        self.obj = obj if obj == 'initial_pos' else obj.replace('_', ' ')
    def error_message(self, error_type: str) -> str:
        return f"Cannot query: {error_type}"
    @staticmethod
    def is_final() -> bool: return True

class QueryAction(QueryBase):
    """Query object coordinates in the initial frame and emit relation triple to initial_pos."""
    description = (
        "Return object's coordinates with agent's initial position as origin, north as y+ axis. "
        "You can only query objects that you have previously observed. "
        "High cost, only use when necessary to eliminate ambiguities."
    )
    def success_message(self, **kwargs) -> str:
        return f"You query {self.obj}: {kwargs.get('answer','unknown')}"
    def __repr__(self): return f"Query({self.obj})"
    def execute(self, room, agent, **kwargs) -> ActionResult:
        if self.obj != 'initial_pos' and (not room.has_object(self.obj)):
            return ActionResult(False, self.get_feedback(False, "object not found"), str(self), 'query', {})
        observed_items = set(kwargs.get('observed_items', []))
        if observed_items is not None and self.obj != 'initial_pos' and self.obj not in observed_items:
            return ActionResult(False, self.get_feedback(False, "object not observed yet"), str(self), 'query', {})
        obj_pos = room.get_object_by_name(self.obj).pos if self.obj != 'initial_pos' else agent.init_pos
        obj_ori = room.get_object_by_name(self.obj).ori if self.obj != 'initial_pos' else agent.init_ori
        v = (obj_pos - agent.init_pos)
        assert np.allclose(agent.init_ori, np.array([0, 1])), "Initial orientation must be north"
        ans_pos = f"({int(v[0])}, {int(v[1])})"
        ori_pair = OrientationRel.get_relative_orientation(tuple(obj_ori), tuple(agent.init_ori))
        ans_ori = OrientationRel.to_string(ori_pair, 'allo', 'orientation')
        ans = ans_pos + ", " + ans_ori
        rel = PairwiseRelationship.relationship(tuple(obj_pos), tuple(agent.init_pos), anchor_ori=tuple(agent.init_ori), full=True)
        return ActionResult(True, self.get_feedback(True, answer=ans), str(self), 'query', {
            'answer': ans,
            'object': self.obj,
            'coords': (int(v[0]), int(v[1])),
            'orientation': ans_ori,
            'relation_triples': [RelationTriple(subject=self.obj, anchor='initial_pos', relation=rel, orientation=tuple(agent.init_ori))]
        })


class FalseBeliefTermAction(TermAction):
    """Terminate false belief exploration with answer."""
    
    format_desc = 'Term(changes="...")'
    description = ("Terminate the exploration phase and submit findings. "
                   "Format: Term(changes=\"obj1: type1, obj2: type2\")")
    example = 'Term(changes="apple: position, chair: orientation")'
    format_pattern = r'^Term\(changes="([^"]+)"\)$'
    cost = 0
    
    def __init__(self, changes_str: str):
        # BaseAction might store args, but TermAction doesn't expect any.
        super(TermAction, self).__init__(changes_str) 
        self.changes = self._parse_changes(changes_str)
        self.raw_changes_str = changes_str

    def _parse_changes(self, changes_str: str) -> List[Any]:
        # "apple: position, chair: orientation"
        changes_map = {}
        parts = changes_str.split(',')
        for part in parts:
            if ':' in part:
                try:
                    obj = ChangedObject.parse(part)
                    if obj.name in changes_map:
                        changes_map[obj.name].merge(obj)
                    else:
                        changes_map[obj.name] = obj
                except ValueError:
                    continue
                    
        return list(changes_map.values())

    def success_message(self, **kwargs) -> str:
        return f"Exploration terminated. Reported changes: {self.raw_changes_str}"
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        # Always terminate. If parsing failed, changes might be empty, which is handled by evaluation logic (low score).
        return ActionResult(True, self.get_feedback(True), str(self), 'term', {'terminated': True, 'reported_changes': self.changes})

    def error_message(self, error_type: str) -> str:
        if error_type == "invalid_format":
            return f"Invalid format for changes. Expected format: 'obj1: type1, obj2: type2'. Got: '{self.raw_changes_str}'"
        return "Cannot terminate exploration: execution failed."
        
    def __repr__(self):
        return f'Term(changes="{self.raw_changes_str}")'


# Action registry for easy lookup
# Expose all observe variants; default flows may still prefer ObserveApprox
ACTION_CLASSES = [
    MoveAction, RotateAction,
    ObserveAction, QueryAction, TermAction
]

def configure_actions(mode: str = 'exploration'):
    """Return the available action classes for a given mode.

    IMPORTANT: This must be pure (no global mutation). Multiple environments can
    run in the same Python process (e.g., service batch mode), so mutating a
    module-level registry will cause non-deterministic parsing across envs.
    """
    if mode == 'exploration':
        return [MoveAction, RotateAction, ObserveAction, QueryAction, TermAction]
    if mode == 'false_belief':
        return [MoveAction, RotateAction, ObserveAction, QueryAction, FalseBeliefTermAction]
    raise ValueError(f"Unknown mode: {mode}")


class ActionSequence:
    """Sequence of actions for spatial exploration"""
    
    def __init__(self, motion_actions: List[BaseAction] = None, final_action: BaseAction = None):
        self.motion_actions = motion_actions or []
        self.final_action = final_action
    
    def __repr__(self):
        motions = ", ".join(str(action) for action in self.motion_actions)
        return f"ActionSequence(motions=[{motions}], final={self.final_action})"

    @classmethod
    def parse(
        cls,
        action_str: str,
        action_classes: Optional[List[type[BaseAction]]] = None,
    ) -> Optional['ActionSequence']:
        m = re.search(r'\[(.*)\]', action_str.strip())
        if not m:
            return None
        # extract top-level actions like JumpTo(table), Rotate(90), Term()
        action_strs = re.findall(r'([A-Za-z]+\([^()]*\))', m.group(1))
        if not action_strs:
            return None

        # Parse all actions
        parsed_actions = []
        for act_s in action_strs:
            act = cls._parse_single_action(act_s.strip(), action_classes=action_classes)
            if not act:
                return None
            parsed_actions.append(act)

        # Otherwise, standard: zero or more motions then exactly one final
        motions, final_action = [], None
        for i, act in enumerate(parsed_actions):
            if i == len(parsed_actions) - 1:
                if not act.is_final():
                    return None
                final_action = act
            else:
                if act.is_final():
                    return None
                motions.append(act)
        if isinstance(final_action, TermAction):
            if motions:
                return None
        return cls(motions, final_action)
    
    @staticmethod
    def _parse_single_action(
        action_str: str,
        action_classes: Optional[List[type[BaseAction]]] = None,
    ) -> Optional[BaseAction]:
        """Parse a single action string using registered action classes"""
        for action_class in (action_classes or ACTION_CLASSES):
            if action := action_class.parse(action_str):
                return action
        return None
    
    @staticmethod
    def get_usage_instructions(vision: bool = False) -> str:
        """Get usage instructions for action sequences"""
        def _desc(cls):
            if vision and cls is ObserveAction:
                return (
                    "Return an RGB image of your current field of view from your current position and facing. "
                    "Use exactly one Observe() per step and make it the last action. "
                    "Never call Term() after Observe()."
                )
            return cls.description

        motion_actions = [cls for cls in ACTION_CLASSES if not cls.is_final()]
        final_actions = [cls for cls in ACTION_CLASSES if cls.is_final()]

        action_desc = (
            "Movement (<M>):\n" +
            "\n".join(f"- {cls.format_desc}: {_desc(cls)}" for cls in motion_actions) +
            "\n" +
            "Final (<F>):\n" +
            "\n".join(f"- {cls.format_desc}: {_desc(cls)}" for cls in final_actions)
        )
        examples = (
            f"Valid: Actions: [JumpTo(red door), Rotate(90), JumpTo(table), Observe()]\n" +
            f"Valid: Actions: [Observe()] | Query(table)\n" +
            f"Invalid (no final action): Actions: [JumpTo(table)]\n" +
            f"Invalid (more than one final action): Actions: [Observe(), Rotate(90), Observe()]\n" +
            f"Invalid (termination with other actions): Actions: [JumpTo(table), Term()]\n\n"
        )
        
        return ACTION_INSTRUCTION.format(
            actions=action_desc,
            examples=examples,
            field_of_view=BaseAction.get_field_of_view(),
            costs="\n".join(f"- {cls.format_desc}: {cls.cost}" for cls in [ObserveAction, QueryAction]),
        )