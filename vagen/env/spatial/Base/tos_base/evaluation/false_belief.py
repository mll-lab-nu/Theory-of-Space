"""False belief task: detect a changed object (rotation or movement)."""

import numpy as np

from .tasks import retry_generate_question
from ..core.relationship import CardinalBinsAllo, PairwiseRelationshipDiscrete, OrientationRel
from ..utils.utils import hash
from .direction import DirectionPov, _store_relation

"""
Task Overview:
1. FalseBeliefDirectionPov: Relative location after object rotation (false belief).
   - Evaluated by: direction and distance match.
"""

FALSE_BELIEF_TEMPLATE = (
    "Facing north in one room, you note some objects' orientation:\n{observations}\n\n"
    "Assume the {anchor_name}'s facing defines 'north' (not true north).\n"
    "Where is {obj_name} relative to {anchor_name}?\n\n"
    "Answer format: <cardinal direction>, <distance>\n"
    "Example: north-west, near\n"
)

# ---- New task: rotate one oriented object, then ask DirectionPov using it as anchor ----
class FalseBeliefDirectionPov(DirectionPov):
    
    QUESTION_TEMPLATE = FALSE_BELIEF_TEMPLATE

    @retry_generate_question
    def generate_question(self) -> str:
        oriented = [o for o in self.room.objects if o.has_orientation]
        if len(oriented) < 1:
            raise ValueError("Need >=1 oriented objects for this task")

        # 1) Rotate one oriented object (anchor A)
        anchor = self.np_random.choice(oriented)
        deg = int(self.np_random.choice([90, 180, 270]))
        rotations = {0: [[1, 0], [0, 1]], 90: [[0, -1], [1, 0]], 180: [[-1, 0], [0, -1]], 270: [[0, 1], [-1, 0]]}
        anchor.ori = anchor.ori @ rotations[deg]

        # 2) Observe facing north; report all oriented objects in the same room
        tmp_agent = self.agent.copy()
        tmp_agent.ori = np.array((0, 1))
        rid = int(getattr(anchor, 'room_id', getattr(self.agent, 'room_id', 0)))
        objs = [o for o in self.room.objects if o.has_orientation and int(getattr(o, 'room_id', -1)) == rid and o.name != anchor.name][:4] + [anchor]
        objs.sort(key=lambda o: o.name)
        def facing(o):
            op = OrientationRel.get_relative_orientation(tuple(o.ori), tuple(tmp_agent.ori))
            return OrientationRel.to_string(op, 'ego', 'orientation')
        observations = "\n".join(f"{o.name}: {facing(o)}" for o in objs)

        # 3) Ask DirectionPov with this rotated object as anchor A
        target_candidates = [i for i, o in enumerate(self.room.objects) if o is not anchor]
        target = self.room.objects[int(self.np_random.choice(target_candidates))]
        rel = PairwiseRelationshipDiscrete.relationship(
            tuple(target.pos),
            tuple(anchor.pos),
            anchor_ori=tuple(anchor.ori),
            bin_system=CardinalBinsAllo(),
        )
        
        # Store the answer directly (open-ended format)
        question = self.QUESTION_TEMPLATE.format(
            observations=observations,
            anchor_name=anchor.name,
            obj_name=target.name
        )
        _store_relation(self, question, rel)
        self.eval_data.kwargs = {"rotated_object": anchor.name, "rotation_degrees": deg}
        return question


if __name__ == "__main__":
    from ..utils.eval_utilities import create_and_plot_room, manual_test_loop
    from .task_types import EvalTaskType

    # Robustness test suggestions:
    # 1. Tuple vs String: Ensure "(North, Near)" and "North, Near" are both accepted.
    # 2. Case insensitivity.
    # 3. Order swapping: Check if distance-first is accepted for this specific task logic.

    task_name = 'false_belief'
    print(f"\nTesting task: {task_name}")
    try:
        room, agent, np_random = create_and_plot_room(task_name)
        task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
        
        manual_test_loop(task_name, task, EvalTaskType.evaluate_prediction)

    except ValueError as e:
        print(f"Skipping {task_name}: {e}")
