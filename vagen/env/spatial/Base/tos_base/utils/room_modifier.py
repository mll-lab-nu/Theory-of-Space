from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.room import Room
from ..core.object import Object


@dataclass
class ChangedObject:
    """Represents a reported/ground-truth change for one object."""

    name: str
    pos: bool = False
    ori: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name}
        if self.pos:
            out["pos"] = True
        if self.ori:
            out["ori"] = True
        return out

    def merge(self, other: "ChangedObject") -> None:
        if self.name != other.name:
            raise ValueError(f"Cannot merge changes for different objects: {self.name} vs {other.name}")
        self.pos = bool(self.pos or other.pos)
        self.ori = bool(self.ori or other.ori)

    @classmethod
    def parse(cls, text: str) -> "ChangedObject":
        """Parse strings like:
        - "apple: position"
        - "chair orientation"
        - "table moved"
        """
        s = str(text).strip().strip('"\'')

        pos_kws = ("position", "location", "moved", "pos")
        ori_kws = ("orientation", "rotation", "rotated", "ori", "facing")

        if ":" in s:
            name, rest = s.split(":", 1)
        else:
            parts = s.rsplit(" ", 1)
            if len(parts) != 2:
                raise ValueError(f"Cannot parse: {s}")
            name, rest = parts[0], parts[1]

        rest_low = str(rest).lower()
        is_pos = any(k in rest_low for k in pos_kws)
        is_ori = any(k in rest_low for k in ori_kws)
        if not (is_pos or is_ori):
            raise ValueError(f"Unknown change type: {rest}")

        return cls(name=str(name).strip().strip('"\'').replace("_", " "), pos=is_pos, ori=is_ori)


class RoomModifier:
    def modify(self, room: Room) -> Tuple[Room, List[ChangedObject]]:
        raise NotImplementedError


class ObjectModifier(RoomModifier):
    """Modify a room by moving OR rotating objects (never both per object)."""

    _ROTATIONS = (np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]))

    def __init__(self, seed: int, n_changes: int = 4, agent_pos: Optional[np.ndarray] = None):
        self.rng = np.random.default_rng(seed)
        # Always enforce four changes: two rotations and two moves.
        self.n_changes = int(n_changes) if n_changes is not None else 4
        self.agent_pos = tuple(map(int, agent_pos)) if agent_pos is not None else None

    def modify(self, room: Room) -> Tuple[Room, List[ChangedObject]]:
        r = room.copy()
        candidates = [o for o in r.objects if o.name != "agent"]
        total_changes = self.n_changes
        orientable = [o for o in candidates if getattr(o, "has_orientation", False)]
        n_orientation = min(2, len(orientable), total_changes)
        orientation_targets = list(self.rng.choice(orientable, size=n_orientation, replace=False)) if n_orientation else []

        remaining = [o for o in candidates if o not in orientation_targets]
        changes: List[ChangedObject] = []

        for obj in orientation_targets:
            assert self._rotate(obj), f"Failed to rotate object {obj.name}"
            changes.append(ChangedObject(name=obj.name, ori=True))

        occupied_positions = {(int(o.pos[0]), int(o.pos[1])) for o in r.all_objects}
        if self.agent_pos is not None:
            occupied_positions.add(self.agent_pos)
        position_pool = list(self.rng.permutation(remaining)) if remaining else []
        while position_pool and len(changes) < total_changes:
            obj = position_pool.pop()
            if self._move(r, obj, occupied_positions):
                changes.append(ChangedObject(name=obj.name, pos=True))

        # Rebuild to refresh object_map / membership derived from positions.
        return (Room.from_dict(r.to_dict()), changes)

    def _move(self, room: Room, obj: Object, occupied_positions: Optional[set] = None) -> bool:
        if getattr(room, "mask", None) is None:
            return False
        mask = room.mask
        valid = np.argwhere((mask >= 1) & (mask < 100))
        if valid.size == 0:
            return False

        cur = (int(obj.pos[0]), int(obj.pos[1]))
        occupied = occupied_positions if occupied_positions is not None else {(int(o.pos[0]), int(o.pos[1])) for o in room.all_objects}
        if occupied_positions is None and self.agent_pos is not None:
            occupied.add(self.agent_pos)
        candidates = [
            tuple(map(int, p))
            for p in valid
            if tuple(map(int, p)) not in occupied
            and ((int(p[0]) - cur[0]) ** 2 + (int(p[1]) - cur[1]) ** 2) >= 36
        ]
        if not candidates:
            return False

        new_pos = candidates[int(self.rng.integers(0, len(candidates)))]
        obj.pos = np.array(new_pos, dtype=int)
        if occupied_positions is not None:
            occupied_positions.discard(cur)
            occupied_positions.add(new_pos)
        return True

    def _rotate(self, obj: Object) -> bool:
        if not getattr(obj, "has_orientation", False):
            return False
        cur = tuple(int(x) for x in getattr(obj, "ori", np.array([0, 1])))
        opts = [r for r in self._ROTATIONS if tuple(int(x) for x in r) != cur]
        if not opts:
            return False
        obj.ori = opts[int(self.rng.integers(0, len(opts)))]
        return True


__all__ = ["ChangedObject", "RoomModifier", "ObjectModifier"]


if __name__ == "__main__":
    import argparse
    from .room_utils import RoomPlotter, initialize_room_from_json
    from .image_handler import ImageHandler
    from .eval_utilities import create_and_plot_room

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.data_dir:
        _, json_data = ImageHandler.load_data(args.data_dir, args.seed)
        room, agent = initialize_room_from_json(json_data)
    else:
        room, agent, _ = create_and_plot_room(seed=args.seed, plot=True)
    RoomPlotter.plot(room, agent, mode="img", save_path=f"room_{args.seed}_original.png")


    modified_room, changes = ObjectModifier(seed=args.seed, n_changes=4, agent_pos=agent.pos).modify(room)
    assert all((c.pos ^ c.ori) for c in changes), "Each object must change position OR orientation (not both)."
    print("Changes:", [c.to_dict() for c in changes])
    print("Modified room:", modified_room.to_dict())
    RoomPlotter.plot(modified_room, agent, mode="img", save_path=f"room_{args.seed}_modified.png")

