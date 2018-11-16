import numpy as np
from enum import Enum

T = 15 # time limit
map_width = 6
map_height = 5
map_size = (map_width, map_height)
our_pos_zero = (0,0)
his_pos_zero = (4,4)
exit_pos = (4,4)

class Step(Enum):
    NONE = (0,0)
    LEFT = (-1,0)
    UP = (0,-1)
    RIGHT = (1,0)
    DOWN = (0,1)

class Direction(Enum):
    NONE = 0
    LEFT = 1
    UP = 2
    RIGHT = 3
    DOWN = 4

    @classmethod
    def inverse(cls, direction):
        if direction is cls.NONE:
            return cls.NONE
        if direction is cls.LEFT:
            return cls.RIGHT
        if direction is cls.UP:
            return cls.DOWN
        if direction is cls.RIGHT:
            return cls.LEFT
        if direction is cls.DOWN:
            return cls.UP

def in_map(pos):
    if (pos[0] < 0 or pos[1] < 0 or pos[0] >= map_width or pos[1] >= map_height):
        return False
    return True

# Our partial state (without the minotaur)
class PartialState:

    def __init__(self, pos):
        self.pos = pos  # (x, y) tuple

    def getActions(self):
        walls_dirs = walls.get(self.pos, [])
        return [it for it in Direction if it not in walls_dirs]  # add the borders-check

    def act(self, direction):
        # Validity check
        if direction not in self.getActions():
            raise Exception("Illegal action!")
        self.pos = PartialState.step(self.pos, direction)

    @classmethod
    def step(cls, pos, direction):
        if direction is Direction.NONE:
            return (pos[0], pos[1])
        if direction is Direction.LEFT:
            return (pos[0] - 1, pos[1])
        if direction is Direction.UP:
            return (pos[0], pos[1] - 1)
        if direction is Direction.RIGHT:
            return (pos[0] + 1, pos[1])
        return (pos[0], pos[1] + 1)

    @classmethod
    def getActions(cls, pos, walls):
        walls_dirs = walls.get(pos, [])
        return [dr for dr in Direction if
                dr not in walls_dirs and in_map(PartialState.step(pos, dr))]  # add the borders-check


class State:
    def __init__(self, part_state, minotaur_pos):
        self.minotaur_pos = minotaur_pos  # (x, y) tuple of the Minotaur's position
        self.part_state = part_state


# One-way walls
walls = { (1,0): [Direction.RIGHT], (1,1): [Direction.RIGHT], (1,2): [Direction.RIGHT],
        (3,1): [Direction.RIGHT], (3,2): [Direction.RIGHT],
        (4,1): [Direction.DOWN], (5,1): [Direction.DOWN],
        (1,3): [Direction.DOWN], (2,3): [Direction.DOWN], (3,3): [Direction.DOWN], (4,3): [Direction.DOWN],
        (3,4): [Direction.RIGHT] }

# New values are appended to the first (base) dictionary
def append_to_dict(base_dict, new_vals):
    for key,vals in new_vals.items():
        base_vals = base_dict.get(key, None)
        if base_vals is None:
            base_vals = []
            base_dict[key] = base_vals
        base_vals.extend(vals)

# Mirroring the walls
mirror_walls = {}
for key,vals in walls.items():
    for val in vals:
        mirror_pos = PartialState.step(key, val)
        mirror_vals = mirror_walls.get(mirror_pos, None)
        if mirror_vals is None:
            mirror_vals = []
            mirror_walls[mirror_pos] = mirror_vals
        mirror_vals.append(Direction.inverse(val))
append_to_dict(walls, mirror_walls)

def value_iteration_method():
    time_left = 30
    state_space_size = (time_left, *map_size, *map_size)
    V = np.zeros(state_space_size)
    theta = 1e-6
    delta = 1

    while delta > theta:
        delta = 0.0
        for t_left, our_x, our_y, his_x, his_y in np.ndindex(state_space_size):
            our_pos = (our_x, our_y)
            his_pos = (his_x, his_y)
            s = (t_left, *our_pos, *his_pos)
            v = V[s]
            if t_left == 0 and our_pos != exit_pos:
                V[s] = -1 # Ran out of time
            elif our_pos == his_pos:
                V[s] = -1 # Eaten
            elif our_pos == exit_pos:
                V[s] = 1 # Made it
            else:
                his_pos_new = [PartialState.step(his_pos, act) for act in PartialState.getActions(his_pos, walls)]
                p = 1/len(his_pos_new)
                best_act_value = -np.infty
                for direction in PartialState.getActions(our_pos, walls):
                    our_pos_new = PartialState.step(our_pos, direction)
                    new_val = np.sum([V[(t_left-1, *our_pos_new, *his_p)] for his_p in his_pos_new]) * p
                    if (new_val > best_act_value):
                        best_act_value = new_val
                V[s] = best_act_value
            new_delta = np.abs(v - V[s])
            if new_delta > delta:
                delta = new_delta


def belman_eq(state, t):
    our_pos, his_pos = state

    # argmax
    for direction in PartialState.getActions(our_pos, walls):
        our_pos_new = PartialState.step(our_pos, direction)
