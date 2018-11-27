from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import itertools

map_size  = (6,3)
start_pos = (0,0)
goal_pos = [(0,0), (0,2),(5,2),(5,0)]
init_po_pos = (2,1)
STATE_SPACE_SIZE = (map_size[0] * map_size[1]) ** 2
THETA = 0.1
MAX_ITER = 20


class ActionSpace(Enum):
    STAY = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class State(object):
    def __init__(self, first=None, second = None):
        if not first: # initial state
            self.pos = start_pos
            self.po_pos = init_po_pos
        else:
            self.pos = first
            self.po_pos = second

    @property
    def idx(self):
        return (*self.pos, *self.po_pos)

class Environment:

    def _a2pos(self, action:ActionSpace, state:State, is_police=False):

        origin = state.po_pos if is_police else state.pos

        if action == ActionSpace.UP:
            return origin[0], origin[1] - 1
        if action == ActionSpace.RIGHT:
            return origin[0] + 1, origin[1]
        if action == ActionSpace.DOWN:
            return origin[0], origin[1] + 1
        if action == ActionSpace.LEFT:
            return origin[0] - 1, origin[1]
        return origin # ActionSpace.Stay

    def can_step(self, pos, state:State, is_police = False):
        origin = state.po_pos if is_police else state.pos
        if pos[0] < 0 or pos[1] < 0: # top-left bound check
            return False
        elif pos[0] >= map_size[0] or pos[1] >= map_size[1]: # bottom right bound check
            return False
        diff = np.abs(origin[0] - pos[0]) + np.abs(origin[1] - pos[1]) # Step check
        if diff > 1:
            return False

        return True

    def list_actions(self, state):
        if(self.reward(state) < 0): # No next state if current state is dead
            return []
        return [a for a in ActionSpace if self.can_step(self._a2pos(a, state), state)] # Check

    def reward(self, state:State):
        if state.pos == state.po_pos:
            return -50
        elif state.pos in goal_pos:
            return 10
        return 0

    def move_police(self, state:State):

        actions = []

        first = 9999
        second = 9999
        for a in ActionSpace:
            next_pos = self._a2pos(a, state, is_police=True)
            if not self.can_step(next_pos, state, is_police=True):
                continue
            if a == ActionSpace.STAY: # Police can't stay
                continue

            actions.append(a)

            # compute the two/three closest moves
            dis = np.linalg.norm(np.subtract(state.pos, next_pos))
            if(dis < first):
                second = first
                first = dis
            elif(dis < second):
                second = dis

        filtered_actions = [] # filter moves
        for a in actions:
            diff =  np.subtract(state.pos, self._a2pos(a, state, is_police=True))
            dis = np.linalg.norm(diff)
            if  dis <= second:
                filtered_actions.append(a)

        return [ self._a2pos(a, state, is_police=True) for a in filtered_actions ] # returns police position

    def step(self, action, state):
        #print('action:', action)

        if(self.reward(state) < 0): # No next state if current state is dead
            return [State()], [0]

        new_pos = self._a2pos(action, state)
        if not self.can_step(new_pos, state):
            raise Exception('Illegal Move')
        police_poses = self.move_police(state)

        next_states = [State(new_pos, next_po_pos) for next_po_pos in police_poses]
        next_rewards = [self.reward(state) for state in next_states]
        next_states_idx = [state.idx for state in next_states]
        next_state_chosen = np.random.choice(next_states)

        return next_states_idx, next_rewards, next_state_chosen

    def print(self, state:State):
        pos = state.pos
        po_pos = state.po_pos

        for j in range(map_size[1]):
            for i in range(map_size[0]):
                if(i,j) == pos and (i,j) == po_pos:
                    print('😵', end='|')
                elif (i,j) == pos:
                    print('😎', end='|')
                elif (i, j) == po_pos:
                    print('👮‍', end='|')
                elif (i,j) in goal_pos:
                    print('🚩️', end='|')
                else:
                    print('  ', end='|')
            print('')
        print('')


class ValueIterator:
    def __init__(self, lmbda, env:Environment):
        self.lmbda = lmbda
        self.env = env
        self.V = np.zeros((*map_size, *map_size))
        x = np.arange(map_size[0])
        y = np.arange(map_size[1])
        xy = list(itertools.product(x, y))
        self.StateSpace = list(itertools.product(xy, xy))

        self.run_state = State()

    def learn(self, ):
        delta = 999
        _lambda = self.lmbda
        initial_state = State()
        benchmark = [self.V[initial_state.idx]]
        iter = 0
        while delta > THETA:
            if(iter > MAX_ITER): break;
            delta = 0.0
            for pos, po_pos in self.StateSpace:
                state = State(pos, po_pos)
                v = self.V[state.idx]

                # max value
                max_a = -np.inf
                for action in self.env.list_actions(state):
                    next_states_idx, rewards, _ = self.env.step(action, state)
                    next_states_idx_reordered = list(zip(next_states_idx[0], next_states_idx[1]))
                    all_v = self.V[next_states_idx_reordered] * _lambda
                    summed = np.sum(rewards) + np.sum(all_v)

                    if summed > max_a:
                        max_a = summed
                if max_a > -np.inf:
                    self.V[state.idx] = max_a
                new_v = self.V[state.idx]
                diff = np.abs(v - new_v)

                delta = np.maximum(delta, np.abs(v - new_v))

            iter += 1
            benchmark.append(self.V[initial_state.idx])

        return benchmark

    def simulate(self, ):

        max_a_v = -np.inf
        max_a = ActionSpace.STAY
        state = self.run_state
        _lambda = self.lmbda

        for action in self.env.list_actions(self.run_state):
            next_states_idx, rewards, _ = self.env.step(action, state)
            next_states_idx_reordered = list(zip(next_states_idx[0], next_states_idx[1]))
            all_v = self.V[next_states_idx_reordered] * _lambda
            summed = np.sum(rewards) + np.sum(all_v)

            if summed > max_a_v:
                max_a_v = summed
                max_a = action

        if max_a_v > -np.inf:
            _, _, next_state = self.env.step(max_a, state)
        self.env.print(next_state)
        self.run_state = next_state
        pass






def main():

    lambdas = [0.01, 0.02, 0.05,0.1,0.2, 0.4, 0.5]
    for lmd in lambdas:
        print('computing with lambda', lmd)
        env = Environment()
        vi = ValueIterator(lmd, env)
        convergence = vi.learn()
        plt.plot(convergence, label='lambda=%.2f' % lmd)

        np.random.seed(123)
        for i in range(100):
            vi.simulate()

    plt.title('Value Iteration with different lambda')
    plt.legend()
    plt.savefig('Q2_lambda_graph.png')
    #plt.ylim((0,100))
    plt.show()



if __name__== "__main__":
  main()

