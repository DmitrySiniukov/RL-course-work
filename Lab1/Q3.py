from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

map_size  = (4,4)
start_pos = (0,0)
goal_pos = (1,1)
init_po_pos = (3,3)
STATE_SPACE_SIZE = (map_size[0] * map_size[1]) ** 2
EPSILON = 0.1
ALPHA = 2.154434e-2
GAMMA = 0.8
ITER = 10_000_000
SARSA_ITER = 500_000

class ActionSpace(Enum):
    STAY = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class Environment:
    def __init__(self):
        # TODO set Initial condition
        self.pos = start_pos
        self.po_pos = init_po_pos
    def _a2pos(self, action:ActionSpace, is_police=False):

        origin = self.po_pos if is_police else self.pos

        if action == ActionSpace.UP:
            return origin[0], origin[1] - 1
        if action == ActionSpace.RIGHT:
            return origin[0] + 1, origin[1]
        if action == ActionSpace.DOWN:
            return origin[0], origin[1] + 1
        if action == ActionSpace.LEFT:
            return origin[0] - 1, origin[1]
        return origin # ActionSpace.Stay

    def can_step(self, pos, is_police = False):
        origin = self.po_pos if is_police else self.pos
        if pos[0] < 0 or pos[1] < 0: # top-left bound check
            return False
        elif pos[0] >= map_size[0] or pos[1] >= map_size[1]: # bottom right bound check
            return False
        diff = np.abs(origin[0] - pos[0]) + np.abs(origin[1] - pos[1]) # Step check
        if diff > 1:
            return False
        return True

    def list_actions(self):
        return [a for a in ActionSpace if self.can_step(self._a2pos(a))] # Check

    def reward(self, new_pos):
        if new_pos == self.po_pos:
            return -10
        elif new_pos == goal_pos:
            return 1
        return 0

    def move_police(self):
        moves = [a for a in ActionSpace if (self.can_step(self._a2pos(a, is_police=True), is_police=True) # Check boundary
                                            and a != ActionSpace.STAY)] # Police action space doesn't have stay
        move = np.random.choice(moves)
        #print('police move:', move)
        self.po_pos = self._a2pos(move, is_police=True)

    def step(self, action):
        #print('action:', action)
        new_pos = self._a2pos(action)
        if not self.can_step(new_pos):
            raise Exception('Illegal Move')
        self.pos = new_pos
        self.move_police()

        return self.state, self.reward(new_pos)

    def print(self):
        for j in range(map_size[1]):
            for i in range(map_size[0]):
                if(i,j) == self.pos and (i,j) == self.po_pos:
                    print('😵', end='|')
                elif (i,j) == self.pos:
                    print('😎', end='|')
                elif (i, j) == self.po_pos:
                    print('👮‍', end='|')
                elif (i,j) == goal_pos:
                    print('🚩️', end='|')
                else:
                    print('  ', end='|')
            print('')
        print('')

    @property
    def state(self):
        return (*self.pos, *self.po_pos)
class QLearner:
    def __init__(self, env:Environment):
        self.Q = np.zeros((*map_size, *map_size, len(ActionSpace)))
        self.Q_counter = np.ones((*map_size, *map_size, len(ActionSpace)))
        self.env = env
        pass
    def epsilon_soft_exploration(self, state):
        actions = self.env.list_actions()
        if np.random.binomial(1, EPSILON) == 1:
            return np.random.choice(actions)
        else:
            action_list = [i.value for i in actions]
            q_idx = (*state, action_list) # select only available actions
            action_values = self.Q[q_idx]
            actions_max_value = np.max(action_values) # argmax does not return multiple max
            max_value_actions = [actions[action_idx] for action_idx, action_value in enumerate(action_values) if action_value == actions_max_value]
            return np.random.choice(max_value_actions)

    def uniform_random_exploration(self, state):
        actions = self.env.list_actions()
        return np.random.choice(actions)

    def q_learning(self, step_size=ALPHA):
        state = self.env.state
        v_plot = []
        rewards = 0.0
        v_plot_x = [] # used for x axis
        for iter in range(1, ITER):

            action = self.uniform_random_exploration(state)


            #rand_action = np.random.choice(self.env.list_actions())
            next_state, reward = self.env.step(action)
            rewards += reward
            # Q-Learning update
            q_idx = (*state, action.value)
            step_size = np.power(self.Q_counter[q_idx], -2/3)
            self.Q_counter[q_idx] += 1
            self.Q[q_idx] += step_size * (
                    reward + GAMMA * np.max(self.Q[(*next_state,)]) -
                    self.Q[q_idx])
            state = next_state

            if(iter % 1000 == 0):
                rewards = 0.0
                v = np.max(self.Q[(*start_pos, *init_po_pos,)])
                v_plot.append(v)
                v_plot_x.append(iter)


            if(iter % 10000 == 0):
                print('i', iter, ', v(s_o):', v)


            # if(iter > 500_000):
            #     self.env.print()
            #     pass
        plt.plot(v_plot_x, v_plot, label='Q-Learning')
        #self.plot(rewards_growth, win_plt, loss_plot)
        return rewards

    def plot(self, rewards, win=None, loss=None, ):
        plt.figure()
        plt.plot(rewards, label='net reward')
        if win: plt.plot(win, label='win reward')
        if loss: plt.plot(loss, label='loss reward')
        plt.legend()
        plt.show()
        pass

class SARSA:
    def __init__(self, epsilon, env:Environment):
        self.epsilon = epsilon
        self.Q = np.zeros((*map_size, *map_size, len(ActionSpace)))
        self.Q_counter = np.ones((*map_size, *map_size, len(ActionSpace)))
        self.env = env
        pass

    def epsilon_soft_exploration(self, state):
        actions = self.env.list_actions()
        epsilon = self.epsilon
        if np.random.binomial(1, epsilon) == 1:
            return np.random.choice(actions)
        else:
            action_list = [i.value for i in actions]
            q_idx = (*state, action_list) # select only available actions
            action_values = self.Q[q_idx]
            actions_max_value = np.max(action_values) # argmax does not return multiple max
            max_value_actions = [actions[action_idx] for action_idx, action_value in enumerate(action_values) if action_value == actions_max_value]
            return np.random.choice(max_value_actions)

    def sarsa(self, step_size=ALPHA):
        state = self.env.state
        rewards = 0.0
        rewards_growth = []
        loss_rewards = 0.0
        loss_plot = []
        win_rewards = 0.0
        win_plt = []
        v_plot = [np.sum(self.Q[(*start_pos,*init_po_pos,)])]
        v_plot_x = [0]


        action = self.epsilon_soft_exploration(state)
        for iter in range(1, SARSA_ITER):
            next_state, reward = self.env.step(action)
            next_action = self.epsilon_soft_exploration(next_state)
            rewards += reward
            # Q-Learning update
            q_idx = (*state, action.value)

            step_size = np.power(self.Q_counter[q_idx], -2/3)
            self.Q_counter[q_idx] += 1

            self.Q[q_idx] += step_size * (
                    reward + GAMMA * self.Q[(*next_state,next_action.value)] -
                    self.Q[q_idx])
            state = next_state
            action = next_action


            if(iter % 10000 == 0):

                rewards_growth.append(rewards)
                loss_plot.append(loss_rewards)
                win_plt.append(win_rewards)

            if(iter % 1000 == 0):
                rewards = 0.0
                loss_rewards = 0.0
                win_rewards = 0.0

                v = np.max(self.Q[(*start_pos,*init_po_pos,)])
                v_plot.append(v)
                v_plot_x.append(iter)
                # print('i', iter, ', v:', v)


            # if(iter > 500_000):
            #     self.env.print()
            #     pass

        plt.plot(v_plot_x, v_plot, label='epsilon %.2f' % self.epsilon)
        plt.legend()

        # self.plot(rewards_growth, win_plt, loss_plot)
        return rewards

def main():
    env = Environment()
    env.print()

    plt.title('Q Learning')
    q = QLearner(env)
    q.q_learning()

    # epsilons = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    # for epsilon in epsilons:
    #     np.random.seed(123)
    #     print('computing epsilon', epsilon)
    #     env = Environment()
    #     env.print()
    #     sa = SARSA(epsilon, env)
    #     sa.sarsa()

    plt.ylabel('value')
    plt.xlabel('iterations')
    plt.show()

if __name__== "__main__":
  main()




