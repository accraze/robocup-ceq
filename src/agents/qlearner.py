import numpy as np
from src.agents.base import Agent


class QLearner(Agent):

    def __init__(self, player_info, num_states, num_actions,
                 lr=1.0, lr_decay=0.999997, lr_min=0.0,
                 eps=0.75, eps_decay=0.99995, eps_min=0.01,
                 gamma=0.9):
        self.state = 0
        self.action = 0
        self.num_states = num_states
        self.num_actions = num_actions

        self.Q = self._init_q_table()
        self.alpha = lr
        self.alpha_decay = lr_decay
        self.alpha_min = lr_min
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.algo_name = 'Q-Learner'
        x, y, has_ball, name = player_info
        super(QLearner, self).__init__(x, y, has_ball, name)

    def _init_q_table(self):
        # Initialized to [-1, 1) uniformly at random
        return np.random.randn(self.num_states, self.num_actions)
        # Initialized to [0, 1) uniformly at random
        # return np.random.random((self.num_states, self.num_actions))
        # Initialized to 0
        # return np.zeros((self.num_states, self.num_actions))

    def query_initial(self, state):
        if np.random.random() < self.eps:
            action = np.random.choice(self.num_actions)
            self.eps = max(
                self.eps * self.eps_decay, self.eps_min)
        else:
            action = np.argmax(self.Q[state])

        # Update current state and action
        self.state = state
        self.action = action

        return action

    def query(self, s, a, o, sp, r, op_Q):
        delta_Q = self.update_Q((s, a, sp, r))

        if np.random.random() < self.eps:
            action = np.random.choice(self.num_actions)
            self.eps = max(
                self.eps * self.eps_decay, self.eps_min)
        else:
            action = np.argmax(self.Q[sp])

        # Update current state and action
        self.state = sp
        self.action = action

        return action, delta_Q

    def update_Q(self, experience_replay):
        s, a, sp, r = experience_replay
        action = np.argmax(self.Q[s])
        prev_Q = self.Q[s, a]
        updated_Q = (1 - self.alpha) * prev_Q + \
            self.alpha * ((1 - self.gamma) * r + self.gamma *
                          self.Q[sp, action] - prev_Q)
        self.Q[s, a] = updated_Q
        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        return abs(updated_Q - prev_Q)
