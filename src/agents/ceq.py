import numpy as np
from cvxopt import matrix, solvers

from src.settings import INITIAL_STATE, PUT, SOUTH, TOTAL_ACTIONS, TOTAL_STATES
from src.utils import lookup_vector_index

from .qlearner import QLearner


class CEQ(QLearner):

    def run(self):
        self.Q1 = self._init_q_table()
        self.Q2 = self._init_q_table()
        self.err = []

        self.e_decayrate = self.e/(self.n_iter)
        self.alpha_decayrate = self.alpha/(self.n_iter)

        self.p1_pas = self._init_pas_table()
        self.p2_pas = self._init_pas_table()

        for episode_num in range(self.n_iter):
            self._run_simulation(episode_num, INITIAL_STATE)
        return self.err, self.Q1

    def _init_q_table(self):
        """Init to zero over state and joint action spaces."""
        return np.zeros((TOTAL_STATES, TOTAL_ACTIONS * TOTAL_ACTIONS))

    def _init_pas_table(self):
        return np.ones((TOTAL_STATES, TOTAL_ACTIONS * TOTAL_ACTIONS))

    def _run_simulation(self, episode_num, initial_state):
        print(episode_num)
        initial_action = lookup_vector_index(self.actions_space, [SOUTH, PUT])
        q_diff_base = self.Q1[initial_state, initial_action]
        self._run_match(initial_state)
        self._decay_hyperparams()
        self.err.append(
            np.abs(self.Q1[initial_state, initial_action] - q_diff_base))

    def _run_match(self, state):
        for t in range(self.timeout):
            actions = self._select_actions(state)
            next_state = self.env.transition(state, actions)
            rewards = self._query_rewards(next_state)
            self._update_q_tables(state, actions, rewards, next_state)
            state = next_state
            if self._goal_is_made(rewards):
                break

    def _select_actions(self, state):
        if np.random.rand() <= self.e:
            p1_action = np.random.randint(TOTAL_ACTIONS*TOTAL_ACTIONS)
            p2_action = np.random.randint(TOTAL_ACTIONS*TOTAL_ACTIONS)
        else:
            p1_action, p2_action = self._get_probablistic_actions(state)
        return [p1_action, p2_action]

    def _get_probablistic_actions(self, state):
        self.p1_pas[state] = self._get_min_q_val(state)
        self.p2_pas[state] = self._get_min_q_val(state)
        p1_idx = np.argmax(self.Q1[state]*self.p1_pas[state])
        p2_idx = np.argmax(self.Q2[state]*self.p2_pas[state])
        return self.actions_space[p1_idx][0], self.actions_space[p2_idx][0]

    def _update_q_tables(self, state, actions, rewards, next_state):
        p1_action = lookup_vector_index(self.actions_space, actions)
        p2_action = lookup_vector_index(self.actions_space, actions[::-1])
        v1 = self.Q1[next_state]*self.p1_pas[next_state]
        v2 = self.Q2[next_state]*self.p2_pas[next_state]
        self.Q1[state, p1_action] = (1. - self.alpha) \
            * self.Q1[state, p1_action] + \
            self.alpha * ((1-self.gamma)
                          * rewards[0] + self.gamma * v1.max())
        self.Q2[state, p2_action] = (1. - self.alpha) \
            * self.Q2[state, p2_action] + \
            self.alpha * ((1-self.gamma) * rewards[1] + self.gamma * v2.max())

    def _build_c(self, state):
        qs = [self.Q1[state], self.Q2[state]]
        return np.array([-q[x] for q in qs for x in range(TOTAL_ACTIONS*TOTAL_ACTIONS)])

    def _get_min_q_val(self, state):
        c = matrix(self._build_c(state))
        G = matrix(np.negative(np.identity(50)))
        h = matrix([0.,] * 50)
        A = matrix([[1.] for x in range(50)])
        b = matrix([1.])
        return self._solve_pas(c,G,h,A,b)

    def _solve_pas(self,c,G,h,A,b):
        """Compute probablistic action space using LP."""
        solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
        sol = solvers.lp(c, G, h, A, b, solver='glpk')['x']
        return [sol[x] for x in range(TOTAL_ACTIONS*TOTAL_ACTIONS)]
