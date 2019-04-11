import numpy as np
from cvxopt import matrix, solvers

from src.settings import INITIAL_STATE, SOUTH, PUT, TOTAL_STATES, TOTAL_ACTIONS

from .qlearner import QLearner


class FoeQ(QLearner):

    def run(self):
        self.Q1 = self._init_q_table()
        self.Q2 = self._init_q_table()
        self.err = []

        self.e_decayrate = self.e/(10*self.n_iter)
        self.alpha_decayrate = self.alpha/self.n_iter

        self.p1_pas = self._init_pas_table()
        self.p2_pas = self._init_pas_table()

        for episode_num in range(self.n_iter):
            self._run_simulation(episode_num, INITIAL_STATE)
        return self.err, self.Q1

    def _init_q_table(self):
        return np.random.rand(TOTAL_STATES, TOTAL_ACTIONS, TOTAL_ACTIONS)

    def _init_pas_table(self):
        return np.ones((TOTAL_STATES, TOTAL_ACTIONS))/TOTAL_ACTIONS 

    def _run_simulation(self, episode_num, initial_state):
        print(episode_num)
        q_diff_base = self.Q1[initial_state, SOUTH, PUT]
        self._run_match(initial_state)
        self._decay_hyperparams()
        self.err.append(np.abs(self.Q1[initial_state, SOUTH, PUT] - q_diff_base))

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
            p1_action = self.actions[np.random.randint(TOTAL_ACTIONS)]
            p2_action = self.actions[np.random.randint(TOTAL_ACTIONS)]
        else:
            self._get_probablistic_actions(state) 
            p1_action = np.argmax(self.p1_pas[state])
            p2_action = np.argmax(self.p2_pas[state])
        return [p1_action, p2_action]

    def _get_probablistic_actions(self, state):
        """Compute action probabilities for given state."""
        self.p1_pas[state] = np.array(self.get_min_q_val(self.Q1[state]))
        self.p2_pas[state] = np.array(self.get_min_q_val(self.Q2[state]))        

    def _update_q_tables(self, state, actions, rewards, next_state):
        action_prob1 = self.p1_pas[next_state] * self.Q1[next_state]
        action_prob2 = self.p2_pas[next_state] * self.Q2[next_state]
        self.Q1[state, actions[0], actions[1]] = (
            1. - self.alpha) * self.Q1[state, actions[0], actions[1]] \
            + self.alpha * (rewards[0] + self.gamma * action_prob1.max())
        self.Q2[state, actions[1], actions[0]] = (
            1. - self.alpha) * self.Q2[state, actions[1], actions[0]] \
            + self.alpha * (rewards[0] + self.gamma * action_prob2.max())

    def get_min_q_val(self, q):
        c = matrix(np.array(self._build_c(q)))
        G = matrix(np.negative(np.identity(TOTAL_ACTIONS)))
        h = matrix(np.array([0.] * TOTAL_ACTIONS))
        A = matrix([[1.] for x in range(TOTAL_ACTIONS)])
        b = matrix(np.array([1.]))
        solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
        sol = solvers.lp(c, G, h, A, b, solver='glpk')['x']
        prime_dist = [sol[x] for x in range(TOTAL_ACTIONS)]
        return prime_dist

    def _build_c(self, q):
        c_arr = []
        for i in range(TOTAL_ACTIONS):
            row_sum = 0.
            for j in range(TOTAL_ACTIONS):
                row_sum = row_sum + q[i, j]
            c_arr.append(row_sum)
        return c_arr
