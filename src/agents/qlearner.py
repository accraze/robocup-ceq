import numpy as np

from src.settings import INITIAL_STATE


class QLearner:

    """
    Base Q-Learner to show
    that the multiagent setup will not converge.
    """

    def __init__(self, environment, eps=.5, gamma=0.99, alpha=.5,
                 decay=.001, n_iterations=10000, timeout=25):
        self.states = environment.states_space
        self.actions = environment.actions
        self.rewards = environment.rewards_space
        self.actions_space = environment.actions_space
        self.env = environment
        self.e = eps
        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay
        self.n_iter = n_iterations
        self.timeout = timeout
        self.e_decayrate = self._set_decay_rate(self.e)
        self.alpha_decayrate = self._set_decay_rate(self.alpha)

    def run(self):
        self.Q1 = self._init_q_table(self.states, self.actions)
        self.Q2 = self._init_q_table(self.states, self.actions)
        self.err = []
        for episode_num in range(self.n_iter):
            self._run_simulation(episode_num, INITIAL_STATE)
        return self.err, self.Q1

    def _set_decay_rate(self, hyperparam):
        return (hyperparam - self.decay)/self.n_iter

    def _init_q_table(self, s, a):
        return np.random.rand(len(s), len(a))

    def _run_simulation(self, episode_num, initial_state):
        print(episode_num)
        q_diff_base = self.Q1[initial_state, 4]
        self._run_match(initial_state)
        self._decay_hyperparams()
        self.err.append(np.abs(self.Q1[initial_state, 4] - q_diff_base))

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
        """
        Either take random action or best possible action.
        """
        if np.random.rand() <= self.e:
            p1_action = self.actions[np.random.randint(5)]
            p2_action = self.actions[np.random.randint(5)]
        else:
            p1_action = self.actions[np.argmax(self.Q1[state])]
            p2_action = self.actions[np.argmax(self.Q2[state])]

        return [p1_action, p2_action]

    def _goal_is_made(self, rewards):
        return (rewards[0] != 0 or rewards[1] != 0)

    def _query_rewards(self, state):
        return self.rewards[state, 0], self.rewards[state, 1]

    def _update_q_tables(self, state, actions, rewards, next_state):
        self.Q1[state, actions[0]] = self.Q1[state, actions[0]] + \
            self.alpha * (rewards[0] + self.gamma *
                          self.Q1[next_state, :].max() -
                          self.Q1[state, actions[0]])

        self.Q2[state, actions[1]] = self.Q2[state, actions[1]] + \
            self.alpha * (rewards[1] + self.gamma *
                          self.Q2[next_state, :].max() -
                          self.Q2[state, actions[1]])

    def _decay_hyperparams(self):
        self.e = self.e - self.e_decayrate
        if self.alpha > .001:
            self.alpha = self.alpha - self.alpha_decayrate
