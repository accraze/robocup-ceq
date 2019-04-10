import numpy as np

from src.settings import INITIAL_STATE

from .qlearner import QLearner


class FriendQ(QLearner):

    def run(self, transition, action_space):
        self.Q1 = self._init_q_table(self.states, action_space)
        self.Q2 = self._init_q_table(self.states, action_space)
        self.err = []
        for episode_num in range(self.n_iter):
            self._run_simulation(episode_num, INITIAL_STATE,
                                 transition, action_space)
        return self.err, self.Q1

    def _run_simulation(self, episode_num, initial_state, transition,
                        action_space):
        print(episode_num)
        q_diff_base = self.Q1[initial_state, 7]  # 7 is for [4, 0] action?
        self._run_match(initial_state, transition)
        self._decay_hyperparams()
        self.err.append(np.abs(self.Q1[initial_state, 7] - q_diff_base))

    def _run_match(self, state, transition, action_space):
        for t in range(self.timeout):
            actions = self._select_actions()
            next_state = transition(state, actions)
            rewards = self._query_rewards(next_state)
            self._update_q_tables(state, actions, rewards, next_state,
                                  action_space)
            state = next_state
            if self._goal_is_made(rewards):
                break

    def _select_actions(self, state, action_space):
        """
        Either take random action or best possible action.
        """
        if np.random.rand() <= self.e:
            p1_action = action_space[np.random.randint(25)][0]
            p2_action = action_space[np.random.randint(25)][1]
        else:
            p1_action = action_space[np.argmax(self.Q1[state])][0]
            p2_action = action_space[np.argmax(self.Q2[state])][1]
        return [p1_action, p2_action]

    def _a_equals(self, a, b):
        return np.array_equal(a, b)

    def _update_q_tables(self, state, actions, rewards, next_state,
                         action_space):
        for i in range(action_space.shape[0]):
            if self._a_equals(action_space[i], np.array(actions)):
                update_action = i
                break

        self.Q1[state, update_action] = self.Q1[state, update_action] + \
            self.alpha * \
            (rewards[0] + self.gamma * self.Q1[next_state, :].max() -
             self.Q1[state, update_action])
        self.Q2[state, update_action] = self.Q2[state, update_action] + \
            self.alpha * \
            (rewards[1] + self.gamma * self.Q2[next_state, :].max() -
             self.Q2[state, update_action])
