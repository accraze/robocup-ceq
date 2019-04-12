from itertools import permutations

import numpy as np

from src.settings import EAST, FIELD_MAX, FIELD_MIN, NORTH, PUT, SOUTH, WEST
from src.utils import lookup_vector_index

# actions = [NORTH, SOUTH, EAST, WEST, PUT]
# actions = [-4,4,1,-1,0]


class Environment:

    actions = [NORTH, SOUTH, EAST, WEST, PUT]

    def __init__(self, transitions, debug=False):
        self.states_space = self._build_state_space()
        self.actions_space = self._build_actions_space()
        self.rewards_space = self._build_rewards_space()
        self.debug = debug
        self.transition_model = transitions

    def _build_state_space(self):
        state_space = np.array([[0, 0, h] for h in range(1, 8)] +
                               [[0, i, j] for i in range(1, 8)
                                for j in range(0, 8) if j != i] +
                               [[1, k, l] for k in range(0, 8)
                                for l in range(0, 8) if k != l])
        return state_space

    def transition(self, state, actions):
        next_state = self.transition_model.get_next_state(
            self.states_space[state], state, self.states_space, actions)
        return next_state

    def _build_actions_space(self):
        actions_space = list(permutations(self.actions, 2))
        for a in self.actions:
            actions_space.append((a, a))

        return np.array(actions_space)

    def _build_rewards_space(self):
        p1_scores = [100., -100.]
        no_scores = [0., 0.]
        p2_scores = [-100., 100.]
        rewards = np.array(
            [p1_scores for x in range(7)] + [no_scores for x in range(14)] +
            [p2_scores for x in range(7)] + [p1_scores for x in range(7)] +
            [no_scores for x in range(14)] + [p2_scores for x in range(7)] +
            [no_scores for x in range(2)] + [p2_scores for x in range(1)] +
            [p1_scores for x in range(1)] + [no_scores for x in range(2)] +
            [p2_scores for x in range(1)] + [p1_scores for x in range(1)] +
            [no_scores for x in range(1)] + [p2_scores for x in range(1)] +
            [p1_scores for x in range(1)] + [no_scores for x in range(2)] +
            [p2_scores for x in range(1)] + [p1_scores for x in range(1)] +
            [no_scores for x in range(1)] + [p2_scores for x in range(1)] +
            [p1_scores for x in range(1)] + [no_scores for x in range(2)] +
            [p2_scores for x in range(1)] + [p1_scores for x in range(1)] +
            [no_scores for x in range(2)] + [p1_scores for x in range(1)] +
            [no_scores for x in range(2)] + [p2_scores for x in range(1)] +
            [p1_scores for x in range(1)] + [no_scores for x in range(2)] +
            [p2_scores for x in range(1)] + [no_scores for x in range(2)] +
            [p2_scores for x in range(1)] + [p1_scores for x in range(1)] +
            [no_scores for x in range(2)] + [p2_scores for x in range(1)] +
            [p1_scores for x in range(1)] + [no_scores for x in range(1)] +
            [p2_scores for x in range(1)] + [p1_scores for x in range(1)] +
            [no_scores for x in range(2)] + [p2_scores for x in range(1)] +
            [p1_scores for x in range(1)] + [no_scores for x in range(1)] +
            [p2_scores for x in range(1)] + [p1_scores for x in range(1)] +
            [no_scores for x in range(2)] + [p2_scores for x in range(1)] +
            [p1_scores for x in range(1)] + [no_scores for x in range(2)])
        return rewards
