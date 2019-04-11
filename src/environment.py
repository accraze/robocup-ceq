from itertools import permutations

import numpy as np

from src.settings import EAST, FIELD_MAX, FIELD_MIN, NORTH, PUT, SOUTH, WEST

# actions = [NORTH, SOUTH, EAST, WEST, PUT]
# actions = [-4,4,1,-1,0]


class Environment:

    actions = [NORTH, SOUTH, EAST, WEST, PUT]

    def __init__(self):
        self.states_space = self._build_state_space()
        self.actions_space = self._build_actions_space()
        self.rewards_space = self._build_rewards_space()

    def test_boundary(self, player, action):
        if (player + action > FIELD_MAX or player + action < FIELD_MIN):
            return 0
        return action

    def transition(self, state, actions):
        ball, player1, player2 = self.states_space[state]
        actions[0] = self.test_boundary(player1, actions[0])
        actions[1] = self.test_boundary(player2, actions[1])
        p1_newstate = player1 + actions[0]
        p2_newstate = player2 + actions[1]

        state_index = state

        # GETTING NEW STATE
        if p1_newstate == p2_newstate:
            first_move = np.random.randint(2)
            if first_move == 0:  # I'm first
                if ball == 0:  # I have a ball
                    # I move, you don't, ball is still mine
                    next_state = [ball, player1 + actions[0], player2]
                else:
                    # i move, you bump into me, I get your ball
                    next_state = [ball - 1, player1 + actions[0], player2]
            else:  # you move first
                if ball == 1:  # you have the ball
                    # you move, you keep the ball, i stay
                    next_state = [ball, player1, player2 + actions[1]]
                else:  # i have the ball
                    # you move, I bump into you, you get the ball
                    next_state = [ball - 1, player1, player2 + actions[1]]
        else:
            next_state = [ball, player1 + actions[0], player2 + actions[1]]

        # GET INDEX OF S' STATE
        for i in range(self.states_space.shape[0]):
            if np.array_equal(self.states_space[i], np.array(next_state)):
                state_index = i

        return state_index

    def _build_state_space(self):
        state_space = np.array([[0, 0, h] for h in range(1, 8)] +
                               [[0, i, j] for i in range(1, 8)
                                for j in range(0, 8) if j != i] +
                               [[1, k, l] for k in range(0, 8)
                                for l in range(0, 8) if k != l])
        return state_space

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
