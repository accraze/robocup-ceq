from itertools import permutations

import numpy as np

from src.settings import EAST, FIELD_MAX, FIELD_MIN, NORTH, PUT, SOUTH, WEST
from src.utils import lookup_vector_index

# actions = [NORTH, SOUTH, EAST, WEST, PUT]
# actions = [-4,4,1,-1,0]


class Environment:

    actions = [NORTH, SOUTH, EAST, WEST, PUT]

    def __init__(self, debug=False):
        self.states_space = self._build_state_space()
        self.actions_space = self._build_actions_space()
        self.rewards_space = self._build_rewards_space()
        self.debug = debug

    def test_boundary(self, player, action):
        if (player + action > FIELD_MAX or player + action < FIELD_MIN):
            return 0
        return action

    def transition(self, state, actions):
        ball, player1, player2 = self.states_space[state]
        self._log('trans: actions: {}'.format(actions))
        actions[0] = self.test_boundary(player1, actions[0])
        actions[1] = self.test_boundary(player2, actions[1])
        self._log('trans: curent_state {} {} {}'.format(ball, player1, player2))
        next_state = self._get_next_state(player1, player2, actions, ball)
        self._log('trans: next_state {}'.format(next_state))
        next_state_index = lookup_vector_index(self.states_space, next_state, state=state)
        return next_state_index

    def _check_collision(self, player_coords):
        return player_coords[0] == player_coords[1]

    def _get_next_state(self, p1, p2, actions, ball):
        player_steps = ((p1 + actions[0]), (p2 + actions[1]))
        if self._check_collision(player_steps):
            self._log('collision! p1: {} p2: {}'.format(player_steps[0], player_steps[1]))
            next_state = self._handle_collision(p1, p2, ball, actions)
        else:
            next_state = self._move_to_ball(p1, ball, p2, actions)
        return next_state

    def _handle_collision(self, p1, p2, ball, actions):
        coin_flip = np.random.randint(2)
        if self._p1_moves_first(coin_flip):
            if self._p1_has_ball(ball):
                next_state = self._p1_move_ball(p1, ball, p2, actions[0])
            else:
                next_state = self._p1_gets_ball(p1, ball, p2, actions[0])
        else:
            if not self._p1_has_ball(ball):
                next_state = self._p2_move_ball(p1, ball, p2, actions[1])
            else:
                next_state = self._p2_gets_ball(p1, ball, p2, actions[1])

        return next_state

    def _p1_moves_first(self, coin_flip):
        return coin_flip == 0

    def _p1_has_ball(self, ball):
        return ball == 0

    def _p1_move_ball(self, p1, ball, p2, action):
        p1 = p1 + action
        return self._new_state(ball, p1, p2)

    def _p1_gets_ball(self, p1_cell, ball, p2_cell, action):
        """
        P1 moves and gets ball.
        """
        ball = ball - 1  # flip ball
        p1_cell = p1_cell + action
        return self._new_state(ball, p1_cell, p2_cell)

    def _p2_move_ball(self, p1, ball, p2, action):
        p2 = p2 + action
        return self._new_state(ball, p1, p2)

    def _p2_gets_ball(self, p1_cell, ball, p2_cell, action):
        """
        P2 moves and gets ball.
        """
        ball = ball + 1  # flip ball
        p2_cell = p2_cell + action
        return self._new_state(ball, p1_cell, p2_cell)

    def _move_to_ball(self, p1_cell, ball, p2_cell, actions):
        p1_cell = p1_cell + actions[0]
        p2_cell = p2_cell + actions[1]
        return self._new_state(ball, p1_cell, p2_cell)

    def _new_state(self, ball, p1_cell, p2_cell):
        return [ball, p1_cell, p2_cell]

    def _build_state_space(self):
        state_space = np.array([[0, 0, h] for h in range(1, 8)] +
                               [[0, i, j] for i in range(1, 8)
                                for j in range(0, 8) if j != i] +
                               [[1, k, l] for k in range(0, 8)
                                for l in range(0, 8) if k != l])
        return state_space

    def _log(self, msg):
        if self.debug:
            print(msg)

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
