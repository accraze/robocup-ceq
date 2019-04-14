import numpy as np

from src.settings import FIELD_MAX, FIELD_MIN
from src.utils import lookup_vector_index


class Transitions:

    """
    Handles all movements for grid soccer.
    Also checks out-of-bounds and collisions.
    """

    def __init__(self, debug=False):
        self.debug = debug

    def get_next_state(self, state, state_idx, states_space, actions):
        ball, player1, player2 = state
        self._log('trans: actions: {}'.format(actions))
        actions[0] = self.test_boundary(player1, actions[0])
        actions[1] = self.test_boundary(player2, actions[1])
        self._log('trans: curent_state {} {} {}'.format(
            ball, player1, player2))
        next_state = self._get_next_state(player1, player2, actions, ball)
        self._log('trans: next_state {}'.format(next_state))
        next_state_index = lookup_vector_index(
            states_space, next_state, state=state_idx)
        return next_state_index

    def test_boundary(self, player, action):
        """Check out of bounds errors."""
        if (player + action > FIELD_MAX or player + action < FIELD_MIN):
            return 0
        return action

    def _log(self, msg):
        if self.debug:
            print(msg)

    def _check_collision(self, player_coords):
        """Check if both player coords are in same cell. Returns bool."""
        return player_coords[0] == player_coords[1]

    def _get_next_state(self, p1, p2, actions, ball):
        """Transition from current state to next state."""
        player_steps = ((p1 + actions[0]), (p2 + actions[1]))
        if self._check_collision(player_steps):
            self._log('collision! p1: {} p2: {}'.format(
                player_steps[0], player_steps[1]))
            next_state = self._handle_collision(p1, p2, ball, actions)
        else:
            next_state = self._advance(p1, ball, p2, actions)
        return next_state

    def _handle_collision(self, p1, p2, ball, actions):
        """Decide who moves and takes possesion of ball during collision."""
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
        """Check results of coinflip. If 0, P1 moves first."""
        return coin_flip == 0

    def _p1_has_ball(self, ball):
        """Check if P1 has ball."""
        return ball == 0

    def _p1_move_ball(self, p1, ball, p2, action):
        """P1 moves with ball."""
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

    def _advance(self, p1_cell, ball, p2_cell, actions):
        """No collision detected. Both players move."""
        p1_cell = p1_cell + actions[0]
        p2_cell = p2_cell + actions[1]
        return self._new_state(ball, p1_cell, p2_cell)

    def _new_state(self, ball, p1_cell, p2_cell):
        return [ball, p1_cell, p2_cell]
