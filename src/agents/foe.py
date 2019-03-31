import numpy as np

from cvxopt import matrix, solvers

from .qlearner import QLearner


class FoeQ(QLearner):

    def run(self, transition, A):
        s_test = 71  # [1,2,1]
        # a_test = 21  # [4,0]

        pi1 = np.ones((112, 5))/5
        pi2 = np.ones((112, 5))/5

        self.e_decayrate = self.e/(10*self.n_iter)
        self.alpha_decayrate = self.alpha/self.n_iter

        self.Q1 = np.random.rand(112, 5, 5)
        self.Q2 = np.random.rand(112, 5, 5)

        rewards = []

        s0 = 71  # Alwats start at the same position, as in the pic
        err = []  # delta in Q(s,a)
        # prob = .2  # 1/len(actions)

        for T in range(self.n_iter):
            s = s0  # always initalize an episode in s0
            q_sa = self.Q1[s_test, 4, 0]

            for t in range(self.timeout):
                choice = np.random.rand()

                if choice <= self.e:
                    a1 = self.actions[np.random.randint(5)]
                    a2 = self.actions[np.random.randint(5)]
                else:
                    # find min value of dist for me
                    pi1[s] = np.array(self.findMinQ(self.Q1[s]))
                    # a1 = actions[np.random.choice(primeDista2)]
                    # find min value of dist for me
                    pi2[s] = np.array(self.findMinQ(self.Q2[s]))

                    # a1 = np.random.choice(actions, p = primeDista2)
                    # a2 = np.random.choice(actions, p = primeDista1)

                    a1 = np.argmax(pi1[s])
                    a2 = np.argmax(pi2[s])

                a = [a1, a2]  # action matrix
                # print a

                # query transition model to obtain s', returns an index value
                s_prime = transition(s, a)

                # query the reward model to obtain r
                r1 = self.rewards[s_prime, 0]
                r2 = self.rewards[s_prime, 1]

                v1 = pi1[s_prime] * self.Q1[s_prime]
                v2 = pi2[s_prime] * self.Q2[s_prime]
                self.Q1[s, a1, a2] = (
                    1. - self.alpha) * self.Q1[s, a1, a2] \
                    + self.alpha * (r1 + self.gamma * v1.max())
                self.Q2[s, a2, a1] = (
                    1. - self.alpha) * self.Q2[s, a2, a1] \
                    + self.alpha * (r2 + self.gamma * v2.max())

                # update s
                s = s_prime
                if self.e > .001:
                    self.e = self.e - self.e_decayrate

                # terminate when a goal is made
                if r1 != 0 or r2 != 0:
                    rewards.append([r1, r2])
                    break

            err.append(np.abs(self.Q1[s_test, 4, 0] - q_sa))
            print(T)
        return err, self.Q1

    def findMinQ(self, Qmin):

        c = matrix([
            Qmin[0, 0] + Qmin[0, 1] + Qmin[0, 2] + Qmin[0, 3] + Qmin[0, 4],
            Qmin[1, 0] + Qmin[1, 1] + Qmin[1, 2] + Qmin[1, 3] + Qmin[1, 4],
            Qmin[2, 0] + Qmin[2, 1] + Qmin[2, 2] + Qmin[2, 3] + Qmin[2, 4],
            Qmin[3, 0] + Qmin[3, 1] + Qmin[3, 2] + Qmin[3, 3] + Qmin[3, 4],
            Qmin[4, 0] + Qmin[4, 1] + Qmin[4, 2] + Qmin[4, 3] + Qmin[4, 4]
        ])

        G = matrix([
            [-1.0, 0., 0., 0., 0.],
            [0., -1.0, 0., 0., 0.],
            [0., 0., -1.0, 0., 0.],
            [0., 0., 0., -1.0, 0.],
            [0., 0., 0., 0., -1.0]
        ])
        h = matrix([0.0, 0.0, 0.0, 0.0, 0.0])

        A = matrix([
            [1.], [1.], [1.], [1.], [1.]
        ])

        b = matrix([1.])
        solvers.options['show_progress'] = False
        sol = solvers.lp(c, G, h, A, b)

        primeDist = [sol['x'][0], sol['x'][1],
                     sol['x'][2], sol['x'][3], sol['x'][4]]
        return primeDist

    def findMinQ2(self, Q1):

        c = matrix([0., 0., 0., 0., 0., 1.0])

        G = matrix(np.array(
            [
                [-1.0, 0., 0., 0., 0., 0.],
                [0., -1.0, 0., 0., 0., 0.],
                [0., 0., -1.0, 0., 0., 0.],
                [0., 0., 0., -1.0, 0., 0.],
                [0., 0., 0., 0., -1.0, 0.],
                [0., 0., 0., 0., 0., -1.0],
                [Q1[0, 0], Q1[0, 1], Q1[0, 2], Q1[0, 3], Q1[0, 4], -1.],
                [Q1[1, 0], Q1[1, 1], Q1[1, 2], Q1[1, 3], Q1[1, 4], -1.],
                [Q1[2, 0], Q1[2, 1], Q1[2, 2], Q1[2, 3], Q1[2, 4], -1.],
                [Q1[3, 0], Q1[3, 1], Q1[3, 2], Q1[3, 3], Q1[3, 4], -1.],
                [Q1[4, 0], Q1[4, 1], Q1[4, 2], Q1[4, 3], Q1[4, 4], -1.]
            ])
        )
        h = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        A = matrix([
            [1.], [1.], [1.], [1.], [1.], [0.]
        ])

        b = matrix([1.])

        solvers.options['show_progress'] = False
        sol = solvers.lp(c, G, h, A, b)

        primeDist = [sol['x'][0], sol['x'][1],
                     sol['x'][2], sol['x'][3], sol['x'][4]]
        return primeDist
