import numpy as np
from cvxopt import matrix, solvers

from .qlearner import QLearner


class CEQ(QLearner):

    def run(self, transition, A):
        s_test = 71  # [1,2,1]

        self.e_decayrate = self.e/(self.n_iter)
        self.alpha_decayrate = self.alpha/self.n_iter

        # initialize Q to zero over the states and joint action space
        self.Q1 = np.random.rand(112, 25)
        self.Q2 = np.random.rand(112, 25)

        pi1 = np.ones((112, 25))
        pi2 = np.ones((112, 25))
        s0 = 71  # Alwats start at the same position, as in the pic
        err = []  # delta in Q(s,a)

        for T in range(self.n_iter):
            s = s0  # always initalize an episode in s0
            q_sa = self.Q1[s_test, 21]

            for t in range(self.timeout):
                choice = np.random.rand()
                # QtableV = Q1.sum()

                if choice <= self.e:
                    a1 = np.random.randint(25)
                    a2 = np.random.randint(25)
                else:

                    # return an array of 25 elements with probs for A1 only
                    prime1 = self.findMinQ(self.Q1[s], self.Q2[s])
                    prime2 = self.findMinQ(self.Q2[s], self.Q1[s])

                    # ideally, I would want this to be averaged
                    pi1[s] = prime1
                    pi2[s] = prime2

                    # 25 prob valuea multiplied by anpther 25 Q values
                    QwithProb1 = self.Q1[s]*prime1
                    QwithProb2 = self.Q2[s]*prime2

                    # returns a number index where it is
                    a1c = np.argmax(QwithProb1)
                    a2c = np.argmax(QwithProb2)

                    a1 = A[a1c][0]
                    a2 = A[a2c][0]

                a = [a1, a2]  # action matrix

                # query transition model to obtain s', returns an index value
                s_prime = transition(s, a)

                # query the reward model to obtain r
                r1 = self.rewards[s_prime, 0]
                r2 = self.rewards[s_prime, 1]

                anp = np.array(a)
                for i in range(A.shape[0]):
                    if np.array_equal(A[i], anp):
                        aindex = i
                        break
                anp = np.array([a2, a1])
                for j in range(A.shape[0]):
                    if np.array_equal(A[j], anp):
                        aindex = j
                        break

                v1 = self.Q1[s_prime]*pi1[s_prime]
                v2 = self.Q2[s_prime]*pi2[s_prime]

                self.Q1[s, i] = (1. - self.alpha) * self.Q1[s, i] + \
                    self.alpha * ((1-self.gamma)
                                  * r1 + self.gamma * v1.max())
                self.Q2[s, j] = (1. - self.alpha) * self.Q2[s, j] + \
                    self.alpha * ((1-self.gamma)
                                  * r2 + self.gamma * v2.max())

                # update s
                s = s_prime
                if self.e > .001:
                    self.e = self.e - self.e_decayrate

                # terminate when a goal is made
                if r1 != 0 or r2 != 0:
                    break

            err.append(np.abs(self.Q1[s_test, 21] - q_sa))
            print(T)
        return err, self.Q1

    def findMinQ(self, Q1, Q2):
        c = matrix([
            -Q1[0], -Q1[1], -Q1[2], -Q1[3], -Q1[4], -
            Q1[5], -Q1[6], -Q1[7], -Q1[8], -Q1[9],
            -Q1[10], -Q1[11], -Q1[12], -Q1[13], -Q1[14], -
            Q1[15], -Q1[16], -Q1[17], -Q1[18], -Q1[19],
            -Q1[20], -Q1[21], -Q1[22], -Q1[23], -Q1[24],

            -Q2[0], -Q2[1], -Q2[2], -Q2[3], -Q2[4], -
            Q2[5], -Q2[6], -Q2[7], -Q2[8], -Q2[9],
            -Q2[10], -Q2[11], -Q2[12], -Q2[13], -Q2[14], -
            Q2[15], -Q2[16], -Q2[17], -Q2[18], -Q2[19],
            -Q2[20], -Q2[21], -Q2[22], -Q2[23], -Q2[24],
        ])

        G = matrix(np.identity(50) * -1)

        h = matrix(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0])

        A = matrix([
            [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
            [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
            [1.], [1.], [1.], [1.], [1.],
            [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
            [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
            [1.], [1.], [1.], [1.], [1.]
        ])

        b = matrix([1.])
        solvers.options['show_progress'] = False
        sol = solvers.lp(c, G, h, A, b)

        primeDist = [sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3],
                     sol['x'][4], sol['x'][5], sol['x'][6], sol['x'][7],
                     sol['x'][8], sol['x'][9],
                     sol['x'][10], sol['x'][11], sol['x'][12], sol['x'][13],
                     sol['x'][14], sol['x'][15], sol['x'][16],
                     sol['x'][17], sol['x'][18], sol['x'][19],
                     sol['x'][20], sol['x'][21], sol['x'][22], sol['x'][23],
                     sol['x'][24],
                     ]
        return primeDist
