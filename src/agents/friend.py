import numpy as np

from .qlearner import QLearner


class FriendQ(QLearner):

    def run(self, transition, A):
        s_test = 71  # [1,2,1]
        a_test = 21  # [4,0]
        self.Q1 = np.random.rand(len(self.states), len(A))
        self.Q2 = np.random.rand(len(self.states), len(A))
        s0 = 71  # Always start in same position
        err = []

        for T in range(self.n_iter):
            s = s0  # always initalize an episode in s0
            q_sa = self.Q1[s_test, a_test]

            for t in range(self.timeout):
                choice = np.random.rand()
                if choice <= self.e:
                    a1c = np.random.randint(25)
                    a1 = A[a1c][0]
                    a2c = np.random.randint(25)
                    a2 = A[a2c][1]
                else:
                    # BEST PAIR OF ACTIONS FROM 25 POSSIBLE ACTIONS
                    a1c = np.argmax(self.Q1[s])
                    a1 = A[a1c][0]
                    a2c = np.argmax(self.Q2[s])
                    a2 = A[a2c][1]

                a = [a1, a2]  # action matrix
                # query transition model to obtain s', returns an index value
                s_prime = transition(s, a)

                # query the reward model to obtain r
                r1 = self.rewards[s_prime, 0]
                r2 = self.rewards[s_prime, 1]

                np_a = np.array(a)
                for i in range(A.shape[0]):
                    if np.array_equal(A[i], np_a):
                        updateA = i
                        break

                self.Q1[s, updateA] = self.Q1[s, updateA] + self.alpha * \
                    (r1 + self.gamma * self.Q1[s_prime,
                                               :].max() - self.Q1[s, updateA])
                self.Q2[s, updateA] = self.Q2[s, updateA] + self.alpha * \
                    (r2 + self.gamma * self.Q2[s_prime,
                                               :].max() - self.Q2[s, updateA])
                # update s
                s = s_prime

                self.e = self.e - self.e_decayrate
                if self.alpha > .001:
                    self.alpha = self.alpha - self.alpha_decayrate

                # terminate when a goal is made
                if r1 != 0 or r2 != 0:
                    break

            err.append(np.abs(self.Q1[s_test, a_test] - q_sa))
            print(T)

        return err, self.Q1
