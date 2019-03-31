import numpy as np


class QLearner:

    def __init__(self, states, actions, rewards, eps=.5, gamma=0.99, alpha=.5,
                 decay=.001, n_iterations=10000, timeout=25):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.e = eps
        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay
        self.n_iter = n_iterations
        self.timeout = timeout
        self.e_decayrate = self._set_decay_rate(self.e)
        self.alpha_decayrate = self._set_decay_rate(self.alpha)

    def run(self, transition):
        self.Q1 = np.random.rand(len(self.states), len(self.actions))
        self.Q2 = np.random.rand(len(self.states), len(self.actions))
        s0 = 71  # Always start in same position
        err = []
        for T in range(self.n_iter):
            s = s0  # init episode to s0
            q_sa = self.Q1[s0, 4]

            for t in range(self.timeout):
                choice = np.random.rand()
                if choice <= self.e:
                    a1 = self.actions[np.random.randint(5)]
                    a2 = self.actions[np.random.randint(5)]
                else:

                    # max(Q1[s]).astype(int) -4, 1, 0
                    a1 = self.actions[np.argmax(self.Q1[s])]
                    a2 = self.actions[np.argmax(self.Q2[s])]

                a = [a1, a2]  # action matrix
                # query transition model to obtain s', returns an index value
                s_prime = transition(s, a)

                # query the reward model to obtain r
                r1 = self.rewards[s_prime, 0]
                r2 = self.rewards[s_prime, 1]

                self.Q1[s, a1] = self.Q1[s, a1] + self.alpha * \
                    (r1 + self.gamma *
                     self.Q1[s_prime, :].max() - self.Q1[s, a1])
                self.Q2[s, a2] = self.Q2[s, a2] + self.alpha * \
                    (r2 + self.gamma *
                     self.Q2[s_prime, :].max() - self.Q2[s, a2])
                # update s
                s = s_prime

                # terminate when a goal is made
                if r1 != 0 or r2 != 0:
                    break
            # Decay Alpha
            self.e = self.e - self.e_decayrate
            if self.alpha > .001:
                self.alpha = self.alpha - self.alpha_decayrate

            err.append(np.abs(self.Q1[s0, 4] - q_sa))
            print (T)
        return err, self.Q1

    def _set_decay_rate(self, hyperparam):
        return (hyperparam - self.decay)/self.n_iter
