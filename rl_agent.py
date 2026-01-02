import numpy as np

class QAgent:
    def __init__(self):
        #400 états (20x20 grid)
        self.q_table = np.zeros((400, 4))
        self.lr = 0.1
        self.gamma = 0.95  # Augmenté pour viser le long terme
        self.eps = 0.5     # Augmenté pour plus d'exploration au début
        self.eps_decay = 0.995 # Ajout d'un decay pour réduire l'exploration petit à petit
        self.eps_min = 0.05

    def act(self, s):
        if np.random.rand() < self.eps:
            return np.random.randint(4)
        return np.argmax(self.q_table[s])

    def learn(self, s, a, r, sn):
        target = r + self.gamma * np.max(self.q_table[sn])
        self.q_table[s, a] += self.lr * (target - self.q_table[s, a])
        
        # Mise à jour de l'exploration
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
