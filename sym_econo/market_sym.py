import torch
import numpy as np
import scipy.special as sp

class TimeSystem:
    def __init__(self):
        #1 tick = 10 sec
        self.time = 0

    def tick(self):
        self.time += 1

class NewsSystem(TimeSystem):
    def __init__(self, centers, sigma, der_probs, intensity=1/6):
        super().__init__()
        self.centers = centers
        assert 0 < sigma < 1, "sigma must be between 0 and 1"
        self.sigma = sigma
        self.weights = []
        for pair in centers:
            w1 = pair[1]/(pair[1] - pair[0])
            w2 = 1 - w1
            self.weights.append((w1, w2))

        self.der_probs = der_probs
        self.intensity = intensity

    def _spawn(self, derivative):
        w1, w2 = self.weights[derivative]
        mean1, mean2 = self.centers[derivative]
        sigma1, sigma2 = self.sigma*mean1, self.sigma*mean2
        choice = np.random.choice([0, 1], p=[w1, w2])
        if choice == 0:
            return np.random.normal(mean1, sigma1)
        else:
            return np.random.normal(mean2, sigma2)

    def tick(self):
        """returns:
        number of derivative affected by news,
        """
        super().tick()
        lmb = self.intensity
        p = lmb*np.exp(-lmb*self.time)
        check = np.random.choice([0, 1], p=[1 - p, p])
        if check:
            der = np.random.choice(range(len(self.der_probs)), p=self.der_probs)
            return der, self._spawn(der)
        else:
            return None

class MarketSystemN(NewsSystem):
    def __init__(self, init_mean, init_scale, c, s, d_p, ints=1/6):
        super().__init__(c, s, d_p, ints)
        self.d = len(init_mean[0]) # Number of different derivatives
        self.N = len(init_mean) # Number of closed systems within one
        self.mean = init_mean
        self.scale = init_scale

    def tick(self):
        regulations = super().tick()
        if regulations != None:
            d_num, diff = regulations


initial_conds_mu = [[100, 200, 200, 300, 300] for i in range(20)]
initial_conds_scale = [[20, 30, 30, 50, 50] for i in range(20)]
initial_conds = np.random.normal(loc=initial_conds_mu, scale=initial_conds_scale, size=(20, 5))

#probability of news being written to exact derivative
news_weights = [2.7, 3.3, 3.4, 3.6, 3.6]
news_cond = sp.softmax(news_weights)
print(news_cond)
