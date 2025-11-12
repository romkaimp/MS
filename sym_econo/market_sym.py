from pathlib import Path

import torch
import numpy as np
import scipy.special as sp
from matplotlib import pyplot as plt

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
        for i in range(len(sigma)):
            assert 0 < sigma[i] < 1, "sigma must be between 0 and 1"
        self.sigma = sigma
        self.weights = []
        for pair in centers:
            w1 = pair[1]/(pair[1] - pair[0])
            w2 = 1 - w1
            self.weights.append((w1, w2))

        self.der_probs = der_probs
        self.intensity = intensity

        self.timer = TimeSystem()

    def _spawn(self, derivative):
        w1, w2 = self.weights[derivative]
        mean1, mean2 = self.centers[derivative]
        sigma1, sigma2 = self.sigma[derivative]*abs(mean1), self.sigma[derivative]*abs(mean2)
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

        self.timer.tick()

        lmb = self.intensity
        p = 1 - np.exp(-lmb*self.timer.time)
        check = np.random.choice([0, 1], p=[1 - p, p])
        if check:
            der = np.random.choice(range(len(self.der_probs)), p=self.der_probs)
            self.timer.time = 0
            return der, self._spawn(der)
        else:
            return None


class MarketEvent(TimeSystem):
    def __init__(self, start, delta, volatility, lifetime, func_type):
        """func_type == ['sigmoid', 'linear', 'exponential']"""
        super().__init__()
        self.start = start
        self.delta = delta
        self.volatility = volatility
        self.life_time = lifetime
        self.func_type = func_type
        assert func_type in ['sigmoid', 'linear', 'exponential'], "func_type must be one of ['sigmoid', 'linear', 'exponential']"
        if func_type == 'sigmoid':
            self.func = lambda t: start + delta / (1 + np.exp(-t*self.volatility + 6/self.volatility))
        elif func_type == 'linear':
            self.func = lambda t: t/lifetime * (start + delta) + (lifetime - t) / lifetime * start
        elif func_type == 'exponential':
            self.func = lambda t: start + delta * (-np.exp(-t*self.volatility*3/lifetime) + 1)

        self.is_active = True

    def tick(self):
        super().tick()
        if self.time >= self.life_time:
            self.is_active = False
        return self.func(self.time), self.is_active

    def __call__(self):
        return self.tick()


class MarketSystemN(NewsSystem):
    def __init__(self, init_mean: list, init_scale: list, N, volatilities: list, *,
                 centers: list[list], sigma, derivative_prod, ints=1/6):
        """volatilities: present speed of news impact, 1 = initial, <1 - slower, >1 - faster"""
        super().__init__(centers, sigma, derivative_prod, ints)
        self.d = len(init_mean) # Number of different derivatives
        self.N = N # Number of closed systems within one
        self.mean = init_mean
        self.scale = init_scale # for stationary system
        self.volatility = volatilities # for transcendental event

        self.effects = [[] for _ in range(self.d)] # first dym - each derivative, second dim - each derivative market event
        self.prices = np.random.normal(loc=[init_mean for _ in range(N)],
                                       scale=[init_scale for _ in range(N)],
                                       size=(N, self.d))
        self.events = np.zeros(d)

    def tick(self):
        regulations = super().tick()
        self.events = np.zeros(self.d)
        if regulations != None:
            d_num, diff = regulations
            self.events[d_num] = diff
            self.effects[d_num].append(MarketEvent(self.mean[d_num], diff, self.volatility[d_num], 6, 'sigmoid'))

        for effect_num, effect in enumerate(self.effects):
            for num, market_event in enumerate(effect):
                self.mean[effect_num], is_active = market_event.tick()
                if not is_active:
                    del market_event

        self.prices = np.random.normal(loc=[self.mean for _ in range(self.N)],
                                       scale=[self.scale for _ in range(self.N)],
                                       size=(self.N, self.d))


class MarketTools:
    def __init__(self, d, N, save_dir="plots"):
        self.d = d
        self.N = N
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_prices(self, prices, numbers, avg=False, marginal=False, events=None, save=None):
        prices = np.asarray(prices)
        if prices.ndim != 3:
            raise ValueError("prices must be a 3D array with shape [T, N, d]")
        T, N_in, d_in = prices.shape
        if N_in != self.N or d_in != self.d:
            raise ValueError(f"prices shape mismatch: expected second dim N={self.N}, third dim d={self.d}, got {prices.shape}")
        if not isinstance(numbers, (list, tuple, np.ndarray)):
            raise ValueError("numbers must be a list/tuple/ndarray of integers")
        numbers = list(numbers)
        if len(numbers) == 0:
            raise ValueError("numbers must contain at least one system index")
        for idx in numbers:
            if not (0 <= idx < self.N):
                raise IndexError(f"system index {idx} out of range [0, {self.N-1}]")

        if events is not None:
            events = np.asarray(events)
            if events.shape != (T, self.d):
                raise ValueError(f"events must have shape [T, d] == {(T, self.d)}, got {events.shape}")

        time = np.arange(T)
        mean_over_N = None
        std_over_N = None
        if avg or marginal:
            mean_over_N = prices.mean(axis=1)
        if marginal:
            std_over_N = prices.std(axis=1, ddof=0)

        ''' asset plotting'''
        for asset in range(self.d):
            fig, ax = plt.subplots(figsize=(10, 4))
            for sys in numbers:
                ax.plot(time, prices[:, sys, asset], label=f"system {sys}")
            if avg and mean_over_N is not None:
                ax.plot(time, mean_over_N[:, asset], label="mean across N", linewidth=2)
            if marginal and mean_over_N is not None and std_over_N is not None:
                mean_series = mean_over_N[:, asset]
                std_series = std_over_N[:, asset]
                upper = mean_series + 3 * std_series
                lower = mean_series - 3 * std_series
                ax.plot(time, mean_series, label="mean (marginal)", linewidth=2)
                ax.fill_between(time, lower, upper, alpha=0.15)

            ymin, ymax = ax.get_ylim()
            yrange = ymax - ymin if ymax > ymin else 1.0
            label_y = ymin + 0.01 * yrange
            label_vpad = 0.02 * yrange

            if events is not None:
                nonzero_idx = np.nonzero(events[:, asset])
                print(nonzero_idx)
                nonzero_idx = nonzero_idx[0]
                for i, t in enumerate(nonzero_idx):
                    val = events[t, asset]
                    ax.axvline(t, linestyle='--', linewidth=1.0, alpha=0.7)
                    prev_count = np.sum(nonzero_idx[:i] == t)
                    y_pos = label_y + prev_count * label_vpad
                    ax.text(t, y_pos, f"{val:g}", rotation=0, ha='center', va='bottom', fontsize=9, backgroundcolor='white', alpha=0.85)

            ax.set_title(f"Asset {asset} — price series (T={T}, shown systems: {numbers})")
            ax.set_xlabel("time (t)")
            ax.set_ylabel("price")
            ax.grid(True)
            ax.legend(loc="best")
            plt.tight_layout()

            if save:
                filename = self._generate_filename(asset, numbers, avg, marginal, save)
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Graph saved as: {filename}")

            plt.show()

    def _generate_filename(self, asset, numbers, avg, marginal, save):
        """Генерирует имя файла для сохранения графика"""
        base_name = f"asset_{asset}"

        # Добавляем информацию о системах
        if len(numbers) == 1:
            base_name += f"_system_{numbers[0]}"
        else:
            systems_str = "_".join(map(str, numbers[:3]))  # Берем первые 3 системы для имени
            if len(numbers) > 3:
                systems_str += f"_and_{len(numbers) - 3}_more"
            base_name += f"_systems_{systems_str}"

        # Добавляем флаги
        flags = []
        if avg:
            flags.append("avg")
        if marginal:
            flags.append("marginal")

        if flags:
            base_name += "_" + "_".join(flags)

        # Добавляем расширение если нужно
        if isinstance(save, str):
            if save.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
                filename = self.save_dir / save
            else:
                filename = self.save_dir / f"{save}_{base_name}.png"
        else:
            filename = self.save_dir / f"{base_name}.png"

        return filename


if __name__ == "__main__":
    initial_conds_mu = [1000, 2000, 2000, 3000, 3000]
    initial_conds_scale = [20, 30, 30, 50, 50]
    # probability of news being written to exact derivative
    news_weights = [2.7, 3.3, 3.4, 3.6, 3.6]
    news_probs = sp.softmax(news_weights)
    #print(news_probs)
    d = 5
    N = 100
    volatilities = [1, 1.2, 1.2, 1.5, 1.5]
    news_sigma = [0.4, 0.6, 0.6, 0.8, 0.8]
    centers = [[50, -50], [100, -120], [120, -100], [1000, -50], [100, -700]]
    #centers = [[centers_[i][j]*5 for j in range(2)] for i in range(5)]
    #news_sigma = [news_sigma[i]*0.1 for i in range(5)]

    market = MarketSystemN(initial_conds_mu, initial_conds_scale, N, volatilities, centers=centers, sigma=news_sigma, derivative_prod=news_probs, ints=1/6)
    T = 100
    prices = [market.prices]
    events = [np.zeros(d)]
    for t in range(T):
        market.tick()
        prices.append(market.prices)
        events.append(market.events)

    events = np.stack(events, axis=0)
    prices = np.stack(prices, axis=0) # [T, N, d]
    print(events.shape)
    print(prices.shape)

    tools = MarketTools(d, N)
    tools.plot_prices(prices, [1, 2, 5], avg=True, marginal=True, events=events, save='./plots')