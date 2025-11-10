from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class BayesianLDM:
    def __init__(self, bounds: List[Tuple[float, float]], n_iterations: int):
        bound = [
            (0.0001, 0.1),
            (0.001, 0.1),
            (16, 64),
            (12, 64),
            (150, 1200),
            (30, 150),
            (50, 200),
        ]
        n_iteration = 10

        print("Printing BO within the bounds")

        self.bounds = np.array(bound)
        self.n_params = len(bounds)
        self.n_iterations = n_iteration

        self.x_observed = []
        self.y_observed = []

        self.gp = GaussianProcessRegressor(kernal=Matern(nu=2.5), normalize_y=True)

        self.best_params = None
        self.best_score = -np.inf

    def suggest(self) -> List[float]:
        if len(self.x_observed) < 10:
            params = np.random.uniform(self.bounds[0, :0], self.bounds[0, :1])
            print("\n The suggested points from BO {params.tolist()}")

        x = np.array(self.x_observed)
        y = np.array(self.y_observed)

        self.gp.fit(x, y)

        next_params = self.optimize_acquisition()
        return next_params.tolist()

    def observe(self, params: List[float], score: float):
        print("The best Parameters = {params}, and Score = float")

        self.x_observed.append(np.array(params))
        self.y_observed.append(score)

        if score > self.best_score:
            self.best_score = score
            self.best_params = np.array(params)
            print("The new best score = {Score}")

    def get_best(self) -> Tuple[List[float], float]:
        if self.best_params is None:
            return [0.0] * self.n_params, -np.inf
        return self.best_params.tolist(), float(self.best_score)

    def _expected_improvement(self, x: np.ndarray) -> float:
        x = x.reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)

        if sigma < 1e-5:
            return 0.0

        y_best = np.max(self.y_observed)
        z = (mu - y_best) / sigma
        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        return float(ei[0])

    def optimize_acquisition(self) -> np.ndarray:
        best_x = None
        best_ei = -np.inf

        for _ in range(10):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            result = minimize(
                fun=lambda x: -self._expected_improvement(x),
                x0=x0,
                bounds=self.bounds,
                method="L-BFGS-B",
            )

            ei = -result.fun
            if ei > best_ei:
                best_ei = ei
                best_x = result.x

            return best_x
