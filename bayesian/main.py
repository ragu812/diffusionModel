from typing import List, Tuple

import torch
<<<<<<< HEAD
from botorch.fit import fit_gpytorch_model
=======
>>>>>>> 35e1a81af236fd72e7d610c39df6d47ebcee00a5
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class BayesianLDM:
    def __init__(self, bounds: List[Tuple[float, float]], n_iterations: int):
        print("Printing BO within the bounds")

        self.bounds = torch.tensor(bounds, dtype=torch.float64, device=device).t()
        self.n_params = len(bounds)
        self.n_iterations = n_iterations

        self.x_observed = []
        self.y_observed = []

        self.best_params = None
        self.best_score = -float("inf")

    def suggest(self) -> List[float]:
        if len(self.x_observed) < 2:
            params = (
                torch.rand(self.n_params, device=device)
                * (self.bounds[1] - self.bounds[0])
                + self.bounds[0]
            )
            print(f"\n The suggested points from BO {params.tolist()}")
            return params.tolist()

        train_x = torch.tensor(self.x_observed, dtype=torch.float64, device=device)
        train_y = torch.tensor(
            self.y_observed, dtype=torch.float64, device=device
        ).unsqueeze(-1)

        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        ei = ExpectedImprovement(model, best_f=train_y.max())
        candidate, _ = optimize_acqf(
            ei,
            bounds=self.bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )

        return candidate.squeeze().tolist()

    def observe(self, params: List[float], score: float):
        print(f"The observed Parameters = {params}, and Score = {score}")

        self.x_observed.append(params)
        self.y_observed.append(score)

        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            print(f"The new best score = {score}")

    def get_best(self) -> Tuple[List[float], float]:
        if self.best_params is None:
            return [0.0] * self.n_params, -float("inf")
        return self.best_params, float(self.best_score)
