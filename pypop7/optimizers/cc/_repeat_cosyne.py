"""Repeat the following library for `COSYNE`:
    https://evotorch.ai/

    The referenced paper can be found by:
    F. Gomez, J. Schmidhuber, R. Miikkulainen
    Accelerated Neural Evolution through Cooperatively Coevolved Synapses
    https://jmlr.org/papers/v9/gomez08a.html

    Luckily our code could repeat the performance of the EvoTroch *well*.
    Therefore, we argue that the repeatability of `COSYNE` could be **well-documented**.
"""
import torch
from evotorch import Problem
from evotorch.algorithms import ga
from pypop7.benchmarks.base_functions import ellipsoid
from evotorch.logging import StdOutLogger


def myellipsoid(x: torch.Tensor) -> torch.Tensor:
    y = []
    for i in range(len(x)):
        y.append(ellipsoid(x[i].numpy()))
    return torch.tensor(y)


problem = Problem(
    "min",
    myellipsoid,
    initial_bounds=(-5.0, 5.0),
    solution_length=10,
    vectorized=True
)
searcher = ga.Cosyne(problem, popsize=20, tournament_size=2, mutation_probability=0.3, mutation_stdev=1.0)
_ = StdOutLogger(searcher, interval=50)

searcher.run(num_generations=3000)
