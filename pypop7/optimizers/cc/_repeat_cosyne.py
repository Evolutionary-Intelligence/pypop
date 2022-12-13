"""
    import torch
    from evotorch import Problem
    from evotorch.algorithms import ga
    from evotorch.logging import StdOutLogger

    def norm(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x, dim=-1)

    problem = Problem('min', norm, initial_bounds=(-5.0, 5.0), solution_length=10)
    searcher = ga.Cosyne(problem, popsize=100, tournament_size=10, mutation_probability=1.0, mutation_stdev=1.0)
    logger = StdOutLogger(searcher)
    searcher.run(num_generations=3000)
"""
