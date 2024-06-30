from pypop7.optimizers.rs.prs import PRS as Optimizer
from pypop7.benchmarks.never_grad import benchmark_photonics


def test_benchmark_nevergrad():
    results = benchmark_photonics(Optimizer)
    assert results['best_so_far_y'] < 1.0
