import time

from pypop7.benchmarks.lbo import benchmark_local_search
from pypop7.optimizers.rs.prs import PRS as Optimizer


def test_benchmark_local_search():
    start_runtime = time.time()
    benchmark_local_search(Optimizer, 2, 3.0, 1, 2)
    print('Total runtime: {:7.5e}.'.format(time.time() - start_runtime))
