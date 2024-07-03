from pypop7.benchmarks.pygmo import lennard_jones
from pypop7.optimizers.de.jade import JADE


def test_lennard_jones():
    results = lennard_jones(JADE, 1000)
    print(results)
    assert results['n_function_evaluations'] == 1000
