import numpy as np

import shifted_functions as sf
from shifted_functions import generate_shift_vector


def generate_shift_vectors(functions=None, ndims=None, seed=None):
    if functions is None:
        functions = [sf.sphere, sf.cigar, sf.discus, sf.cigar_discus,
                     sf.ellipsoid, sf.different_powers, sf.schwefel221, sf.step,
                     sf.schwefel222, sf.rosenbrock, sf.exponential, sf.schwefel12]
    if ndims is None:
        ndims = [2, 10, 100, 200, 1000, 2000]
    if seed is None:
        seed = 20211207

    rng = np.random.default_rng(seed)
    seeds = rng.integers(np.iinfo(np.int64).max, size=(len(functions), len(ndims)))

    for i, f in enumerate(functions):
        for j, d in enumerate(ndims):
            generate_shift_vector(f, d, -9.5, 9.5, seeds[i, j])


if __name__ == '__main__':
    generate_shift_vectors()
