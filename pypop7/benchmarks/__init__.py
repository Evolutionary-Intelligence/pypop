"""This Python module covers a set of benchmark functions.
    Here we do NOT include the *testing* code for simplicity.

    For online documentations of benchmark functions, please refer to:
        https://pypop.readthedocs.io/en/latest/benchmarks.html
"""
import pypop7.benchmarks.cases  # *test* cases for all benchmarking functions (based on sampling)
# utilities for saving, reading, plotting, and analyzing optimization results
import pypop7.benchmarks.utils
import pypop7.benchmarks.base_functions  # base forms of benchmarking functions
import pypop7.benchmarks.shifted_functions  # shift/transform forms of benchmarking functions
import pypop7.benchmarks.rotated_functions  # rotation forms of benchmarking functions
import pypop7.benchmarks.continuous_functions  # rotation-shift forms of benchmarking functions
import pypop7.benchmarks.data_science  # loss functions from data science
# NeverGrad: https://github.com/facebookresearch/nevergrad
import pypop7.benchmarks.never_grad  # photonics model from *NeverGrad*
# PyGMO: https://esa.github.io/pygmo2/install.html
# import pypop7.benchmarks.pygmo  # optional owing to its specific installation way
