"""This Python module covers a set of benchmark functions. For online documentations of all
    benchmark functions, please refer to: https://pypop.readthedocs.io/en/latest/benchmarks.html
"""
# for all functions with only two dimensions, mainly for visualization
import pypop7.benchmarks.ndim_2
# *test* cases for all artificially-constructed benchmarking functions
import pypop7.benchmarks.cases
# utilities for saving, reading, plotting, and analyzing optimization results
import pypop7.benchmarks.utils
# base forms of artificially-constructed benchmarking functions
import pypop7.benchmarks.base_functions
# shift/transform forms of artificially-constructed benchmarking functions
import pypop7.benchmarks.shifted_functions
# rotation forms of artificially-constructed benchmarking functions
import pypop7.benchmarks.rotated_functions
# rotation-shift forms of artificially-constructed benchmarking functions
import pypop7.benchmarks.continuous_functions
# a set of common loss functions from data science
import pypop7.benchmarks.data_science
# NeverGrad: https://github.com/facebookresearch/nevergrad
import pypop7.benchmarks.never_grad  # photonics model from *NeverGrad*
# PyGMO: https://esa.github.io/pygmo2/install.html
# import pypop7.benchmarks.pygmo  # optional owing to its specific installation way
