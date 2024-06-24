Benchmarking Functions for BBO
==============================

In this open-source module, we provide a set of benchmarking/test functions which have been commonly used
in the **black-box/zeroth-order/gradient-free optimization** community. In the coming days, we are planning
to add some challenging optimization models from various **real-world applications**. Since this is a
long-term development Python project, welcome anyone to make open-source contributions.

For a set of 23 benchmarking/test functions, their **base** forms, **shifted/transformed** forms,
**rotated** forms, and **rotated-shifted** forms have been coded and well-tested. Typically, their
**rotated-shifted** forms should be employed in **Comparison Experiments** for BBO, in order to
avoid possible biases towards certain search points (e.g., the origin) or separability.

Checking of Coding Correctness
------------------------------

For all testing Python code, please refer to the following links for all details:

* `for base forms <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/benchmarks/test_base_functions.py>`_
* `for shifted/transformed forms <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/benchmarks/test_shifted_functions.py>`_
* `for rotated forms <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/benchmarks/test_rotated_functions.py>`_
* `for rotated-shifted forms <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/benchmarks/test_continuous_functions.py>`_

Base Functions
--------------

In the following, we will introduce **base** forms of common benchmarking functions,
as presented below:

.. autofunction:: pypop7.benchmarks.base_functions.sphere

* Jastrebski, G.A. and Arnold, D.V., 2006, July. Improving evolution strategies through active covariance matrix adaptation. In IEEE International Conference on Evolutionary Computation (pp. 2814-2821). IEEE.

.. autofunction:: pypop7.benchmarks.base_functions.cigar

* Jastrebski, G.A. and Arnold, D.V., 2006, July. Improving evolution strategies through active covariance matrix adaptation. In IEEE International Conference on Evolutionary Computation (pp. 2814-2821). IEEE.

.. autofunction:: pypop7.benchmarks.base_functions.discus

* Jastrebski, G.A. and Arnold, D.V., 2006, July. Improving evolution strategies through active covariance matrix adaptation. In IEEE International Conference on Evolutionary Computation (pp. 2814-2821). IEEE.

.. autofunction:: pypop7.benchmarks.base_functions.cigar_discus

* Jastrebski, G.A. and Arnold, D.V., 2006, July. Improving evolution strategies through active covariance matrix adaptation. In IEEE International Conference on Evolutionary Computation (pp. 2814-2821). IEEE.

.. autofunction:: pypop7.benchmarks.base_functions.ellipsoid

* Jastrebski, G.A. and Arnold, D.V., 2006, July. Improving evolution strategies through active covariance matrix adaptation. In IEEE International Conference on Evolutionary Computation (pp. 2814-2821). IEEE.

.. autofunction:: pypop7.benchmarks.base_functions.different_powers

* Jastrebski, G.A. and Arnold, D.V., 2006, July. Improving evolution strategies through active covariance matrix adaptation. In IEEE International Conference on Evolutionary Computation (pp. 2814-2821). IEEE.

.. autofunction:: pypop7.benchmarks.base_functions.schwefel221

.. autofunction:: pypop7.benchmarks.base_functions.step

.. autofunction:: pypop7.benchmarks.base_functions.schwefel222

.. autofunction:: pypop7.benchmarks.base_functions.rosenbrock

* Jastrebski, G.A. and Arnold, D.V., 2006, July. Improving evolution strategies through active covariance matrix adaptation. In IEEE International Conference on Evolutionary Computation (pp. 2814-2821). IEEE.

.. autofunction:: pypop7.benchmarks.base_functions.schwefel12

.. autofunction:: pypop7.benchmarks.base_functions.exponential

.. autofunction:: pypop7.benchmarks.base_functions.griewank

.. autofunction:: pypop7.benchmarks.base_functions.bohachevsky

.. autofunction:: pypop7.benchmarks.base_functions.ackley

.. autofunction:: pypop7.benchmarks.base_functions.rastrigin

.. autofunction:: pypop7.benchmarks.base_functions.scaled_rastrigin

.. autofunction:: pypop7.benchmarks.base_functions.skew_rastrigin

.. autofunction:: pypop7.benchmarks.base_functions.levy_montalvo

.. autofunction:: pypop7.benchmarks.base_functions.michalewicz

.. autofunction:: pypop7.benchmarks.base_functions.salomon

.. autofunction:: pypop7.benchmarks.base_functions.shubert

.. autofunction:: pypop7.benchmarks.base_functions.schaffer

Shifted/Transformed Forms
-------------------------

In the following, we will introduce **shifted/transformed** forms of the above base functions,
as presented below:

.. autofunction:: pypop7.benchmarks.shifted_functions.generate_shift_vector

.. autofunction:: pypop7.benchmarks.shifted_functions.load_shift_vector

.. autofunction:: pypop7.benchmarks.shifted_functions.sphere

Rotated Forms
-------------

In the following, we will introduce **rotated** forms of the above base functions,
as presented below:

.. autofunction:: pypop7.benchmarks.rotated_functions.generate_rotation_matrix

.. autofunction:: pypop7.benchmarks.rotated_functions.load_rotation_matrix

Rotated-Shifted Forms
---------------------

In the following, we will introduce **rotated-shifted** forms of the above base functions,
as presented below:

Benchmarking for Large-Scale BBO (LBO)
--------------------------------------

Here we provide two different benchmarking cases (**local vs global search**) for large-scale
black-box optimization (LBO):

.. autofunction:: pypop7.benchmarks.lbo.benchmark_local_search

.. autofunction:: pypop7.benchmarks.lbo.benchmark_global_search

Test Classes and Data
---------------------

Here we provide a set of test classes and test data for benchmarking functions.
Note that these are used only for the *testing* purpose.

.. autoclass:: pypop7.benchmarks.cases.Cases
   :members:

.. autofunction:: pypop7.benchmarks.cases.get_y_sphere

.. autofunction:: pypop7.benchmarks.cases.get_y_cigar

.. autofunction:: pypop7.benchmarks.cases.get_y_discus

.. autofunction:: pypop7.benchmarks.cases.get_y_cigar_discus

.. autofunction:: pypop7.benchmarks.cases.get_y_ellipsoid

.. autofunction:: pypop7.benchmarks.cases.get_y_different_powers

.. autofunction:: pypop7.benchmarks.cases.get_y_schwefel221

.. autofunction:: pypop7.benchmarks.cases.get_y_step

.. autofunction:: pypop7.benchmarks.cases.get_y_schwefel222

.. autofunction:: pypop7.benchmarks.cases.get_y_rosenbrock

.. autofunction:: pypop7.benchmarks.cases.get_y_schwefel12

.. autofunction:: pypop7.benchmarks.cases.get_y_griewank

.. autofunction:: pypop7.benchmarks.cases.get_y_bohachevsky

.. autofunction:: pypop7.benchmarks.cases.get_y_ackley

.. autofunction:: pypop7.benchmarks.cases.get_y_rastrigin
