Benchmarking Functions for BBO
==============================

In this open-source module, we provide a set of benchmarking/test functions which have been commonly used
in the **black-box/zeroth-order/gradient-free optimization** community. In the coming days, we are planning
to add some challenging optimization models from various **real-world applications**. Since this is a
long-term development Python project, welcome anyone to make open-source contributions.

For a set of 23 benchmarking/test functions, their **base** forms, **shifted/transformed** forms,
**rotated** forms, and **rotated-shifted** forms have been coded and well-tested. Typically, their
**rotated-shifted** forms should be employed in **Comparision Experiments** for BBO, in order to
avoid possible biasness towards certain search points (e.g., the origin) or separability.

Base Functions
--------------

.. autofunction:: pypop7.benchmarks.base_functions.sphere

.. autofunction:: pypop7.benchmarks.base_functions.cigar

.. autofunction:: pypop7.benchmarks.base_functions.discus

.. autofunction:: pypop7.benchmarks.base_functions.cigar_discus

.. autofunction:: pypop7.benchmarks.base_functions.ellipsoid

.. autofunction:: pypop7.benchmarks.base_functions.different_powers

.. autofunction:: pypop7.benchmarks.base_functions.schwefel221

.. autofunction:: pypop7.benchmarks.base_functions.step

.. autofunction:: pypop7.benchmarks.base_functions.schwefel222

.. autofunction:: pypop7.benchmarks.base_functions.rosenbrock

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
