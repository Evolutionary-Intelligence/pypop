Benchmarking Functions for BBO
==============================

In this open-source Python module, we have provided a set of **benchmarking/test functions**
which have been commonly used in the **black-box/zeroth-order/gradient-free/derivative-free
optimization** community. 

.. Note :: In the coming days, we are planning to add some challenging BBO models from various
   **real-world applications**. Since this is a *long-term* development project, welcome
   anyone to make open-source contributions to it.

For a set of 23 benchmarking/test functions, their **base** forms, **shifted/transformed**
forms, **rotated** forms, and **rotated-shifted** forms have been coded and *well-tested*.
Typically, their **rotated-shifted** forms should be employed in **Comparison Experiments**
for BBO, in order to avoid possible biases towards certain search points (e.g., the origin)
or separability.

Checking of Coding Correctness
------------------------------

For all testing Python code of benchmarking functions, please refer to the following openly
accessible www links for details (In fact, we have spent much time in checking of coding
correctness):

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

In the following, we will introduce **shifted/transformed** forms of the above
`base functions <https://pypop.readthedocs.io/en/latest/benchmarks.html#base-functions>`_,
as presented below:

.. autofunction:: pypop7.benchmarks.shifted_functions.generate_shift_vector

.. autofunction:: pypop7.benchmarks.shifted_functions.load_shift_vector

.. autofunction:: pypop7.benchmarks.shifted_functions.sphere

.. autofunction:: pypop7.benchmarks.shifted_functions.cigar

.. autofunction:: pypop7.benchmarks.shifted_functions.discus

.. autofunction:: pypop7.benchmarks.shifted_functions.cigar_discus

.. autofunction:: pypop7.benchmarks.shifted_functions.ellipsoid

.. autofunction:: pypop7.benchmarks.shifted_functions.different_powers

.. autofunction:: pypop7.benchmarks.shifted_functions.schwefel221

.. autofunction:: pypop7.benchmarks.shifted_functions.step

.. autofunction:: pypop7.benchmarks.shifted_functions.schwefel222

.. autofunction:: pypop7.benchmarks.shifted_functions.rosenbrock

.. autofunction:: pypop7.benchmarks.shifted_functions.schwefel12

.. autofunction:: pypop7.benchmarks.shifted_functions.exponential

.. autofunction:: pypop7.benchmarks.shifted_functions.griewank

.. autofunction:: pypop7.benchmarks.shifted_functions.bohachevsky

.. autofunction:: pypop7.benchmarks.shifted_functions.ackley

.. autofunction:: pypop7.benchmarks.shifted_functions.rastrigin

.. autofunction:: pypop7.benchmarks.shifted_functions.scaled_rastrigin

.. autofunction:: pypop7.benchmarks.shifted_functions.skew_rastrigin

.. autofunction:: pypop7.benchmarks.shifted_functions.levy_montalvo

.. autofunction:: pypop7.benchmarks.shifted_functions.michalewicz

.. autofunction:: pypop7.benchmarks.shifted_functions.salomon

.. autofunction:: pypop7.benchmarks.shifted_functions.shubert

.. autofunction:: pypop7.benchmarks.shifted_functions.schaffer

Rotated Forms
-------------

In the following, we will introduce **rotated** forms of the above
`base functions <https://pypop.readthedocs.io/en/latest/benchmarks.html#base-functions>`_,
as presented below:

.. autofunction:: pypop7.benchmarks.rotated_functions.generate_rotation_matrix

.. autofunction:: pypop7.benchmarks.rotated_functions.load_rotation_matrix

.. autofunction:: pypop7.benchmarks.rotated_functions.generate_shift_vector

.. autofunction:: pypop7.benchmarks.rotated_functions.load_shift_vector

.. autofunction:: pypop7.benchmarks.rotated_functions.sphere

.. autofunction:: pypop7.benchmarks.rotated_functions.cigar

.. autofunction:: pypop7.benchmarks.rotated_functions.discus

.. autofunction:: pypop7.benchmarks.rotated_functions.cigar_discus

.. autofunction:: pypop7.benchmarks.rotated_functions.ellipsoid

.. autofunction:: pypop7.benchmarks.rotated_functions.different_powers

.. autofunction:: pypop7.benchmarks.rotated_functions.schwefel221

.. autofunction:: pypop7.benchmarks.rotated_functions.step

.. autofunction:: pypop7.benchmarks.rotated_functions.schwefel222

.. autofunction:: pypop7.benchmarks.rotated_functions.rosenbrock

.. autofunction:: pypop7.benchmarks.rotated_functions.schwefel12

.. autofunction:: pypop7.benchmarks.rotated_functions.exponential

.. autofunction:: pypop7.benchmarks.rotated_functions.griewank

.. autofunction:: pypop7.benchmarks.rotated_functions.bohachevsky

.. autofunction:: pypop7.benchmarks.rotated_functions.ackley

.. autofunction:: pypop7.benchmarks.rotated_functions.rastrigin

.. autofunction:: pypop7.benchmarks.rotated_functions.scaled_rastrigin

.. autofunction:: pypop7.benchmarks.rotated_functions.skew_rastrigin

.. autofunction:: pypop7.benchmarks.rotated_functions.levy_montalvo

.. autofunction:: pypop7.benchmarks.rotated_functions.michalewicz

.. autofunction:: pypop7.benchmarks.rotated_functions.salomon

.. autofunction:: pypop7.benchmarks.rotated_functions.shubert

.. autofunction:: pypop7.benchmarks.rotated_functions.schaffer

Rotated-Shifted Forms
---------------------

In the following, we will introduce **rotated-shifted** forms of the above
`base functions <https://pypop.readthedocs.io/en/latest/benchmarks.html#base-functions>`_,
as presented below:

.. autofunction:: pypop7.benchmarks.continuous_functions.load_shift_and_rotation

.. autofunction:: pypop7.benchmarks.continuous_functions.sphere

.. autofunction:: pypop7.benchmarks.continuous_functions.cigar

.. autofunction:: pypop7.benchmarks.continuous_functions.discus

.. autofunction:: pypop7.benchmarks.continuous_functions.cigar_discus

.. autofunction:: pypop7.benchmarks.continuous_functions.ellipsoid

.. autofunction:: pypop7.benchmarks.continuous_functions.different_powers

.. autofunction:: pypop7.benchmarks.continuous_functions.schwefel221

.. autofunction:: pypop7.benchmarks.continuous_functions.step

.. autofunction:: pypop7.benchmarks.continuous_functions.schwefel222

.. autofunction:: pypop7.benchmarks.continuous_functions.rosenbrock

.. autofunction:: pypop7.benchmarks.continuous_functions.schwefel12

.. autofunction:: pypop7.benchmarks.continuous_functions.exponential

.. autofunction:: pypop7.benchmarks.continuous_functions.griewank

.. autofunction:: pypop7.benchmarks.continuous_functions.bohachevsky

.. autofunction:: pypop7.benchmarks.continuous_functions.ackley

.. autofunction:: pypop7.benchmarks.continuous_functions.rastrigin

.. autofunction:: pypop7.benchmarks.continuous_functions.scaled_rastrigin

.. autofunction:: pypop7.benchmarks.continuous_functions.skew_rastrigin

.. autofunction:: pypop7.benchmarks.continuous_functions.levy_montalvo

.. autofunction:: pypop7.benchmarks.continuous_functions.michalewicz

.. autofunction:: pypop7.benchmarks.continuous_functions.salomon

.. autofunction:: pypop7.benchmarks.continuous_functions.shubert

.. autofunction:: pypop7.benchmarks.continuous_functions.schaffer

Benchmarking for Large-Scale BBO (LBO)
--------------------------------------

Here we provide two different benchmarking cases (**local vs global search**) for large-scale
black-box optimization (LBO):

.. autoclass:: pypop7.benchmarks.lbo.Experiment

.. autoclass:: pypop7.benchmarks.lbo.Experiments

.. autofunction:: pypop7.benchmarks.lbo.benchmark_local_search

.. autofunction:: pypop7.benchmarks.lbo.benchmark_global_search

Black-Box Classification from Data Science
------------------------------------------

Here we provide a family of **black-box classifications** from data science:

.. autofunction:: pypop7.benchmarks.data_science.cross_entropy_loss_lr

.. autofunction:: pypop7.benchmarks.data_science.cross_entropy_loss_l2

.. autofunction:: pypop7.benchmarks.data_science.square_loss_lr

.. autofunction:: pypop7.benchmarks.data_science.logistic_loss_lr

.. autofunction:: pypop7.benchmarks.data_science.logistic_loss_l2

.. autofunction:: pypop7.benchmarks.data_science.tanh_loss_lr

.. autofunction:: pypop7.benchmarks.data_science.hinge_loss_perceptron

.. autofunction:: pypop7.benchmarks.data_science.loss_margin_perceptron

.. autofunction:: pypop7.benchmarks.data_science.loss_svm

.. autofunction:: pypop7.benchmarks.data_science.mpc2023_nonsmooth

.. autofunction:: pypop7.benchmarks.data_science.read_parkinson_disease_classification

.. autofunction:: pypop7.benchmarks.data_science.read_semeion_handwritten_digit

.. autofunction:: pypop7.benchmarks.data_science.read_cnae9

.. autofunction:: pypop7.benchmarks.data_science.read_madelon

.. autofunction:: pypop7.benchmarks.data_science.read_qsar_androgen_receptor

Benchmarking on Photonics Models from NeverGrad
-----------------------------------------------

Please refer to `NeverGrad <https://github.com/facebookresearch/nevergrad>`_ for an
introduction.

.. autofunction:: pypop7.benchmarks.never_grad.benchmark_photonics

Benchmarking of Controllers on Gymnasium
----------------------------------------

Please refer to `Gymnasium <https://gymnasium.farama.org/>`_ for an
introduction (from `Farama Foundation <https://farama.org/>`_).

.. autoclass:: pypop7.benchmarks.gymnasium.Cartpole
   :members:

Lennard-Jones Cluster Optimization from PyGMO
---------------------------------------------

Please refer to `pagmo2 <https://esa.github.io/pagmo2/docs/cpp/problems/lennard_jones.html>`_
for an introduction (from `European Space Agency <https://www.esa.int/>`_) to this 444-d
Lennard-Jones cluster optimization problem from `PyGMO <https://esa.github.io/pygmo2/>`_.

.. autofunction:: pypop7.benchmarks.pygmo.lennard_jones

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
