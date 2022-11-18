Tutorials
=========

Here we provide several interesting examples to help `newbie` better use this library by studying and running them.

* Lens Shape Optimization (15-dimensional)

For all optimizers from this library `PyPop7`, we also provide a *toy* example on their corresponding
`API <https://pypop.readthedocs.io/_/downloads/en/latest/pdf/>`_ documentations.

Lens Shape Optimization
-----------------------

.. image:: images/lens_optimization.gif
   :width: 321px
   :align: center

The above figure shows the evolution of lens shape,
optimized by `MAES <https://pypop.readthedocs.io/en/latest/es/maes.html>`_.

The objective of Lens Shape Optimization is to find the optimal shape of glass body such that parallel incident light
rays are concentrated in a given point on a plane while using a minimum of glass material possible.
Refer to `Beyer, 2020, GECCO <https://dl.acm.org/doi/abs/10.1145/3377929.3389870>`_ for more details.

To repeat this interesting figure, please run the following code:
https://github.com/Evolutionary-Intelligence/pypop/blob/main/tutorials/lens_optimization.py.
