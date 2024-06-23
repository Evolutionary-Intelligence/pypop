Util Functions for BBO
======================

In this open-source module, we provide some **common** utils functions for BBO, as presented below:

Plot 2-D Fitness Landscape
--------------------------

.. autofunction:: pypop7.benchmarks.utils.generate_xyz

.. autofunction:: pypop7.benchmarks.utils.plot_contour

The online figure generated in the above Example is shown below:

.. image:: images/contour_ellipsoid.png
   :width: 321px
   :align: center

Plot 3-D Fitness Landscape
--------------------------

.. autofunction:: pypop7.benchmarks.utils.plot_surface

The online figure generated in the above Example is shown below:

.. image:: images/surface_ellipsoid.png
   :width: 321px
   :align: center

Save Optimization Results via Object Serialization
--------------------------------------------------

For **object serialization**, we use the standard library (`pickle
<https://docs.python.org/3/library/pickle.html>`_) of Python.

.. autofunction:: pypop7.benchmarks.utils.save_optimization

Check Optimization Results
--------------------------

Plot Convergence Curves via Matplotlib
--------------------------------------

Compare Multiple Optimizers
---------------------------

Accelerate Computation via Numba
--------------------------------
