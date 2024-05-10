Changing Log
============

Version: 0.0.80
---------------

* Add a new optimizer class called Simultaneous Perturbation Stochastic Approximation (**SPSA**):

  * https://github.com/jgomezdans/spsa

Version: 0.0.79
---------------

* Remove the installation dependency on `torch` according to suggestion of `TimOrtkamp <https://github.com/TimOrtkamp>`_:

  * https://github.com/Evolutionary-Intelligence/pypop/discussions/181

* Add a new case for online documentation of applications:

  * https://openreview.net/forum?id=eQerjHehcM

Version: 0.0.78
---------------

* Fix errors of both optimizer classes (**HCC** and **COCMA**) owing to recent update of optimizer class **CMAES**:

  * https://github.com/Evolutionary-Intelligence/pypop/commit/0ffa8e0671f40a714f9294d85490b6b654bf4b16  (for **HCC**)
  * https://github.com/Evolutionary-Intelligence/pypop/commit/95d9c53dc0c4898cc73b13b229f8072825f78a24  (for **COCMA**)

* Add reference for online documentation of tutorials:

  * https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation

* Update `np.alltrue` to `np.all` for optimizer class **CCPSO2**:

  * https://github.com/Evolutionary-Intelligence/pypop/pull/177/commits/900c87353ac78ab27bf0f75f12a1267eb915ef69
  * https://numpy.org/devdocs/release/1.25.0-notes.html

    * `np.alltrue` is deprecated. Use `np.all` instead.

* Add a new case for online documentation of applications:

  * https://github.com/jeancroy/RP-fit

Version: 0.0.77
---------------

* Fix error of optimizer class **LAMCTS** owing to recent update of optimizer class **CMAES**:

  * https://github.com/Evolutionary-Intelligence/pypop/commit/108bba9b103a2da1e98961467037180717456070

Version: 0.0.76
---------------

* Add *early stopping* according to suggestion of `FiksII <https://github.com/FiksII>`_:

  * https://github.com/Evolutionary-Intelligence/pypop/issues/175
