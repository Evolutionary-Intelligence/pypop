Changing Log
============

Version: 0.0.78
---------------

* Add reference for online documentation of tutorials:

  * https://github.com/Evolutionary-Intelligence/DistributedEvolutionaryComputation

* Update `np.alltrue` to `np.all` for optimizer class **CCPSO2**:

  * https://github.com/Evolutionary-Intelligence/pypop/pull/177/commits/900c87353ac78ab27bf0f75f12a1267eb915ef69
  * https://numpy.org/devdocs/release/1.25.0-notes.html

    * `np.alltrue` is deprecated. Use `np.all` instead.

Version: 0.0.77
---------------

* Fix error of optimizer class **LAMCTS** owing to recent update of optimizer class **CMAES**:

  * https://github.com/Evolutionary-Intelligence/pypop/commit/108bba9b103a2da1e98961467037180717456070

Version: 0.0.76
---------------

* Add *early stopping* according to suggestion of `FiksII <https://github.com/FiksII>`_:

  * https://github.com/Evolutionary-Intelligence/pypop/issues/175
