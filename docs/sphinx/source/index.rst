.. This is the official documents of PyPop7 (Pure-PYthon library of POPulation-based black-box OPtimization).

Welcome to PyPop7's Documentation!
==================================

.. image:: https://img.shields.io/badge/GitHub-PyPop7-red.svg
.. image:: https://img.shields.io/badge/PyPI-pypop7-yellowgreen.svg
.. image:: https://img.shields.io/badge/license-GNU%20GPL--v3.0-green.svg
.. image:: https://readthedocs.org/projects/pypop/badge/?version=latest
.. image:: https://pepy.tech/badge/pypop7

**PyPop7** is a *Pure-PYthon* library of *POPulation-based OPtimization* for single-objective, real-parameter, black-box problems. Its main goal is to provide a *unified* interface and *elegant* implementations for **Black-Box Optimizers (BBO)**, *particularly population-based optimizers*, in order to facilitate research repeatability and also real-world applications.

More specifically, for alleviating the notorious **curse of dimensionality** of BBO (almost based on *iterative sampling*), the primary focus of PyPop7 is to cover their **State-Of-The-Art (SOTA) implementations for Large-Scale Optimization (LSO)**, though many of their other versions and variants are also included here (for benchmarking/mixing purpose, and sometimes even for practical purpose).

.. image:: images/logo.png
   :width: 321px
   :align: center

.. note::
   Now this library is still under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   design-philosophy
   es/es
   es/res
   es/ssaes
   es/dsaes
   es/csaes
   es/saes
   es/maes
   eda/eda
   eda/umda
   eda/emna
   ep/ep
   ep/cep
   ep/fep
   ds/ds
   ds/cs
   ds/hj
   ds/nm
   rs/rs
   rs/prs
   rs/rhc
   rs/arhc
   rs/srs
   sponsor
