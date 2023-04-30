Software Summary
================

.. note:: This page is **actively** updated now, since many open-source software and code for black-box optimization
   are still missed here. We will happy to add it if you find some *important* work omitted here.

Python
------

* https://esa.github.io/pygmo2/ ( `pygmo <https://esa.github.io/pygmo2/>`_ is a **well-designed** and **well-maintained** Python library for **parallel optimization**. )

  * https://esa.github.io/pygmo2/algorithms.html#pygmo.de
  * https://esa.github.io/pygmo2/algorithms.html#pygmo.sade
  * https://esa.github.io/pygmo2/algorithms.html#pygmo.de1220
  * https://esa.github.io/pygmo2/algorithms.html#pygmo.pso
  * https://esa.github.io/pygmo2/algorithms.html#pygmo.pso_gen
  * https://esa.github.io/pygmo2/algorithms.html#pygmo.sea
  * https://esa.github.io/pygmo2/algorithms.html#pygmo.sga
  * https://esa.github.io/pygmo2/algorithms.html#pygmo.simulated_annealing
  * https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes
  * https://esa.github.io/pygmo2/algorithms.html#pygmo.xnes

* https://github.com/CMA-ES/pycma ( **CMA-ES** )

  * https://github.com/akimotolab/multi-fidelity ( **DD-CMA-ES** )

    * https://gist.github.com/youheiakimoto/08b95b52dfbf8832afc71dfff3aed6c8 ( **VD-CMA** )
    * https://gist.github.com/youheiakimoto/2fb26c0ace43c22b8f19c7796e69e108 ( **VKD-CMA** )
    * https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518 ( **DD-CMA-ES** )
    * https://github.com/akimotolab/CMAES_Tutorial ( **CMA-ES** )

  * https://github.com/CyberAgentAILab/cmaes ( **CMA-ES** )

    * https://github.com/c-bata/benchmark-warm-starting-cmaes ( **CMA-ES** )
    * https://github.com/EvoConJP/CMA-ES_with_Margin ( **CMA-ES** )

  * https://github.com/NiMlr/High-Dim-ES-RL ( **CMA-ES** )

  * https://github.com/optuna/optuna ( **CMA-ES** )

* https://github.com/dietmarwo/fast-cma-es ( **CMA-ES** )

* https://github.com/google/evojax

* https://github.com/google/vizier

* https://github.com/fmfn/BayesianOptimization
* https://github.com/ljvmiranda921/pyswarms (PSO)
* https://github.com/mradaideh/neorl
* https://github.com/nnaisense/evotorch (https://evotorch.ai/)

* https://github.com/RobertTLange/evosax

* https://pymoo.org/ (`pymoo <https://pymoo.org/>`_ *offers state of the art single- and multi-objective optimization algorithms*.)

  * https://pymoo.org/algorithms/soo/cmaes.html
  * https://pymoo.org/algorithms/soo/de.html
  * https://pymoo.org/algorithms/soo/es.html
  * https://pymoo.org/algorithms/soo/ga.html
  * https://pymoo.org/algorithms/soo/g3pcx.html
  * https://pymoo.org/algorithms/soo/isres.html
  * https://pymoo.org/algorithms/soo/nelder.html
  * https://pymoo.org/algorithms/soo/pattern.html
  * https://pymoo.org/algorithms/soo/pso.html
  * https://pymoo.org/algorithms/soo/sres.html

* https://scipy.org/

  * https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
  * https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
  * https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html
  * https://docs.scipy.org/doc/scipy/reference/optimize.minimize_scalar-brent.html

Some interesting code snapshots involving population-based methods are shown below:

* https://github.com/apourchot/CEM-RL (CEM)
* https://github.com/facebookresearch/LA-MCTS (BO/CMA-ES)

  * https://github.com/facebookresearch/LaMCTS

* https://github.com/huawei-noah/vega (AutoML)
* https://github.com/SimonBlanke/Gradient-Free-Optimizers (Discrete Optimization)

The below libraries seem to be not *actively* maintained (according to the last update time):

* https://github.com/blaa/PyGene (Now it is not actively maintained.)
* https://github.com/hardmaru/estool (Now it is not actively maintained.)
* https://github.com/HIPS/Spearmint (Now it is not actively maintained.)
* https://github.com/hyperopt/hyperopt (Now it is not actively maintained.)
* https://github.com/LDNN97/Evolutionary-Optimization-Algorithms (Now it is not actively maintained.)
* https://github.com/pybrain/pybrain (Now it is not actively maintained.)

  * https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/fem.py
  * https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/nes.py
  * https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/rank1.py
  * https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/snes.py
  * https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/ves.py
  * https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/xnes.py
  * https://github.com/chanshing/xnes
* https://github.com/scikit-optimize/scikit-optimize (Now it is not actively maintained.)
* https://github.com/strongio/evolutionary-optimization (Now it is not actively maintained.)

R
-

https://cran.r-project.org/web/views/Optimization.html

* https://cran.r-project.org/web/packages/adagio/index.html (NM/HJ)
* https://cran.r-project.org/web/packages/CEoptim/index.html (CEM)
* https://cran.r-project.org/web/packages/cmaes/index.html (CMA-ES)
* https://cran.r-project.org/web/packages/DEoptim/index.html (DE)
* https://cran.r-project.org/web/packages/DEoptimR/index.html (JDE)
* https://cran.r-project.org/web/packages/GA/index.html (GA)
* https://cran.r-project.org/web/packages/genalg/index.html (GA)
* https://cran.r-project.org/web/packages/GenSA/index.html (SA)
* https://cran.r-project.org/web/packages/neldermead/index.html (NM)
* https://cran.r-project.org/web/packages/nloptr/index.html
* https://cran.r-project.org/web/packages/NMOF/index.html (DE/GA/PSO/SA)
* https://cran.r-project.org/web/packages/pso/index.html (PSO)
* https://cran.r-project.org/web/packages/RCEIM/index.html (CEM)
* https://cran.r-project.org/web/packages/rCMA/index.html (CMA-ES)
* https://cran.r-project.org/web/packages/rgenoud/index.html (GA)
* https://github.com/hzambran/hydroPSO (Now it is not actively maintained.)
* https://github.com/jakobbossek/ecr2

`IOHanalyzer <https://github.com/IOHprofiler/IOHanalyzer>`_ is *a performance analyzer for
Iterative Optimization Heuristics (IOHs)*.

Matlab
------

* https://cse-lab.seas.harvard.edu/cse-lab-software (Now it is not actively maintained.)

  * https://gitlab.ethz.ch/mavt-cse/cma-es

* https://divis-gmbh.de/es-software/ (ES)

  * The Octave source code (proprietary implementations) can be downloaded only for non-commercial use.

* https://github.com/blockchain-group/DIRECTGO
* https://github.com/ProbabilisticNumerics/entropy-search (ESEGO)
* https://people.idsia.ch/~sun/enes.rar (ENES)

C
-

* https://github.com/CMA-ES/c-cmaes (Now it is not actively maintained.)
* https://www.egr.msu.edu/~kdeb/codes/g3pcx/g3pcx.tar (G3PCX)

C++
---

* https://github.com/chgagne/beagle (Now it is not actively maintained.)
* https://github.com/CMA-ES/libcmaes (CMA-ES)
* https://github.com/Shark-ML/Shark (Now it is not actively maintained.)

  * https://github.com/Shark-ML/Shark/blob/master/include/shark/Algorithms/DirectSearch/VDCMA.h (VD-CMA)
  * https://github.com/Shark-ML/Shark/blob/master/include/shark/Algorithms/DirectSearch/LMCMA.h (LM-CMA)

* https://www.cs.wm.edu/~va/software/DirectSearch/direct_code/

Java
----

* https://github.com/GMUEClab/ecj (https://cs.gmu.edu/~eclab/projects/ecj/)
* https://github.com/sdarg/opt4j/ (https://sdarg.github.io/opt4j/)
* https://www.isa.us.es/fom/modules/portalWFInterface/init.php (Discrete Optimization)
* https://jmetal.sourceforge.net/ (Now it is not actively maintained.)
* http://www.jamesframework.org/ (Now it is not actively maintained.)
* https://github.com/dwdyer/watchmaker (Now it is not actively maintained.)
* https://github.com/jenetics/jenetics (GA/GP)

C#
--

* https://github.com/heal-research/HeuristicLab (https://dev.heuristiclab.com/trac.fcgi/wiki)

Others
------

`https://github.com/CMA-ES <https://github.com/CMA-ES>`_ is *a collection of various implementations of
the powerful CMA-ES algorithm*.
  * https://github.com/CMA-ES/c-cmaes (C)
  * https://github.com/CMA-ES/libcmaes (C++)
  * https://github.com/CMA-ES/pycma (Python)

https://nlopt.readthedocs.io/en/latest/

* https://coin-or.github.io/Ipopt/
* http://zhar.net/howto/html/ (Now it is not actively maintained.)
* https://sop.tik.ee.ethz.ch/pisa/principles.html (Now it is not actively maintained.)
