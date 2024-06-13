Some (Rather All) Papers on Black-Box Optimization (BBO)
========================================================

Mathematical Theories for BBO
-----------------------------

* Agarwal, A., Bartlett, P.L., Ravikumar, P. and Wainwright, M.J., 2012. Information-theoretic lower bounds on the oracle complexity of stochastic convex optimization. IEEE Transactions on Information Theory, 58(5), pp.3235-3249.

Random Search (RS)
------------------

* Sel, B., Tawaha, A., Ding, Y., Jia, R., Ji, B., Lavaei, J. and Jin, M., 2023, June. Learning-to-Learn to Guide Random Search: Derivative-Free Meta Blackbox Optimization on Manifold. In Learning for Dynamics and Control Conference (pp. 38-50). PMLR.
* Li, L. and Talwalkar, A., 2020. Random search and reproducibility for neural architecture search. In Uncertainty in Artificial Intelligence (pp. 367-377). PMLR.
* Sener, O. and Koltun, V., 2019, September. `Learning to guide random search <https://openreview.net/forum?id=B1gHokBKwS>`_. In International Conference on Learning Representations (ICLR).
* Chechkin, A. and Sokolov, I., 2018. Random search with resetting: A unified renewal approach. Physical Review Letters, 121(5), p.050601.
* Falcón-Cortés, A., Boyer, D., Giuggioli, L. and Majumdar, S.N., 2017. Localization transition induced by learning in random searches. Physical Review Letters, 119(14), p.140603.
* Chupeau, M., Bénichou, O. and Voituriez, R., 2015. Cover times of random searches. Nature Physics, 11(10), pp.844-847.
* Qi, Y., Mao, X., Lei, Y., Dai, Z. and Wang, C., 2014, May. The strength of random search on automated program repair. In Proceedings of International Conference on Software Engineering (pp. 254-265). IEEE.
* Hein, A.M. and McKinley, S.A., 2012. Sensing and decision-making in random search. Proceedings of the National Academy of Sciences, 109(30), pp.12070-12074.
* Tejedor, V., Voituriez, R. and Bénichou, O., 2012. Optimizing persistent random searches. Physical Review Letters, 108(8), p.088103.
* Zabinsky, Z.B., 2003. Stochastic adaptive search for global optimization. Springer Science & Business Media.
* Viswanathan, G.M., Buldyrev, S.V., Havlin, S., Da Luz, M.G.E., Raposo, E.P. and Stanley, H.E., 1999. Optimizing the success of random searches. Nature, 401(6756), pp.911-914.
* Yakowitz, S. and Lugosi, E., 1990. Random search in the presence of noise, with application to machine learning. SIAM Journal on Scientific and Statistical Computing, 11(4), pp.702-712.
* Devroye, L.P., 1978. Progressive global random search of continuous functions. Mathematical Programming, 15(1), pp.330-342.
* Schrack, G. and Choit, M., 1976. Optimized relative step size random searches. Mathematical Programming, 10(1), pp.230-244. [ https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/rs.py ]
* Schumer, M.A. and Steiglitz, K., 1968. Adaptive step size random search. IEEE Transactions on Automatic Control, 13(3), pp.270-276. [ https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/rs.py ]
* Matyas, J., 1965. `Random optimization <https://archive.org/details/sim_automation-and-remote-control_1965-02_26_2/page/n1/mode/2up>`_. Automation and Remote control, 26(2), pp.246-253. [ https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/rs.py | `ru <https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=at&paperid=11288&option_lang=eng>`_ ]
* Karnopp, D.C., 1963. `Random search techniques for optimization problems <https://www.sciencedirect.com/science/article/abs/pii/0005109863900189>`_. Automatica, 1(2-3), pp.111-121. [ https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/rs.py ]
* Rastrigin, L.A., 1963. `The convergence of the random search method in the extremal control of a many parameter system <https://archive.org/details/sim_automation-and-remote-control_1963-11_24_11/mode/2up?view=theater>`_. Automaton & Remote Control, 24, pp.1337-1342. [ https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/rs.py ]
* Brooks, S.H., 1958. `A discussion of random methods for seeking maxima <https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244>`_. Operations Research, 6(2), pp.244-251. [ https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/rs/rs.py ]

Mathematical Optimization
-------------------------

* Cartis, C., Massart, E., & Otemissov, A. (2022). Bound-constrained global optimization of functions with low effective dimensionality using multiple random embeddings. Mathematical Programming, 1-62.
* Cartis, C., Massart, E., & Otemissov, A. (2022). Global optimization using random embeddings. Mathematical Programming, 1-49.
* Rios, L. M., & Sahinidis, N. V. (2013). Derivative-free optimization: A review of algorithms and comparison of software implementations. Journal of Global Optimization, 56(3), 1247-1293.
* Collette, Y., Hansen, N., Pujol, G., Salazar Aponte, D. and Le Riche, R., 2013. Object‐oriented programming of optimizers–Examples in Scilab. Multidisciplinary Design Optimization in Computational Mechanics, pp.499-538.
* Rinnooy Kan, A.H.G. and Timmer, G.T., 1987. Stochastic global optimization methods part I: Clustering methods. Mathematical Programming, 39, pp.27-56.

SPSA
----

* Spall, J. C., Hill, S. D., & Stark, D. R. (2006). Theoretical framework for comparing several stochastic optimization approaches. Probabilistic and Randomized Methods for Design under Uncertainty, 99-117.
* Spall, J. C. (1999, December). Stochastic optimization and the simultaneous perturbation method. In Proceedings of Conference on Winter Simulation (pp. 101-109).

ES
--

* Nomura, M., Akimoto, Y. and Ono, I., 2024. CMA-ES with learning rate adaptation. arXiv preprint arXiv:2401.15876.
* Spettel, P. and Beyer, H.G., 2022. On the design of a matrix adaptation evolution strategy for optimization on general quadratic manifolds. ACM Transactions on Evolutionary Learning and Optimization, 2(3), pp.1-32.
* Hellwig, M. and Beyer, H.G., 2020. On the steady state analysis of covariance matrix self-adaptation evolution strategies on the noisy ellipsoid model. Theoretical Computer Science, 832, pp.98-122.
* Maheswaranathan, N., Metz, L., Tucker, G., Choi, D., & Sohl-Dickstein, J. (2019, May). Guided evolutionary strategies: Augmenting random search with surrogate gradients. In International Conference on Machine Learning (pp. 4264-4273). PMLR.
* Choromanski, K., Rowland, M., Sindhwani, V., Turner, R., & Weller, A. (2018, July). Structured evolution with compact architectures for scalable policy optimization. In International Conference on Machine Learning (pp. 970-978). PMLR.
* Beyer, H.G. and Hellwig, M., 2016. The dynamics of cumulative step size adaptation on the ellipsoid model. Evolutionary Computation, 24(1), pp.25-57.
* Beyer, H.G., 2014. Convergence analysis of evolutionary algorithms that are based on the paradigm of information geometry. Evolutionary Computation, 22(4), pp.679-709.
* Pošík, P., Huyer, W. and Pál, L., 2012. A comparison of global search algorithms for continuous black box optimization. Evolutionary Computation, 20(4), pp.509-541.
* Arnold, D.V. and Salomon, R., 2007. Evolutionary gradient search revisited. IEEE Transactions on Evolutionary Computation, 11(4), pp.480-495.
* Ulmer, H., Streichert, F. and Zell, A., 2005. Model assisted evolution strategies. In Knowledge Incorporation in Evolutionary Computation (pp. 333-355). Springer Berlin Heidelberg.
* Arnold, D.V. and Beyer, H.G., 2004. Performance analysis of evolutionary optimization with cumulative step length adaptation. IEEE Transactions on Automatic Control, 49(4), pp.617-622.
* Beyer, H.G. and Arnold, D.V., 2003. Qualms regarding the optimality of cumulative path length control in CSA/CMA-evolution strategies. Evolutionary Computation, 11(1), pp.19-28.
* Schwefel, H.P., 1981. Numerical optimization of computer models. John Wiley & Sons, Inc.

EDA
---

* Lu, M., Ning, S., Liu, S., Sun, F., Zhang, B., Yang, B. and Wang, L., 2023, June. OPT-GAN: A broad-spectrum global optimizer for black-box problems by learning distribution. In Proceedings of AAAI Conference on Artificial Intelligence (pp. 12462-12472).
* Teytaud, F. and Teytaud, O., 2009, July. Why one must use reweighting in estimation of distribution algorithms. In Proceedings of ACM Annual Conference on Genetic and Evolutionary Computation (pp. 453-460).

PSO
---

* Camacho-Villalón, C. L., Dorigo, M., & Stützle, T. (2021). PSO-X: A component-based framework for the automatic design of particle swarm optimization algorithms. IEEE Transactions on Evolutionary Computation, 26(3), 402-416.
* Bonyadi, M. R., & Michalewicz, Z. (2015). Analysis of stability, local convergence, and transformation sensitivity of a variant of the particle swarm optimization algorithm. IEEE Transactions on Evolutionary Computation, 20(3), 370-385.
* Su, S., Zhang, Z., Liu, A. X., Cheng, X., Wang, Y., & Zhao, X. (2014). Energy-aware virtual network embedding. IEEE/ACM Transactions on Networking, 22(5), 1607-1620.
* De Oca, M. A. M., Stutzle, T., Birattari, M., & Dorigo, M. (2009). Frankenstein's PSO: A composite particle swarm optimization algorithm. IEEE Transactions on Evolutionary Computation, 13(5), 1120-1132.
* Mendes, R., Kennedy, J., & Neves, J. (2004). The fully informed particle swarm: simpler, maybe better. IEEE Transactions on Evolutionary Computation, 8(3), 204-210.
* Clerc, M., & Kennedy, J. (2002). The particle swarm-explosion, stability, and convergence in a multidimensional complex space. IEEE Transactions on Evolutionary Computation, 6(1), 58-73.

MA
--

* Lozano, M., Herrera, F., Krasnogor, N., & Molina, D. (2004). Real-coded memetic algorithms with crossover hill-climbing. Evolutionary Computation, 12(3), 273-302.
* Renders, J. M., & Flasse, S. P. (1996). Hybrid methods using genetic algorithms for global optimization. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 26(2), 243-258.

GA
--

* Kumar, A., Liu, B., Miikkulainen, R. and Stone, P., 2022, July. Effective mutation rate adaptation through group elite selection. In Proceedings of ACM Genetic and Evolutionary Computation Conference (pp. 721-729).
* Drugan, M. M., & Thierens, D. (2010). Geometrical recombination operators for real-coded evolutionary mcmcs. Evolutionary Computation, 18(2), 157-198.
* Clune, J., Misevic, D., Ofria, C., Lenski, R.E., Elena, S.F. and Sanjuán, R., 2008. Natural selection fails to optimize mutation rates for long-term adaptation on rugged fitness landscapes. PLoS Computational Biology, 4(9), p.e1000187.

NM
--
* Gao, F., & Han, L. (2012). Implementing the Nelder-Mead simplex algorithm with adaptive parameters. Computational Optimization and Applications, 51(1), 259-277.

BO
--
* Tan, J. and Nayman, N., 2023, July. Two-stage kernel Bayesian optimization in high dimensions. In Uncertainty in Artificial Intelligence (pp. 2099-2110). PMLR.
* Liu, S., Feng, Q., Eriksson, D., Letham, B. and Bakshy, E., 2023, April. Sparse Bayesian optimization. In International Conference on Artificial Intelligence and Statistics (pp. 3754-3774). PMLR.
* Kandasamy, K., Krishnamurthy, A., Schneider, J. and Póczos, B., 2018, March. Parallelised Bayesian optimisation via Thompson sampling. In International Conference on Artificial Intelligence and Statistics (pp. 133-142). PMLR.
* Hernández-Lobato, J.M., Requeima, J., Pyzer-Knapp, E.O. and Aspuru-Guzik, A., 2017, July. Parallel and distributed Thompson sampling for large-scale accelerated exploration of chemical space. In International Conference on Machine Learning (pp. 1470-1479). PMLR.
* Shah, A. and Ghahramani, Z., 2015. Parallel predictive entropy search for batch global optimization of expensive objective functions. Advances in Neural Information Processing Systems, 28.
* Snoek, J., Larochelle, H. and Adams, R.P., 2012. Practical Bayesian optimization of machine learning algorithms. Advances in Neural Information Processing Systems, 25.
* Ginsbourger, D., Le Riche, R. and Carraro, L., 2010. Kriging is well-suited to parallelize optimization. In Computational Intelligence in Expensive Optimization Problems (pp. 131-162). Berlin, Heidelberg: Springer Berlin Heidelberg.
* Jones, D.R., Schonlau, M. and Welch, W.J., 1998. Efficient global optimization of expensive black-box functions. Journal of Global Optimization, 13, pp.455-492.

SA
--

* Correia, A.H., Worrall, D.E. and Bondesan, R., 2023, April. Neural simulated annealing. In International Conference on Artificial Intelligence and Statistics (pp. 4946-4962). PMLR.

BBO/DFO/ZOO
-----------

* Antonakopoulos, K., Vu, D.Q., Cevher, V., Levy, K. and Mertikopoulos, P., 2022, June. UnderGrad: A universal black-box optimization method with almost dimension-free convergence rate guarantees. In International Conference on Machine Learning (pp. 772-795). PMLR.
* Arango, S.P., Jomaa, H.S., Wistuba, M. and Grabocka, J., 2021. Hpo-b: A large-scale reproducible benchmark for black-box hpo based on openml. arXiv preprint arXiv:2106.06257.
* Flaxman, A. D., Kalai, A. T., & McMahan, H. B. (2005, January). Online convex optimization in the bandit setting: gradient descent without a gradient. In Proceedings of Annual ACM-SIAM symposium on Discrete Algorithms (pp. 385-394).
