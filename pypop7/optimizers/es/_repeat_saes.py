"""Repeat the following paper for `SAES`:
    Beyer, H.G., 2020, July.
    Design principles for matrix adaptation evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion (pp. 682-700). ACM.
    https://dl.acm.org/doi/abs/10.1145/3377929.3389870

    Luckily our Python code could repeat the data reported in the original Matlab code (as shown below) *well*.
    Therefore, we argue that its repeatability could be **well-documented**.

    The following Matlab code comes from https://homepages.fhv.at/hgb/downloads/mu_mu_I_lambda-ES.oct
      with only slight modifications:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This is a simple Octave implementation of the (mu/mu_I, lambda)-sigmaSA-ES
    % as discussed in
    % http://www.scholarpedia.org/article/Evolution_Strategies
    % Note, if you want to use this in Matlab, you have to copy each function
    % definition in a separate m-file.
    % The code presented below should be regarded as a skeleton only
    % Note, the code presented is to be used under GNU General Public License
    % Author: Hans-Georg Beyer
    % Email: Hans-Georg.Beyer_AT_fhv.at
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % here comes the ES example:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mu = 3;                 % number of parents
    lambda = 12;            % number of offspring
    yInit = ones(30,1);     % initial parent vector
    sigmaInit = 1;          % initial global mutation strength sigma
    sigmaMin = 1e-10;       % ES stops when sigma is smaller than sigmaMin

    % initialization:
    n = length(yInit);      % determine search space dimensionality n
    tau = 1/sqrt(2*n);      % self-adaptation learning rate
    % initializing individual population:
    Individual.y = yInit;
    Individual.sigma = sigmaInit;
    Individual.F = fitness(Individual.y);
    for i=1:mu; ParentPop{i} = Individual; end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % evolution loop of the (mu/mu_I, lambda)-sigma-SA-ES
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n_generations = 0; %+
    while(1)
     n_generations = n_generations + 1; %+
     Recombinant = recombine(ParentPop);              % recombine parents
     for l = 1:lambda;                                % generate lambda offspring
      OffspringIndividual.sigma = Recombinant.sigma * exp(tau*randn); % mutate sigma
      OffspringIndividual.y = Recombinant.y + OffspringIndividual.sigma * randn(n, 1); % mutate object parameter
      OffspringIndividual.F = fitness(OffspringIndividual.y); % determine fitness
      OffspringPop{l} = OffspringIndividual;                  % offspring complete
     end;
     ParentPop = SortPop(OffspringPop, mu);   % sort population
     disp(ParentPop{1}.F);                    % display best fitness in population
     if ( ParentPop{1}.sigma < sigmaMin ) break; end; % termination criterion
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Remark: Final approximation of the optimizer is in "ParentPop{1}.y"
    %         corresponding fitness is in "ParentPop{1}.F" and the final
    %         mutation strength is in "ParentPop{1}.sigma"
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp(ParentPop{1}.F); %+ 1.3899e-18
    disp(n_generations); %+ 481
"""
import numpy as np

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.es.saes import SAES


if __name__ == '__main__':
    problem = {'fitness_function': sphere,
               'ndim_problem': 30}
    options = {'seed_rng': 1,  # undefined in the original paper
               'fitness_threshold': 1.3899e-18,
               'n_individuals': 12,
               'n_parents': 3,
               'saving_fitness': 1,
               'x': np.ones((30,)),
               'sigma': 1,
               'is_restart': False}
    saes = SAES(problem, options)
    results = saes.optimize()
    print(results)
    print(results['best_so_far_y'])  # 1.3530314470392882e-18
    print(results['sigma'])  # 1.7694109040812627e-10
    print(results['_n_generations'])  # 483
