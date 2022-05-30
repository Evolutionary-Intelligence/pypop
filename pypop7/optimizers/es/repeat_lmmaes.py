import pickle
import numpy as np
import seaborn as sns

from pypop7.benchmarks.base_functions import sphere
from pypop7.optimizers.es.lmmaes import LMMAES as Solver

sns.set_theme(style='darkgrid')


def write_pickle(problem_dim, result):
    file_name = 'sphere_' + str(problem_dim) + '.pickle'
    file = open(file_name, 'wb')
    pickle.dump(result, file)
    file.close()


# read data from pickle file
def read_pickle(problem_dim):
    import sys
    from pypop7.optimizers.core import optimizer
    sys.modules['optimizer'] = optimizer
    file_name = "sphere_" + str(problem_dim) + ".pickle"
    with open(file_name, 'rb') as handle:
        result = pickle.load(handle)
        if problem_dim == 128:
            print(result['fitness'])
        return result


# run plot with all these data in pickle file
def run_plot():
    import matplotlib.pyplot as plt
    ndim_problems = [128, 256, 512, 1024, 2048, 4096, 8192]
    colors = ['r', 'orange', 'y', 'limegreen', 'cyan', 'b', 'purple']
    plt.figure()
    for k in range(len(ndim_problems)):
        problem_dim = ndim_problems[k]
        result = read_pickle(problem_dim)
        plt.plot(result['fitness'][:, 0], result['fitness'][:, 1], color=colors[k],
                 label="N = "+str(problem_dim), linestyle='dashed')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([1, 1e6])
    plt.ylim([1e-10, 3e5])
    plt.yticks(ticks=[1e-10, 1e-5, 1e0, 1e5])
    plt.xticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    plt.xlabel("number of function evaluations")
    plt.ylabel("objective")
    plt.title("Sphere")
    plt.show()


if __name__ == '__main__':
    ndim_problems = [128, 256, 512, 1024, 2048, 4096, 8192]
    for i in range(len(ndim_problems)):
        ndim_problem = ndim_problems[i]
        problem = {'fitness_function': sphere,
                   'ndim_problem': ndim_problem,
                   'lower_boundary': -5 * np.ones((ndim_problem,)),
                   'upper_boundary': 5 * np.ones((ndim_problem,))}
        options = {'max_function_evaluations': 2e6,
                   'fitness_threshold': 1e-10,
                   'max_runtime': 3600,  # 1 hours
                   'seed_rng': 0,
                   'x': 4 * np.ones((ndim_problem,)),  # mean
                   'sigma': 3,
                   'verbose_frequency': 200,
                   'record_fitness': True,
                   'record_fitness_frequency': 1}
        solver = Solver(problem, options)
        results = solver.optimize()
        print(results)
        # write results into pickle file
        write_pickle(ndim_problem, results)
    # plot the fitness graph for all these dimension's fitness
    run_plot()
