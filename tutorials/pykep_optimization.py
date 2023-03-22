"""This is a simple demo that optimize some function provided by pykep
    https://github.com/esa/pykep
    The document of pykep is located at:
    https://esa.github.io/pykep/
"""
import pygmo as pg  # It is recommend to use conda, and you'd better to use pygmo==2.18
import pykep as pk  # It is recommend to use conda
import matplotlib.pyplot as plp

from pypop7.optimizers.pso.spso import SPSO as Solver


fig, axes = plp.subplots(nrows=3, ncols=2, sharex='col', sharey='row', figsize=(15, 15))
problems = [pk.trajopt.gym.cassini2, pk.trajopt.gym.eve_mga1dsm, pk.trajopt.gym.messenger, pk.trajopt.gym.rosetta, pk.trajopt.gym.em5imp, pk.trajopt.gym.em7imp]

# We now loop through the 6 problems:
for prob_number in range(0, 6):

    udp = problems[prob_number]

    def udpfunc(x):
        return udp.fitness(x)[0]

    # I define the problems to be handled:
    prob = pg.problem(udp)

    pro = {'fitness_function': udpfunc,
            'ndim_problem': len(prob.get_lb()),
            'lower_boundary': prob.get_lb(),
            'upper_boundary': prob.get_ub()}
    opt = {'seed_rng': 0,
            'max_function_evaluations': 2e4,
            'fitness_threshold': 1e-10,
            'verbose': 1e2,
            'saving_fitness': 1}
    solver = Solver(pro, opt)
    res = solver.optimize()

    # We finally plot the results in a semilog plot, for each problem:
    if prob_number == 0:
        axes[0, 0].semilogy(res['fitness'][:, 0], res['fitness'][:, 1], 'k--', label='SPSO')
        axes[0, 0].set_title('Cassini 2: eval=20000, pop=20')
        axes[0, 0].set_xlim([0, 20000])

    elif prob_number == 1:
        axes[0, 1].semilogy(res['fitness'][:, 0], res['fitness'][:, 1], 'k--', label='SPSO')
        axes[0, 1].set_title('E-V-E MGA 1DSM: eval=20000, pop=20')
        axes[0, 1].set_xlim([0, 20000])

    elif prob_number == 2:
        axes[1, 0].semilogy(res['fitness'][:, 0], res['fitness'][:, 1], 'k--', label='SPSO')
        axes[1, 0].set_title('Messenger: eval=20000, pop=20')
        axes[1, 0].set_xlim([0, 20000])

    elif prob_number == 3:
        axes[1, 1].semilogy(res['fitness'][:, 0], res['fitness'][:, 1], 'k--', label='SPSO')
        axes[1, 1].set_title('Rosetta: eval=20000, pop=20')
        axes[1, 1].set_xlim([0, 20000])

    elif prob_number == 4:
        axes[2, 0].semilogy(res['fitness'][:, 0], res['fitness'][:, 1], 'k--', label='SPSO')
        axes[2, 0].set_title('E-M 5 imp: eval=20000, pop=20')
        axes[2, 0].set_xlim([0, 20000])

    elif prob_number == 5:
        axes[2, 1].semilogy(res['fitness'][:, 0], res['fitness'][:, 1], 'k--', label='SPSO')
        axes[2, 1].set_title('E-M 7 imp: eval=20000, pop=20')
        axes[2, 1].set_xlim([0, 20000])

for ax in axes.flat:
    ax.set(xlabel='Fevals [-]', ylabel='Best [m/s]')
    ax.grid()

# We save the results to a jpg file:
plp.savefig("pykep_optimization.jpg")
