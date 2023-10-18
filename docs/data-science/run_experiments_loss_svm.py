import os
import time
import pickle  # for data storage
import argparse

import numpy as np  # for numerical computing
from sklearn.preprocessing import Normalizer

import pypop7.benchmarks.data_science as ds


class LossSVMQAR(object):
    def __init__(self):
        self.x, self.y = ds.read_qsar_androgen_receptor(is_10=False)
        transformer = Normalizer().fit(self.x)
        self.x = transformer.transform(self.x)

    def __format__(self, format_spec):
        return 'loss_svm_qar'

    def __call__(self, w):
        return ds.loss_svm(w, self.x, self.y)


class LossSVMM(object):
    def __init__(self):
        self.x, self.y = ds.read_madelon(is_10=False)
        transformer = Normalizer().fit(self.x)
        self.x = transformer.transform(self.x)

    def __format__(self, format_spec):
        return 'loss_svm_m'

    def __call__(self, w):
        return ds.loss_svm(w, self.x, self.y)


class LossSVMC(object):
    def __init__(self):
        self.x, self.y = ds.read_cnae9(is_10=False)
        transformer = Normalizer().fit(self.x)
        self.x = transformer.transform(self.x)

    def __format__(self, format_spec):
        return 'loss_svm_c'

    def __call__(self, w):
        return ds.loss_svm(w, self.x, self.y)


class LossSVMSHD(object):
    def __init__(self):
        self.x, self.y = ds.read_semeion_handwritten_digit(is_10=False)
        transformer = Normalizer().fit(self.x)
        self.x = transformer.transform(self.x)

    def __format__(self, format_spec):
        return 'loss_svm_shd'

    def __call__(self, w):
        return ds.loss_svm(w, self.x, self.y)


class LossSVMPDC(object):
    def __init__(self):
        self.x, self.y = ds.read_parkinson_disease_classification(is_10=False)
        transformer = Normalizer().fit(self.x)
        self.x = transformer.transform(self.x)

    def __format__(self, format_spec):
        return 'loss_svm_pdc'

    def __call__(self, w):
        return ds.loss_svm(w, self.x, self.y)


class Experiment(object):
    """Each experiment consists of four settings:
        `index`       : experiment (trial) index (`int`, >= 1),
        `function`    : test function (`func`),
        `seed`        : seed for random number generation (`int`, >= 0),
        `ndim_problem`: function dimensionality (`int`, > 0).
    """
    def __init__(self, index, function, seed, ndim_problem):
        self.index = index  # index of each experiment (trial)
        assert self.index > 0
        self.function = function  # function of each experiment to be *minimized*
        self.seed = seed  # random seed of each experiment
        assert self.seed >= 0
        self.ndim_problem = ndim_problem  # function dimensionality
        assert self.ndim_problem > 0
        self._folder = 'pypop7_benchmarks_lso'  # folder to save all data
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        # to set file name for each experiment
        self._file = os.path.join(self._folder, 'Algo-{}_Func-{}_Dim-{}_Exp-{}.pickle')

    def run(self, optimizer):
        # to define all the necessary properties of the objective/cost function to be minimized
        problem = {'fitness_function': self.function,  # cost function
                   'ndim_problem': self.ndim_problem,  # dimension
                   'upper_boundary': 10.0*np.ones((self.ndim_problem,)),  # search boundary
                   'lower_boundary': -10.0*np.ones((self.ndim_problem,))}
        # to define all the necessary properties of the black-box optimizer considered
        options = {'max_function_evaluations': np.Inf,  # here we focus on the *wall-clock* time
                   # 'max_function_evaluations': np.Inf,
                   'max_runtime': 60*60*3,  # maximal runtime to be allowed (seconds)
                   'fitness_threshold': 1e-10,  # fitness threshold to stop the optimization process
                   'seed_rng': self.seed,  # seed for random number generation (RNG)
                   'saving_fitness': 2000,  # to compress the convergence data (for saving storage space)
                   'verbose': 0,  # to not print verbose information (for simplicity)
                   'sigma': 20.0/3.0,  # note that not all optimizers will use this setting (for e.g., ESs)
                   'temperature': 100}  # note that not all optimizers will use this setting (for e.g., SAs)
        solver = optimizer(problem, options)  # to initialize the optimizer object
        results = solver.optimize()  # to run the optimization/search/evolution process
        file = self._file.format(solver.__class__.__name__,
                                 solver.fitness_function,
                                 solver.ndim_problem,
                                 self.index)
        with open(file, 'wb') as handle:  # to save all data in .pickle format
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Experiments(object):
    """A set of *independent* experiments starting and ending in the given index range."""
    def __init__(self, start, end):
        self.start = start  # starting index of independent experiments
        assert self.start > 0
        self.end = end  # ending index of independent experiments
        assert self.end > 0 and self.end >= self.start
        self.indices = range(self.start, self.end + 1)  # index range (1-based rather 0-based)
        loss_svm_pdc = LossSVMPDC()
        loss_svm_shd = LossSVMSHD()
        loss_svm_c = LossSVMC()
        loss_svm_m = LossSVMM()
        loss_svm_qar = LossSVMQAR()
        self.functions = [loss_svm_pdc,
                          loss_svm_shd,
                          loss_svm_c,
                          loss_svm_m,
                          loss_svm_qar]
        self.ndim_problem = [loss_svm_pdc.x.shape[1] + 1,
                             loss_svm_shd.x.shape[1] + 1,
                             loss_svm_c.x.shape[1] + 1,
                             loss_svm_m.x.shape[1] + 1,
                             loss_svm_qar.x.shape[1] + 1]
        self.seeds = np.random.default_rng(2023).integers(  # to generate all random seeds *in advances*
            np.iinfo(np.int64).max, size=(len(self.functions), 20))

    def run(self, optimizer):
        for index in self.indices:  # independent experiments
            print('* experiment: {:d} ***:'.format(index))
            for d, f in enumerate(self.functions):  # for each function
                start_time = time.time()
                print('  * function: {:s}:'.format(f))
                experiment = Experiment(index, f, self.seeds[d, index], self.ndim_problem[d])
                experiment.run(optimizer)
                print('    runtime: {:7.5e}.'.format(time.time() - start_time))


if __name__ == '__main__':
    start_runtime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)  # starting index
    parser.add_argument('--end', '-e', type=int)  # ending index
    parser.add_argument('--optimizer', '-o', type=str)  # optimizer
    args = parser.parse_args()
    params = vars(args)
    if params['optimizer'] == 'LAMCTS':  # 2020
        from pypop7.optimizers.bo.lamcts import LAMCTS as Optimizer
    elif params['optimizer'] == 'SRS':  # 2001
        from pypop7.optimizers.rs.srs import SRS as Optimizer
    elif params['optimizer'] == 'BES':  # 2022
        from pypop7.optimizers.rs.bes import BES as Optimizer
    elif params['optimizer'] == 'FEP':  # 1999
        from pypop7.optimizers.ep.fep import FEP as Optimizer
    elif params['optimizer'] == 'GENITOR':
        from pypop7.optimizers.ga.genitor import GENITOR as Optimizer
    elif params['optimizer'] == 'G3PCX':
        from pypop7.optimizers.ga.g3pcx import G3PCX as Optimizer
    elif params['optimizer'] == 'GL25':
        from pypop7.optimizers.ga.gl25 import GL25 as Optimizer
    elif params['optimizer'] == 'CSA':
        from pypop7.optimizers.sa.csa import CSA as Optimizer
    elif params['optimizer'] == 'ESA':
        from pypop7.optimizers.sa.esa import ESA as Optimizer
    elif params['optimizer'] == 'NSA':
        from pypop7.optimizers.sa.nsa import NSA as Optimizer
    elif params['optimizer'] == 'COEA':
        from pypop7.optimizers.cc.coea import COEA as Optimizer
    elif params['optimizer'] == 'COSYNE':
        from pypop7.optimizers.cc.cosyne import COSYNE as Optimizer
    elif params['optimizer'] == 'COCMA':
        from pypop7.optimizers.cc.cocma import COCMA as Optimizer
    elif params['optimizer'] == 'HCC':
        from pypop7.optimizers.cc.hcc import HCC as Optimizer
    elif params['optimizer'] == 'SPSOL':
        from pypop7.optimizers.pso.spsol import SPSOL as Optimizer
    elif params['optimizer'] == 'CPSO':
        from pypop7.optimizers.pso.cpso import CPSO as Optimizer
    elif params['optimizer'] == 'CLPSO':
        from pypop7.optimizers.pso.clpso import CLPSO as Optimizer
    elif params['optimizer'] == 'CCPSO2':
        from pypop7.optimizers.pso.ccpso2 import CCPSO2 as Optimizer
    elif params['optimizer'] == 'TDE':
        from pypop7.optimizers.de.tde import TDE as Optimizer
    elif params['optimizer'] == 'SCEM':
        from pypop7.optimizers.cem.scem import SCEM as Optimizer
    elif params['optimizer'] == 'DSCEM':
        from pypop7.optimizers.cem.dscem import DSCEM as Optimizer
    elif params['optimizer'] == 'MRAS':
        from pypop7.optimizers.cem.mras import MRAS as Optimizer
    elif params['optimizer'] == 'DCEM':
        from pypop7.optimizers.cem.dcem import DCEM as Optimizer
    elif params['optimizer'] == 'UMDA':
        from pypop7.optimizers.eda.umda import UMDA as Optimizer
    elif params['optimizer'] == 'EMNA':
        from pypop7.optimizers.eda.emna import EMNA as Optimizer
    elif params['optimizer'] == 'AEMNA':
        from pypop7.optimizers.eda.aemna import AEMNA as Optimizer
    elif params['optimizer'] == 'RPEDA':
        from pypop7.optimizers.eda.rpeda import RPEDA as Optimizer
    elif params['optimizer'] == 'SGES':
        from pypop7.optimizers.nes.sges import SGES as Optimizer
    elif params['optimizer'] == 'XNES':
        from pypop7.optimizers.nes.xnes import XNES as Optimizer
    elif params['optimizer'] == 'SNES':
        from pypop7.optimizers.nes.snes import SNES as Optimizer
    elif params['optimizer'] == 'R1NES':
        from pypop7.optimizers.nes.r1nes import R1NES as Optimizer
    elif params['optimizer'] == 'ASGA':  # 2021
        from pypop7.optimizers.ga.asga import ASGA as Optimizer
    elif params['optimizer'] == 'MMES':  # 2021
        from pypop7.optimizers.es.mmes import MMES as Optimizer
    elif params['optimizer'] == 'SAMAES':  # 2020
        from pypop7.optimizers.es.samaes import SAMAES as Optimizer
    elif params['optimizer'] == 'SAES':  # 2020
        from pypop7.optimizers.es.saes import SAES as Optimizer
    elif params['optimizer'] == 'DDCMA':  # 2020
        from pypop7.optimizers.es.ddcma import DDCMA as Optimizer
    elif params['optimizer'] == 'FCMAES':  # 2020
        from pypop7.optimizers.es.fcmaes import FCMAES as Optimizer
    elif params['optimizer'] == 'LMMAES':  # 2019
        from pypop7.optimizers.es.lmmaes import LMMAES as Optimizer
    elif params['optimizer'] == 'RMES':  # 2018
        from pypop7.optimizers.es.rmes import RMES as Optimizer
    elif params['optimizer'] == 'R1ES':  # 2018
        from pypop7.optimizers.es.r1es import R1ES as Optimizer
    elif params['optimizer'] == 'FMAES':  # 2017
        from pypop7.optimizers.es.fmaes import FMAES as Optimizer
    elif params['optimizer'] == 'MAES':  # 2017
        from pypop7.optimizers.es.maes import MAES as Optimizer
    elif params['optimizer'] == 'LMCMA':  # 2017
        from pypop7.optimizers.es.lmcma import LMCMA as Optimizer
    elif params['optimizer'] == 'GS':  # 2017
        from pypop7.optimizers.rs.gs import GS as Optimizer
    elif params['optimizer'] == 'CCMAES2016':  # 2016
        from pypop7.optimizers.es.ccmaes2016 import CCMAES2016 as Optimizer
    elif params['optimizer'] == 'VKDCMA':  # 2016
        from pypop7.optimizers.es.vkdcma import VKDCMA as Optimizer
    elif params['optimizer'] == 'CMAES':  # 2016
        from pypop7.optimizers.es.cmaes import CMAES as Optimizer
    elif params['optimizer'] == 'OPOA2015':  # 2015
        from pypop7.optimizers.es.opoa2015 import OPOA2015 as Optimizer
    elif params['optimizer'] == 'LMCMAES':  # 2014
        from pypop7.optimizers.es.lmcmaes import LMCMAES as Optimizer
    elif params['optimizer'] == 'VDCMA':  # 2014
        from pypop7.optimizers.es.vdcma import VDCMA as Optimizer
    elif params['optimizer'] == 'SHADE':  # 2013
        from pypop7.optimizers.de.shade import SHADE as Optimizer
    elif params['optimizer'] == 'IPSO':  # 2011
        from pypop7.optimizers.pso.ipso import IPSO as Optimizer
    elif params['optimizer'] == 'CODE':  # 2011
        from pypop7.optimizers.de.code import CODE as Optimizer
    elif params['optimizer'] == 'OPOA2010':  # 2010
        from pypop7.optimizers.es.opoa2010 import OPOA2010 as Optimizer
    elif params['optimizer'] == 'ENES':  # 2009
        from pypop7.optimizers.nes.enes import ENES as Optimizer
    elif params['optimizer'] == 'CCMAES2009':  # 2009
        from pypop7.optimizers.es.ccmaes2009 import CCMAES2009 as Optimizer
    elif params['optimizer'] == 'OPOC2009':  # 2009
        from pypop7.optimizers.es.opoc2009 import OPOC2009 as Optimizer
    elif params['optimizer'] == 'JADE':  # 2009
        from pypop7.optimizers.de.jade import JADE as Optimizer
    elif params['optimizer'] == 'SEPCMAES':  # 2008
        from pypop7.optimizers.es.sepcmaes import SEPCMAES as Optimizer
    elif params['optimizer'] == 'ONES':  # 2008
        from pypop7.optimizers.nes.ones import ONES as Optimizer
    elif params['optimizer'] == 'OPOC2006':  # 2006
        from pypop7.optimizers.es.opoc2006 import OPOC2006 as Optimizer
    elif params['optimizer'] == 'LEP':  # 2004
        from pypop7.optimizers.ep.lep import LEP as Optimizer
    elif params['optimizer'] == 'RHC':  # 2004
        from pypop7.optimizers.rs.rhc import RHC as Optimizer
    elif params['optimizer'] == 'ARHC':  # 2004
        from pypop7.optimizers.rs.arhc import ARHC as Optimizer
    elif params['optimizer'] == 'GPS':  # 1997
        from pypop7.optimizers.ds.gps import GPS as Optimizer
    elif params['optimizer'] == 'CDE':  # 1997
        from pypop7.optimizers.de.cde import CDE as Optimizer
    elif params['optimizer'] == 'SPSO':  # 1995
        from pypop7.optimizers.pso.spso import SPSO as Optimizer
    elif params['optimizer'] == 'CSAES':  # [Ostermeier et al., 1994]
        from pypop7.optimizers.es.csaes import CSAES as Optimizer
    elif params['optimizer'] == 'DSAES':  # [Ostermeier et al., 1994]
        from pypop7.optimizers.es.dsaes import DSAES as Optimizer
    elif params['optimizer'] == 'CEP':  # [Bäck&Schwefel, 1993]
        from pypop7.optimizers.ep.cep import CEP as Optimizer
    elif params['optimizer'] == 'SSAES':  # [Schwefel, 1984]
        from pypop7.optimizers.es.ssaes import SSAES as Optimizer
    elif params['optimizer'] == 'RES':  # [Rechenberg, 1973]
        from pypop7.optimizers.es.res import RES as Optimizer
    elif params['optimizer'] == 'NM':  # [Nelder&Mead, 1965]
        from pypop7.optimizers.ds.nm import NM as Optimizer
    elif params['optimizer'] == 'POWELL':  # [Powell, 1964]
        from pypop7.optimizers.ds.powell import POWELL as Optimizer
    elif params['optimizer'] == 'HJ':  # [Hooke&Jeeves, 1961]
        from pypop7.optimizers.ds.hj import HJ as Optimizer
    elif params['optimizer'] == 'PRS':  # [Brooks, 1958]
        from pypop7.optimizers.rs.prs import PRS as Optimizer
    elif params['optimizer'] == 'CS':  # [Fermi&Metropolis, 1952]
        from pypop7.optimizers.ds.cs import CS as Optimizer
    experiments = Experiments(params['start'], params['end'])
    experiments.run(Optimizer)
    print('*** Total runtime: {:7.5e} ***.'.format(time.time() - start_runtime))
