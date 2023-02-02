from pypop7.optimizers.es.es import ES  # Evolution Strategies
from pypop7.optimizers.es.res import RES  # Rechenberg’s (1+1)-Evolution Strategy with 1/5th success rule
from pypop7.optimizers.es.ssaes import SSAES  # Schwefel's Self-Adaptation Evolution Strategy
from pypop7.optimizers.es.dsaes import DSAES  # Derandomized Self-Adaptation Evolution Strategy
from pypop7.optimizers.es.csaes import CSAES  # Cumulative Step-size self-Adaptation Evolution Strategy
from pypop7.optimizers.es.saes import SAES  # Self-Adaptation Evolution Strategy
from pypop7.optimizers.es.samaes import SAMAES  # Self-Adaptation Matrix Adaptation Evolution Strategy
from pypop7.optimizers.es.cmaes import CMAES  # Covariance Matrix Adaptation Evolution Strategy
from pypop7.optimizers.es.opoc2006 import OPOC2006  # (1+1)-Cholesky-CMA-ES (2006)
from pypop7.optimizers.es.sepcmaes import SEPCMAES  # Separable Covariance Matrix Adaptation Evolution Strategy
from pypop7.optimizers.es.opoc2009 import OPOC2009  # (1+1)-Cholesky-CMA-ES (2009)
from pypop7.optimizers.es.ccmaes2009 import CCMAES2009  # Cholesky-CMA-ES (2009)
from pypop7.optimizers.es.opoa2010 import OPOA2010  # (1+1)-Active-CMA-ES (2010)
from pypop7.optimizers.es.lmcmaes import LMCMAES  # Limited-Memory CMA-ES (2014)
from pypop7.optimizers.es.opoa2015 import OPOA2015  # (1+1)-Active-CMA-ES (2015)
from pypop7.optimizers.es.ccmaes2016 import CCMAES2016
from pypop7.optimizers.es.lmcma import LMCMA  # Limited-Memory Covariance Matrix Adaptation (2017)
from pypop7.optimizers.es.maes import MAES  # Matrix Adaptation Evolution Strategy (2017)
from pypop7.optimizers.es.r1es import R1ES  # Rank-One Evolution Strategy (2018)
from pypop7.optimizers.es.rmes import RMES  # Rank-M Evolution Strategy (2018)
from pypop7.optimizers.es.lmmaes import LMMAES  # Limited-Memory Matrix Adaptation Evolution Strategy (2019)
from pypop7.optimizers.es.fmaes import FMAES  # Fast Matrix Adaptation Evolution Strategy (2020)
from pypop7.optimizers.es.fcmaes import FCMAES  # Fast Covariance Matrix Adaptation Evolution Strategy (2020)
from pypop7.optimizers.es.ddcma import DDCMA  # Diagonal Decoding Covariance Matrix Adaptation (2020)
from pypop7.optimizers.es.mmes import MMES  # Mixture Model-based Evolution Strategy (2021)


__all__ = [ES,  # base (abstract) class
           RES, SSAES, DSAES, CSAES, SAES, SAMAES,  # representative ES versions during early development
           CMAES,  # state-of-the-art ES version
           OPOC2006, OPOC2009, CCMAES2009, OPOA2010, OPOA2015, CCMAES2016, MAES, FMAES, DDCMA,  # modern ES versions
           SEPCMAES, R1ES, RMES, LMCMAES, LMCMA, LMMAES, FCMAES, MMES]  # especially for large-scale BBO
