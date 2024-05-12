from pypop7.optimizers.cc.cc import CC  # abstract class for all cooperative coevolution (CC) classes [1994]
from pypop7.optimizers.cc.coea import COEA
from pypop7.optimizers.cc.hcc import HCC
from pypop7.optimizers.cc.cosyne import COSYNE
from pypop7.optimizers.cc.cocma import COCMA


__all__ = [CC,  # Cooperative Coevolution [1994]
           COEA,  # CoOperative co-Evolutionary Algorithm (COEA)
           HCC,  # Hierarchical Cooperative Co-evolution (HCC)
           COSYNE,  # CoOperative SYnapse NEuroevolution (COSYNE)
           COCMA]  # CoOperative CO-evolutionary Covariance Matrix Adaptation (COCMA)
