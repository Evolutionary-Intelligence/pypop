from pypop7.optimizers.ep.ep import EP  # abstract class for all evolutionary programming (EP) classes [1962-1966]
from pypop7.optimizers.ep.cep import CEP  # Classical Evolutionary Programming with self-adaptive mutation [1993]
from pypop7.optimizers.ep.fep import FEP
from pypop7.optimizers.ep.lep import LEP


__all__ = [EP,  # Evolutionary Programming [1962-1966]
           CEP,  # Classical Evolutionary Programming with self-adaptive mutation [1993]
           FEP,  # Fast Evolutionary Programming with self-adaptive mutation (FEP)
           LEP]  # Lévy-distribution based Evolutionary Programming (LEP)
