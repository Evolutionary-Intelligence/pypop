from pypop7.optimizers.de.de import DE  # abstract class for all DE classes
from pypop7.optimizers.de.cde import CDE
from pypop7.optimizers.de.tde import TDE
from pypop7.optimizers.de.jade import JADE
from pypop7.optimizers.de.code import CODE


__all__ = [DE,  # Differential Evolution (DE)
           CDE,  # Classic Differential Evolution (CDE)
           TDE,  # Trigonometric-mutation Differential Evolution (TDE)
           JADE,  # Adaptive Differential Evolution (JADE)
           CODE]  # COmposite Differential Evolution (CODE)
