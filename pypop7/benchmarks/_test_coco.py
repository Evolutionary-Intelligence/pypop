from pypop7.optimizers.es.maes import MAES
from pypop7.benchmarks.coco import coco_bbob


if __name__ == '__main__':
    coco_bbob(MAES)
