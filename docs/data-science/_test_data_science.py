import numpy as np
from sklearn.preprocessing import Normalizer

import pypop7.benchmarks.data_science as ds


if __name__ == '__main__':
    x, y = ds.read_parkinson_disease_classification()
    assert x.shape == (756, 753)
    assert y.shape == (756,)
    assert sum(y == 0) == 192
    assert sum(y == 1) == 564
    assert sum(y == 0) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    w = np.zeros((x.shape[1] + 1,))
    loss = ds.cross_entropy_loss_lr(w, x, y)
    print(loss)  # 0.6931471805599454
    loss = ds.square_loss_lr(w, x, y)
    print(loss)  # 0.25

    x, y = ds.read_semeion_handwritten_digit()
    assert x.shape == (1593, 256)
    assert y.shape == (1593,)
    assert sum(y == 0) == 1435
    assert sum(y == 1) == 158
    assert sum(y == 0) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    w = np.zeros((x.shape[1] + 1,))
    loss = ds.cross_entropy_loss_lr(w, x, y)
    print(loss)  # 0.6931471805599452
    loss = ds.square_loss_lr(w, x, y)
    print(loss)  # 0.25

    x, y = ds.read_cnae9()
    assert x.shape == (1080, 856)
    assert y.shape == (1080,)
    assert sum(y == 0) == 960
    assert sum(y == 1) == 120
    assert sum(y == 0) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    w = np.zeros((x.shape[1] + 1,))
    loss = ds.cross_entropy_loss_lr(w, x, y)
    print(loss)  # 0.6931471805599453
    loss = ds.square_loss_lr(w, x, y)
    print(loss)  # 0.25

    x, y = ds.read_madelon()
    assert x.shape == (2000, 500)
    assert y.shape == (2000,)
    assert sum(y == -1) == 1000
    assert sum(y == 1) == 1000
    assert sum(y == -1) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    transformer = Normalizer().fit(x)
    print(x)
    x = transformer.transform(x)
    print(x)
    w = 3.0*np.ones((x.shape[1] + 1,))
    loss = ds.logistic_loss_lr(w, x, y)
    print(loss)  # 34.97093739858563
    loss = ds.logistic_loss_l2(w, x, y)
    print(loss)  # 36.09818739858564

    x, y = ds.read_qsar_androgen_receptor()
    assert x.shape == (1687, 1024)
    assert y.shape == (1687,)
    assert sum(y == -1) == 1488
    assert sum(y == 1) == 199
    assert sum(y == -1) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    transformer = Normalizer().fit(x)
    print(x)
    x = transformer.transform(x)
    print(x)
    w = 3.0*np.ones((x.shape[1] + 1,))
    loss = ds.logistic_loss_lr(w, x, y)
    print(loss)  # 26.06566955795496
    loss = ds.logistic_loss_l2(w, x, y)
    print(loss)  # 28.799813007866042
