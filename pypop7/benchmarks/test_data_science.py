import numpy as np  # engine for numerical computing
from sklearn.preprocessing import MinMaxScaler, Normalizer

import pypop7.benchmarks.data_science as ds


def test_read_parkinson_disease_classification():
    x, y = ds.read_parkinson_disease_classification()
    assert x.shape == (756, 753)
    assert y.shape == (756,)
    assert sum(y == 0) == 192
    assert sum(y == 1) == 564
    assert sum(y == 0) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    x, y = ds.read_parkinson_disease_classification(is_10=False)
    assert x.shape == (756, 753)
    assert y.shape == (756,)
    assert sum(y == -1) == 192
    assert sum(y == 1) == 564
    assert sum(y == -1) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)


def test_read_semeion_handwritten_digit():
    x, y = ds.read_semeion_handwritten_digit()
    assert x.shape == (1593, 256)
    assert y.shape == (1593,)
    assert sum(y == 0) == 1435
    assert sum(y == 1) == 158
    assert sum(y == 0) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    x, y = ds.read_semeion_handwritten_digit(is_10=False)
    assert x.shape == (1593, 256)
    assert y.shape == (1593,)
    assert sum(y == -1) == 1435
    assert sum(y == 1) == 158
    assert sum(y == -1) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)


def test_read_cnae9():
    x, y = ds.read_cnae9()
    assert x.shape == (1080, 856)
    assert y.shape == (1080,)
    assert sum(y == 0) == 960
    assert sum(y == 1) == 120
    assert sum(y == 0) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    x, y = ds.read_cnae9(is_10=False)
    assert x.shape == (1080, 856)
    assert y.shape == (1080,)
    assert sum(y == -1) == 960
    assert sum(y == 1) == 120
    assert sum(y == -1) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)


def test_read_madelon():
    x, y = ds.read_madelon()
    assert x.shape == (2000, 500)
    assert y.shape == (2000,)
    assert sum(y == -1) == 1000
    assert sum(y == 1) == 1000
    assert sum(y == -1) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    x, y = ds.read_madelon(is_10=True)
    assert x.shape == (2000, 500)
    assert y.shape == (2000,)
    assert sum(y == 0) == 1000
    assert sum(y == 1) == 1000
    assert sum(y == 0) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)


def test_read_qsar_androgen_receptor():
    x, y = ds.read_qsar_androgen_receptor()
    assert x.shape == (1687, 1024)
    assert y.shape == (1687,)
    assert sum(y == -1) == 1488
    assert sum(y == 1) == 199
    assert sum(y == -1) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    x, y = ds.read_qsar_androgen_receptor(is_10=True)
    assert x.shape == (1687, 1024)
    assert y.shape == (1687,)
    assert sum(y == 0) == 1488
    assert sum(y == 1) == 199
    assert sum(y == 0) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)
    scalar = MinMaxScaler()
    scalar.fit(x)
    x = scalar.transform(x)
    print(np.min(x), np.max(x))  # 0.0 1.0
