import numpy as np  # engine for numerical computing
from sklearn.preprocessing import MinMaxScaler, Normalizer

import pypop7.benchmarks.data_science as ds


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
