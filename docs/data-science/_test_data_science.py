import pypop7.benchmarks.data_science as ds


if __name__ == '__main__':
    x, y = ds.read_cnae9()
    assert x.shape == (1080, 856)
    assert y.shape == (1080,)
    assert sum(y == 0) == 960
    assert sum(y == 1) == 120
    assert sum(y == 0) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    x, y = ds.read_madelon()
    assert x.shape == (2000, 500)
    assert y.shape == (2000,)
    assert sum(y == -1) == 1000
    assert sum(y == 1) == 1000
    assert sum(y == -1) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    x, y = ds.read_qsar_androgen_receptor()
    assert x.shape == (1687, 1024)
    assert y.shape == (1687,)
    assert sum(y == -1) == 1488
    assert sum(y == 1) == 199
    assert sum(y == -1) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)
