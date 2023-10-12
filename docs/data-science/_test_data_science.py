import pypop7.benchmarks.data_science as ds


if __name__ == '__main__':
    x, y = ds.read_qsar_androgen_receptor()
    assert x.shape == (1687, 1024)
    assert y.shape == (1687,)
    assert sum(y == -1) == 1488
    assert sum(y == 1) == 199
    print(x.dtype, y.dtype)
