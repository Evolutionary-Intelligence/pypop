from pypop7.benchmarks.data_science import read_qsar_androgen_receptor


def test_read_qsar_androgen_receptor():
    x, y = read_qsar_androgen_receptor()
    assert x.shape == (1687, 1024)
    assert y.shape == (1687,)
