import numpy as np
from sklearn.preprocessing import Normalizer

import pypop7.benchmarks.data_science as ds


def test_cross_entropy_loss_lr():
    x, y = ds.read_parkinson_disease_classification()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1] + 1,))
    print(ds.cross_entropy_loss_lr(w, x, y))  # 1.7379784430717373
    cross_entropy_loss_lr = ds.CrossEntropyLossLR()
    print(cross_entropy_loss_lr(w, x, y))  # 1.7379784430717373
    assert ds.cross_entropy_loss_lr(w, x, y) == cross_entropy_loss_lr(w, x, y)


def test_cross_entropy_loss_l2():
    x, y = ds.read_parkinson_disease_classification()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1] + 1,))
    print(ds.cross_entropy_loss_l2(w, x, y))  # 6.226073681166976
    cross_entropy_loss_l2 = ds.CrossEntropyLossL2()
    print(cross_entropy_loss_l2(w, x, y))  # 6.226073681166976
    assert ds.cross_entropy_loss_l2(w, x, y) == cross_entropy_loss_l2(w, x, y)


def test_square_loss_lr():
    x, y = ds.read_parkinson_disease_classification()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1] + 1,))
    print(ds.square_loss_lr(w, x, y))  # 0.6055914432091616
    square_loss_lr = ds.SquareLossLR()
    print(square_loss_lr(w, x, y))  # 0.6055914432091616
    assert ds.square_loss_lr(w, x, y) == square_loss_lr(w, x, y)


def test_logistic_loss_lr():
    x, y = ds.read_qsar_androgen_receptor()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1] + 1,))
    print(ds.logistic_loss_lr(w, x, y))  # 26.06566955795496
    logistic_loss_lr = ds.LogisticLossLR()
    print(logistic_loss_lr(w, x, y))  # 26.06566955795496
    assert ds.logistic_loss_lr(w, x, y) == logistic_loss_lr(w, x, y)


def test_logistic_loss_l2():
    x, y = ds.read_qsar_androgen_receptor()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1] + 1,))
    print(ds.logistic_loss_l2(w, x, y))  # 28.799813007866042
    logistic_loss_l2 = ds.LogisticLossL2()
    print(logistic_loss_l2(w, x, y))  # 28.799813007866042
    assert ds.logistic_loss_l2(w, x, y) == logistic_loss_l2(w, x, y)


def test_tanh_loss_lr():
    x, y = ds.read_qsar_androgen_receptor()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1] + 1,))
    print(ds.tanh_loss_lr(w, x, y))  # 3.52815643380549
    tanh_loss_lr = ds.TanhLossLR()
    print(tanh_loss_lr(w, x, y))  # 3.52815643380549
    assert ds.tanh_loss_lr(w, x, y) == tanh_loss_lr(w, x, y)


def test_hinge_loss_perceptron():
    x, y = ds.read_qsar_androgen_receptor()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1] + 1,))
    print(ds.hinge_loss_perceptron(w, x, y))  # 26.065669550828645
    hinge_loss_perceptron = ds.HingeLossPerceptron()
    print(hinge_loss_perceptron(w, x, y))  # 26.065669550828645
    assert ds.hinge_loss_perceptron(w, x, y) == hinge_loss_perceptron(w, x, y)


def test_loss_margin_perceptron():
    x, y = ds.read_qsar_androgen_receptor()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1] + 1,))
    print(ds.loss_margin_perceptron(w, x, y))  # 26.947708673531668
    loss_margin_perceptron = ds.LossMarginPerceptron()
    print(loss_margin_perceptron(w, x, y))  # 26.947708673531668
    assert ds.loss_margin_perceptron(w, x, y) == loss_margin_perceptron(w, x, y)


def test_loss_svm():
    x, y = ds.read_qsar_androgen_receptor()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1] + 1,))
    print(ds.loss_svm(w, x, y))  # 36.172708673531666
    loss_svm = ds.LossSVM()
    print(loss_svm(w, x, y))  # 36.172708673531666
    assert ds.loss_svm(w, x, y) == loss_svm(w, x, y)


def test_mpc2023_nonsmooth():
    x, y = ds.read_qsar_androgen_receptor()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1],))
    print(ds.mpc2023_nonsmooth(w, x, y))  # 28.046327372522406
    mpc2023_nonsmooth = ds.MPC2023Nonsmooth()
    print(mpc2023_nonsmooth(w, x, y))  # 28.046327372522406
    assert ds.mpc2023_nonsmooth(w, x, y) == mpc2023_nonsmooth(w, x, y)


def test_read_parkinson_disease_classification():
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


def test_read_semeion_handwritten_digit():
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

    x, y = ds.read_semeion_handwritten_digit(is_10=False)
    assert x.shape == (1593, 256)
    assert y.shape == (1593,)
    assert sum(y == -1) == 1435
    assert sum(y == 1) == 158
    assert sum(y == -1) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    transformer = Normalizer().fit(x)
    print(x)
    x = transformer.transform(x)
    print(x)
    w = 3.0 * np.ones((x.shape[1] + 1,))
    loss = ds.logistic_loss_lr(w, x, y)
    print(loss)  # 27.198973500103
    loss = ds.logistic_loss_l2(w, x, y)
    print(loss)  # 27.924962200667974


def test_read_cnae9():
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

    x, y = ds.read_cnae9(is_10=False)
    assert x.shape == (1080, 856)
    assert y.shape == (1080,)
    assert sum(y == -1) == 960
    assert sum(y == 1) == 120
    assert sum(y == -1) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    transformer = Normalizer().fit(x)
    print(x)
    x = transformer.transform(x)
    print(x)
    w = 3.0 * np.ones((x.shape[1] + 1,))
    loss = ds.logistic_loss_lr(w, x, y)
    print(loss)  # 9.246709678819988
    loss = ds.logistic_loss_l2(w, x, y)
    print(loss)  # 12.81754301215332


def test_read_madelon():
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
    w = 3.0 * np.ones((x.shape[1] + 1,))
    loss = ds.logistic_loss_lr(w, x, y)
    print(loss)  # 34.97093739858563
    loss = ds.logistic_loss_l2(w, x, y)
    print(loss)  # 36.09818739858564

    x, y = ds.read_madelon(is_10=True)
    assert x.shape == (2000, 500)
    assert y.shape == (2000,)
    assert sum(y == 0) == 1000
    assert sum(y == 1) == 1000
    assert sum(y == 0) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    w = np.zeros((x.shape[1] + 1,))
    loss = ds.cross_entropy_loss_lr(w, x, y)
    print(loss)  # 0.6931471805599454
    loss = ds.square_loss_lr(w, x, y)
    print(loss)  # 0.25


def test_read_qsar_androgen_receptor():
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
    loss = ds.tanh_loss_lr(w, x, y)
    print(loss)  # 3.52815643380549

    x, y = ds.read_qsar_androgen_receptor(is_10=True)
    assert x.shape == (1687, 1024)
    assert y.shape == (1687,)
    assert sum(y == 0) == 1488
    assert sum(y == 1) == 199
    assert sum(y == 0) + sum(y == 1) == len(y)
    print(x.dtype, y.dtype)

    w = np.zeros((x.shape[1] + 1,))
    loss = ds.cross_entropy_loss_lr(w, x, y)
    print(loss)  # 0.6931471805599453
    loss = ds.square_loss_lr(w, x, y)
    print(loss)  # 0.25


if __name__ == '__main__':
    test_cross_entropy_loss_lr()
    test_cross_entropy_loss_l2()
    test_logistic_loss_lr()
    test_logistic_loss_l2()
    test_tanh_loss_lr()
    test_hinge_loss_perceptron()
    test_loss_margin_perceptron()
    test_loss_svm()
    test_mpc2023_nonsmooth()

    test_read_parkinson_disease_classification()
    test_read_semeion_handwritten_digit()
    test_read_cnae9()
    test_read_madelon()
    test_read_qsar_androgen_receptor()
