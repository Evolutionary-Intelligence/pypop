import numpy as np  # engine for numerical computing
from numpy import genfromtxt  # to read datasets

from pypop7.benchmarks.base_functions import BaseFunction


def cross_entropy_loss_lr(w, x, y):
    """Cross-Entropy Loss Function of Logistic Regression (LR with binary labels/classes {0, 1}).

        Note that this loss function for binary (two-class) classification is always convex
          regardless of the used dataset. It is very often used in practice to perform LR.

        https://jermwatt.github.io/machine_learning_refined/notes/6_Linear_twoclass_classification/6_2_Cross_entropy.html
        https://openreview.net/forum?id=BJe-DsC5Fm (2019)
    """
    loss = np.empty(len(y))
    for i in range(len(y)):
        p = 1.0/(1.0 + np.exp(-(w[0] + np.dot(x[i], w[1:]))))
        loss[i] = -y[i]*np.log(p) - (1.0 - y[i])*np.log(1.0 - p)
    return np.mean(loss)


class CrossEntropyLossLR(BaseFunction):
    def __call__(self, w, x, y):
        return cross_entropy_loss_lr(w, x, y)


def cross_entropy_loss_l2(w, x, y):
    """"Cross-Entropy Loss Function with L2-Regularization of Logistic Regression (LR with binary labels {0, 1}).

        https://jermwatt.github.io/machine_learning_refined/ (2020)
        https://epubs.siam.org/doi/abs/10.1137/17M1154679?journalCode=sjope8 (2018)
    """
    return cross_entropy_loss_lr(w, x, y) + np.sum(np.square(w))/(2.0*len(y))


class CrossEntropyLossL2(BaseFunction):
    def __call__(self, w, x, y):
        return cross_entropy_loss_l2(w, x, y)


def square_loss_lr(w, x, y):
    """Square Loss Function of Logistic Regression (LR with binary labels/classes {0, 1}).

        Note that this loss function for binary classification is generally non-convex (non-linear least squares).

        https://jermwatt.github.io/machine_learning_refined/ (2020)
        https://openreview.net/forum?id=ryxz8CVYDH (2020)
        https://epubs.siam.org/doi/abs/10.1137/1.9781611976236.23 (2020)
        https://openreview.net/forum?id=BJe-DsC5Fm (2019)
        https://proceedings.neurips.cc/paper/2018/file/ba9a56ce0a9bfa26e8ed9e10b2cc8f46-Paper.pdf (2018)
        https://epubs.siam.org/doi/abs/10.1137/17M1154679?journalCode=sjope8 (2018)
    """
    loss = np.empty(len(y))
    for i in range(len(y)):
        loss[i] = np.square(y[i] - 1.0/(1.0 + np.exp(-(w[0] + np.dot(x[i], w[1:])))))
    return np.mean(loss)


class SquareLossLR(BaseFunction):
    def __call__(self, w, x, y):
        return square_loss_lr(w, x, y)


def logistic_loss_lr(w, x, y):
    """Logistic Loss Function of Logistic Regression (LR with binary labels/classes {-1, 1}).

        AKA softmax cost (always convex regardless of the dataset used).

        https://www.tandfonline.com/doi/full/10.1080/00031305.2021.2006781
        https://github.com/jermwatt/machine_learning_refined/blob/main/notes/6_Linear_twoclass_classification/6_3_Softmax.ipynb
    """
    loss = np.empty(len(y))
    for i in range(len(y)):
        loss[i] = np.log(1.0 + np.exp(-y[i]*(w[0] + np.dot(x[i], w[1:]))))
    return np.mean(loss)


class LogisticLossLR(BaseFunction):
    def __call__(self, w, x, y):
        return logistic_loss_lr(w, x, y)


def logistic_loss_l2(w, x, y):
    """Logistic Loss Function with L2-Regularization of Logistic Regression (LR with binary labels/classes {-1, 1}).

        https://epubs.siam.org/doi/abs/10.1137/17M1154679?journalCode=sjope8 (2018)
    """
    return logistic_loss_lr(w, x, y) + np.sum(np.square(w))/(2.0*len(y))


class LogisticLossL2(BaseFunction):
    def __call__(self, w, x, y):
        return logistic_loss_l2(w, x, y)


def tanh_loss_lr(w, x, y):
    """Tanh Loss Function of Logistic Regression (LR with binary labels/classes {-1, 1}).

        Note that this loss function for binary classification is generally non-convex (non-linear least squares).

        https://github.com/jermwatt/machine_learning_refined/blob/main/notes/6_Linear_twoclass_classification/6_3_Softmax.ipynb
    """
    loss = np.empty(len(y))
    for i in range(len(y)):
        loss[i] = np.square(2.0/(1.0 + np.exp(-(w[0] + np.dot(x[i], w[1:])))) - 1.0 - y[i])
    return np.mean(loss)


class TanhLossLR(BaseFunction):
    def __call__(self, w, x, y):
        return tanh_loss_lr(w, x, y)


def hinge_loss_perceptron(w, x, y):
    """Hinge Loss Function of Perceptron.

        AKA perceptron cost, rectified linear unit cost.

        This cost function is always convex but only has a single discontinuous derivative in each variable dimension.
        Note that it always has a trivial solution at the origin, thus one may need to take care in practice to avoid
        finding it (or a point too close to it) **accidentally**.

        https://colab.research.google.com/github/jermwatt/machine_learning_refined/blob/main/notes/6_Linear_twoclass_classification/6_4_Perceptron.ipynb
    """
    loss = np.empty(len(y))
    for i in range(len(y)):
        loss[i] = np.max(0, -y[i]*(w[0] + np.dot(x[i], w[1:])))
    return np.mean(loss)


class HingeLossPerceptron(BaseFunction):
    def __call__(self, w, x, y):
        return hinge_loss_perceptron(w, x, y)


def loss_margin_perceptron(w, x, y):
    """Loss Function of Margin Perceptron (with binary labels/classes {-1, 1}).

        https://jermwatt.github.io/machine_learning_refined/ (2020)
    """
    loss = np.empty(len(y))
    for i in range(len(y)):
        loss[i] = np.maximum(0.0, 1.0 - y[i]*(w[0] + np.dot(x[i], w[1:])))
    return np.mean(loss)


class LossMarginPerceptron(BaseFunction):
    def __call__(self, w, x, y):
        return loss_margin_perceptron(w, x, y)


def loss_svm(w, x, y, r=None):
    """Loss Function of Support Vector Machines (SVM with binary labels/classes {-1, 1}).

        https://jermwatt.github.io/machine_learning_refined/notes/6_Linear_twoclass_classification/6_5_SVMs.html (2020)
    """
    if r is None:
        r = 1e-3
    loss = np.empty(len(y))
    for i in range(len(y)):
        loss[i] = np.maximum(0.0, 1.0 - y[i]*(w[0] + np.dot(x[i], w[1:])))
    return np.mean(loss) + r*np.sum(np.square(w))


class LossSVM(BaseFunction):
    def __call__(self, w, x, y, r=None):
        return loss_svm(w, x, y, r=None)


def mpc2023_nonsmooth(w, x, y):
    """Nonsmooth Function from MPC-2023.

        https://link.springer.com/article/10.1007/s12532-023-00233-9 (2023)
    """
    loss = np.empty(len(y))
    for i in range(len(y)):
        loss[i] = np.abs(np.dot(w, x[i]) - y[i])
    return np.mean(loss)


def read_parkinson_disease_classification():
    """Sakar,C., Serbes,Gorkem, Gunduz,Aysegul, Nizam,Hatice, and Sakar,Betul. (2018).
        Parkinson's Disease Classification.
        UCI Machine Learning Repository.
        https://doi.org/10.24432/C5MS4X

        # Data: https://archive.ics.uci.edu/static/public/470/parkinson+s+disease+classification.zip
        # Instances: 756
        # Features: 753 (Delete `id` from 754 Features)
        # Class: 0/1
        # Missing Values: No
    """
    d = genfromtxt('pd_speech_features.csv', delimiter=',')
    x, y = d[2:, 1:-1], d[2:, -1]
    return x, y


def read_semeion_handwritten_digit(is_10=True):
    """Tactile,Srl, Massimo,Buscema, and Stefano,Terzi (1994).
        Semeion Handwritten Digit.
        UCI Machine Learning Repository.
        https://doi.org/10.24432/C5SC8V

        # Data: https://archive.ics.uci.edu/static/public/178/semeion+handwritten+digit.zip
        # Instances: 1593
        # Features: 256 (For 266 Features: the last 10 columns are class labels for digit 0 - 9)
        # Class: 0/1
        # Missing Values: No
    """
    d = genfromtxt('semeion.data', delimiter=' ')
    x, y = d[:, :256], d[:, -1]  # only to classify digit 9
    if not is_10:
        y[y == 0] = -1
    return x, y


def read_cnae9(is_10=True):
    """Ciarelli,Patrick and Oliveira,Elias. (2012).
        CNAE-9.
        UCI Machine Learning Repository.
        https://doi.org/10.24432/C51G7P.

        # Data: https://archive.ics.uci.edu/static/public/233/cnae+9.zip
        # Instances: 1080
        # Features: 856
        # Class: 0/1
        # Missing Values: No
    """
    d = genfromtxt('CNAE-9.data', delimiter=',')
    x, y = d[:, 1:], d[:, 0]
    index_not_9, index_9 = y != 9, y == 9
    if is_10:
        y[index_not_9], y[index_9] = 0, 1
    else:
        y[index_not_9], y[index_9] = -1, 1
    return x, y


def read_madelon(is_10=False):
    """Guyon,Isabelle. (2008).
        Madelon.
        UCI Machine Learning Repository.
        https://doi.org/10.24432/C5602H.

        # Data: https://archive.ics.uci.edu/static/public/171/madelon.zip
        # Instances: 2000
        # Features: 500
        # Class: -1/1
        # Missing Values: No
    """
    x = genfromtxt('madelon_train.data', delimiter=' ')
    y = genfromtxt('madelon_train.labels')
    if is_10:  # to convert all labels with -1 to 0
        y[y == -1] = 0
    return x, y


def read_qsar_androgen_receptor(is_10=False):
    """QSAR androgen receptor. (2019).
        UCI Machine Learning Repository.
        https://doi.org/10.24432/C53317.

        # Data: https://archive.ics.uci.edu/static/public/509/qsar+androgen+receptor.zip
        # Instances: 1687
        # Features: 1024
        # Class: -1/1
        # Missing Values: No
    """
    d = genfromtxt('qsar_androgen_receptor.csv', delimiter=';')
    x, y = d[:, :-1], d[:, -1]
    if is_10:  # to convert all labels with -1 to 0
        y[y == -1] = 0
    return x, y
