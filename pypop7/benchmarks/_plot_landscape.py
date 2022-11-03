from pypop7.benchmarks.utils import plot_contour
from pypop7.benchmarks.base_functions import ackley, rastrigin
from pypop7.benchmarks.rotated_functions import generate_rotation_matrix, ellipsoid


def cd(x):  # from https://arxiv.org/pdf/1610.00040v1.pdf
    return 7*(x[0]**2) + 6*x[0]*x[1] + 8*(x[1]**2)


if __name__ == '__main__':
    # plot multi-modality
    plot_contour(rastrigin, [-10, 10], [-10, 10])
    # plot non-convexity
    plot_contour(ackley, [-10, 10], [-10, 10])
    # plot non-separability
    plot_contour(cd, [-10, 10], [-10, 10])
    # plot ill-condition
    generate_rotation_matrix(ellipsoid, 2, 72)
    plot_contour(ellipsoid, [-10, 10], [-10, 10],
                 levels=[1e0, 1e3, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9])
