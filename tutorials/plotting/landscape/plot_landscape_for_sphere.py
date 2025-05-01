import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pypop7.benchmarks.base_functions import sphere
from pypop7.benchmarks.utils import generate_xyz


font_size = 10
font_family = 'Times New Roman'
matplotlib.rcParams['font.family'] = font_family
matplotlib.rcParams['font.size'] = font_size


# helper function for 2D-plotting
def plot_contour(func, x, y):
    levels = [sphere(i * np.ones((2,))) for i in np.arange(0, 12, 2)]
    x, y, z = generate_xyz(func, x, y, 500)
    plt.contourf(x, y, z, cmap='Blues', levels=levels)
    plt.contour(x, y, z, colors='white', levels=levels)


if __name__ == '__main__':
    bound=[-10.0, 10.0]
    plt.figure(figsize=(2.5, 2.5))
    plt.title('Sphere', fontsize=font_size)
    plt.xlim(bound)
    plt.ylim(bound)
    plt.xticks(fontsize=font_size, fontfamily=font_family)
    plt.yticks(fontsize=font_size, fontfamily=font_family)
    plot_contour(sphere, bound, bound)
    plt.xlabel('X1', fontsize=font_size)
    plt.ylabel('X2', fontsize=font_size, labelpad=-1)
    plt.savefig(str(sphere.__name__) + '.png',
                dpi=700, bbox_inches='tight')
    plt.show()
