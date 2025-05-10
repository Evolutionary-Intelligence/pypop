import matplotlib
import matplotlib.pyplot as plt

from pypop7.benchmarks.utils import generate_xyz
from pypop7.benchmarks.base_functions import cigar as bf


font_size = 10
font_family = 'Times New Roman'
matplotlib.rcParams['font.family'] = font_family
matplotlib.rcParams['font.size'] = font_size


if __name__ == '__main__':
    bound = [-10.0, 10.0]
    fig = plt.figure(figsize=(2.5, 2.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x, y, z = generate_xyz(bf, bound, bound, 300)
    ax.plot_surface(x, y, z, cmap='Blues',
                    linewidth=0, antialiased=False)
    ax.set_title(bf.__name__, fontsize=font_size)
    ax.set_xlim(bound)
    ax.set_ylim(bound)
    ax.set_xticks([-10.0, 0.0, 10.0],
                  ['-10.0', '0.0', '10.0'],
                  fontsize=font_size, fontfamily=font_family)
    ax.set_yticks([-10.0, 0.0, 10.0],
                  ['-10.0', '0.0', '10.0'],
                  fontsize=font_size, fontfamily=font_family)
    ax.set_zticks([3e7, 6e7, 9e7],
                  ['3e7', '6e7', '9e7'],
                  fontsize=font_size, fontfamily=font_family)
    ax.set_xlabel('X1', fontsize=font_size)
    ax.set_ylabel('X2', fontsize=font_size)
    ax.set_zlabel('Fitness', fontsize=font_size,
                  rotation='vertical')
    plt.savefig('surface_{0}.png'.format(bf.__name__),
                dpi=700, bbox_inches='tight')
    plt.show()
