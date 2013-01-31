import matplotlib
matplotlib.rc('font', family='Serif', weight='bold', size=24)

from numpy import linspace, meshgrid, ravel, subtract, exp, zeros_like, random, reshape
from scipy import ndimage

def make_surface(n, length_scale):
    # set up arrays of x and y coordinates defining a grid
    t = linspace(-1.0, 1.0, n)
    x, y = meshgrid(t, t)

    x = ravel(x)
    y = ravel(y)

    # compute the square Euclidean distance between each pair of grid points
    delta_x = subtract.outer(x, x)
    delta_y = subtract.outer(y, y)
    dist_squared = delta_x ** 2 + delta_y ** 2

    # sample a Gaussian process over the grid using a square-exponential covariance function
    cov = exp(-dist_squared/(2.0*(length_scale**2)))
    mean = zeros_like(x)
    surface = random.multivariate_normal(mean, cov)

    # reshape the sampled values into a 2d array (matching the grid layout)
    surface_2d = reshape(surface, (n, n))

    # resize the sampled values using interpolation (cheap way of faking higher resolution)
    surface_2d_interp = ndimage.zoom(surface_2d, 5)
    return surface_2d_interp



from pylab import gca, contourf, contour, show, colorbar, subplot, subplots_adjust, cm, suptitle

plots_across = 3
plots_down = 2

for plot_i in xrange(plots_across):
    for plot_j in xrange(plots_down):
        surface = make_surface(20, length_scale=random.uniform(0.1, 0.8))
        subplot(plots_down, plots_across, plot_j * plots_across + plot_i + 1)

        n_levels = 15
        contourf(surface, n_levels, zorder=0, cmap=cm.Spectral_r)
        contour(surface, n_levels, colors=('k', ), zorder=1)
        axes = gca()
        # axes.set_aspect(1.0)
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
subplots_adjust(wspace=0.05, hspace=0.05)
suptitle('my gaussian processes, let me show you them')
show()

