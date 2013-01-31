"""
hand-rolled version of R's pairs plot
"""

import numpy
import sys
import itertools
import matplotlib
matplotlib.rc('font', family='Sans-Serif', weight='bold', size=16)
import pylab

if len(sys.argv) != 2:
    print 'usage: data.csv'
    sys.exit(1)

file_name = sys.argv[1]
data = numpy.loadtxt(file_name, delimiter=',', skiprows=1)
col_names = numpy.genfromtxt(file_name, delimiter=',', names=True).dtype.names

x = data[:, :-1]
y = data[:, -1]
y_values = numpy.unique(y)
y_classes = numpy.digitize(y, y_values) - 1

colours = 'brgymck'
y_colours = [colours[i] for i in y_classes]

n_examples = x.shape[0]
n_cols = x.shape[1]
for i, j in itertools.product(range(n_cols), repeat=2):
    pylab.subplot(n_cols, n_cols, (n_cols-j-1)*n_cols + i + 1)
    if i != j:
        pylab.scatter(x[:, i], x[:, j], c=y_colours)
    pylab.xticks([])
    pylab.yticks([])
    if j == 0:
        pylab.xlabel(str(col_names[i]))
    if i == 0:
        pylab.ylabel(str(col_names[j]), rotation=45)
delta = 0.10
gamma = 0.01
pylab.subplots_adjust(left=delta, right=1-delta, bottom=delta,
    top=1-delta, wspace=gamma, hspace=gamma)

title = 'dataset "%s" : %d examples, %d features, %d classes' % (
    file_name, n_examples, n_cols, len(y_values))
pylab.suptitle(title)
pylab.show()

