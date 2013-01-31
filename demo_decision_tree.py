"""
decision tree classifier demo

requires:
    scikit-learn
    numpy
    pydot etc to output pretty plots
"""


# command line setup
import sys

def pause():
    print
    print '*** intermission ***'
    raw_input()

if len(sys.argv) not in (2, 3, 4, 5):
    print 'usage: train.csv [test.csv [min_samples_split [min_samples_leaf]]]'
    sys.exit(1)

train_file_name = sys.argv[1]
test_file_name = sys.argv[2] if len(sys.argv) > 2 else None
min_samples_split = int(sys.argv[3]) if len(sys.argv) > 3 else 2
min_samples_leaf = int(sys.argv[4]) if len(sys.argv) > 4 else 1



# read training data set from csv, load into 2d numpy array
import numpy
train_data = numpy.loadtxt(train_file_name, delimiter=',', skiprows=1)

print 'got training data of shape %s from file "%s":' % (
    str(train_data.shape), train_file_name)
print train_data

pause()



# fit a decision tree classifier to the training data
print 'fitting decision tree classifier'
print '\tmin_samples_split = %d' % min_samples_split
print '\tmin_samples_leaf = %d' % min_samples_leaf


from sklearn import tree

x_train = train_data[:, :-1]
y_train = train_data[:, -1]

model = tree.DecisionTreeClassifier(
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf)

model.fit(x_train, y_train)

pause()



# measure the classifier's error on the training data
print 'making predictions on training data'

y_train_predictions = model.predict(x_train)

error = lambda expected, predicted : numpy.mean(expected != predicted)
train_error = error(y_train, y_train_predictions)

print 'average 0-1 error on training data: %.3f' % train_error
pause()



# plot the tree to a pdf so we can look at it
import StringIO, pydot
dot_data = StringIO.StringIO()
col_names = numpy.genfromtxt(train_file_name, delimiter=',', names=True).dtype.names # hack
tree.export_graphviz(model, out_file=dot_data, feature_names=col_names)
dot_data = dot_data.getvalue()
graph_file_name = 'model_tree_%d.pdf' % len(dot_data)
print 'plotting tree to "%s"' % graph_file_name
graph = pydot.graph_from_dot_data(dot_data)
graph.write_pdf(graph_file_name)
import subprocess
subprocess.call(['evince', graph_file_name])

pause()



# make predictions on test data set and measure error
if test_file_name:
    test_data = numpy.loadtxt(test_file_name, delimiter=',', skiprows=1)
    print 'got test data of shape %s from file "%s":' % (
        str(test_data.shape), test_file_name)
    print test_data
    pause()

    print 'making predictions on test data'

    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    y_test_predictions = model.predict(x_test)
    test_error = error(y_test, y_test_predictions)

    print 'average 0-1 error on test data: %.3f' % test_error
    pause()

