import csv
import math

import matplotlib as matplotlib
import numpy
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier

MAX_DEPTH = 2


def normalize(w):
    sum = 0
    for i in range(len(w)):
        sum += w[i]
    res = []
    for i in range(len(w)):
        res.append(w[i] / sum)
    return res


def get_classifier(xs, ys, T):
    bs = []
    trees = []
    n = len(xs)
    w = [1.0 / n for _ in range(n)]
    for t in range(T):
        tree = DecisionTreeClassifier(splitter='random', max_depth=MAX_DEPTH)
        tree.fit(xs, ys, sample_weight=w)
        trees.append(tree)
        res = tree.predict(xs)
        n_t = 0
        for i in range(n):
            if ys[i] * res[i] < 0:
                n_t += w[i]
        b_t = 0.5 * math.log((1 - n_t) / n_t)
        bs.append(b_t)
        for i in range(n):
            w[i] = w[i] * math.exp(-b_t * ys[i] * res[i])
        w = normalize(w)
    return [bs, trees]


def sgn(u):
    if u < 0:
        return -1
    return 1


def get_class_value(x, classifier):
    sum = 0
    for i in range(len(classifier[0])):
        res = classifier[1][i].predict([x])[0]
        sum += res * classifier[0][i]
    return sum


def get_class(x, classifier):
    return sgn(get_class_value(x, classifier))


def draw_graphic(xs, ys, filename):
    matplotlib.pyplot.plot(xs, ys, "k-")
    for i in range(len(xs)):
        matplotlib.pyplot.plot(xs[i], ys[i], "ro")
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.clf()


def draw_plot(xs, ys, T, graph_file):
    N = 100
    min_x = min(xs, key=lambda x: x[0])[0]
    max_x = max(xs, key=lambda x: x[0])[0]
    min_y = min(xs, key=lambda x: x[1])[1]
    max_y = max(xs, key=lambda x: x[1])[1]
    X, Y = numpy.mgrid[min_x:max_x:complex(0, N), min_y:max_y:complex(0, N)]
    matrix = []
    classifier = get_classifier(xs, ys, T)
    for i in range(len(X)):
        matrix.append([])
        for j in range(len(Y)):
            matrix[i].append(get_class_value([X[i][0], Y[0][j]], classifier))
    fig, ax0 = matplotlib.pyplot.subplots()
    eps = 0.6
    c = ax0.pcolor(X, Y, matrix, cmap='rainbow', vmin=-eps, vmax=eps)
    fig.colorbar(c, ax=ax0)
    for i in range(len(xs)):
        color = "black" if get_class(xs[i], classifier) == ys[i] else "white"
        symb = '+' if ys[i] == 1 else '_'
        matplotlib.pyplot.plot(xs[i][0], xs[i][1], symb, color=color)
    matplotlib.pyplot.savefig(graph_file)


filename = '../data/geyser.csv'
xs = []
ys = []
with open(filename) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        xs.append([float(x) for x in row[:-1]])
        ys.append(float(row[-1]))

tree_cnts = [1, 2, 3, 8, 13, 21, 34, 55]
for T in tree_cnts:
    graph_file = "geyser" + str(T) + ".png"
    draw_plot(xs, ys, T, graph_file)

# for T in tree_cnts:
#     classifier = get_classifier(xs, ys, T)
#     right = 0
#     for i in range(len(xs)):
#         prediction = get_class(xs[i], classifier)
#         if prediction == ys[i]:
#             right += 1
#     accuracy = right / len(xs)
#     print(str(T) + ' ' + str(accuracy))
