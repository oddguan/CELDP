import numpy as np
import math
import random


def get_mid_points(x):
    return (x[1:] + x[:-1]) / 2


def get_majority_vote(y, classes):
    result = classes
    if result is None:
        result = np.unique(y)
    votes = np.zeros(len(result))
    for i, c in enumerate(result):
        votes[i] = np.sum(y == c)
    majority_index = np.argmax(votes)
    return result[majority_index]


def sign(a):
    return 0 if a == 0 else 1 if a > 0 else -1


def get_laplace(epsilon):
    if epsilon <= 0:
        return 0
    mu = 0
    b = 1.0 / epsilon
    a = random.uniform(-0.5, 0.5)
    return mu - b * sign(a) * math.log(1 - 2 * abs(a))


def load_dataset(file):
    f = open(file)
    n = len(f.readline().split(','))
    X = []
    y = []
    for line in f.readlines():
        curr_X = []
        line = line.strip().split(',')
        for i in range(n - 1):
            curr_X.append(float(line[i]))
        X.append(curr_X)
        y.append(float(line[-1]))
    return np.array(X), np.array(y)


def get_weight(w):
    result = []
    for w_p in w:
        F_p = w_p / np.sum(w)
        result.append(F_p)
    return result


def horizontal_split_data(X, y, part=5):
    if part == 1:
        return X, y
    else:
        n = X.shape[0]
        random.seed(114514)
        values = random.sample(range(1, n), part - 1)
        values.sort()
        values.append(n)
        k = 0
        j = 0
        result_X = {}
        result_y = {}
        for i in values:
            result_X[j] = X[k:i]
            result_y[j] = y[k:i]
            k = i
            j += 1
        return result_X, result_y
