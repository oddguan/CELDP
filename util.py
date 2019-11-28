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
