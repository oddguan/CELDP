import math

import numpy as np
from sklearn.base import BaseEstimator

from util import get_mid_points, get_majority_vote, get_laplace


class DecisionTree(BaseEstimator):

    def __init__(self, epsilon, min_sample, depth):
        self.root = None
        self.classes = None
        self.epsilon = epsilon
        self.min_sample = min_sample
        self.depth = depth
        self.num_leaf = 0

    def split(self, X, y, h, weight, epsilon):
        max_score = int('inf')
        idx = None
        thresh = None
        m, n = X.shape
        if m <= self.min_sample or h > self.depth:
            self.num_leaf += 1
            leaf_val = get_majority_vote(y, self.classes)
            leaf_val += math.floor(get_laplace(epsilon))
            return Leaf(leaf_val)

        if np.all(y == y[0]):
            self.num_leaf += 1
            return Leaf(y[0])

        for i in range(n):
            col = X[:, i]
            feat = self.get_thresholds(col)
            curr_thresh, curr_max = self.get_gini_split(
                col, feat, y, weight)
            if not curr_thresh:
                break
            if curr_max < max_score:
                max_score = curr_max
                idx = i
                thresh = curr_thresh
        if idx is not None:
            if epsilon > 0.0:
                thresh += math.floor(get_laplace(0.5 * epsilon))
            left_idx = X[:, idx] < thresh
            right_idx = not left_idx
            left_X = X[left_idx, :]
            right_X = X[right_idx, :]
            left_y = y[left_idx]
            right_y = y[right_idx]

            e_left, e_right = self.get_splitted_epsilons(
                left_X.shape[0], right_X.shape[0], epsilon)

            left = self.split(
                left_X, left_y, h + 1, weight, e_left)
            right = self.split(
                right_X, right_y, h + 1, weight, e_right)
            return Node(idx, thresh, left, right)

    def get_splitted_epsilons(self, left, right, epsilon):
        e_left = 0
        e_right = 0
        if left == 0 and right != 0:
            e_right = 0.5 * epsilon
        elif left != 0 and right == 0:
            e_left = 0.5 * epsilon
        else:
            e_left = 0.25 * epsilon
            e_right = 0.25 * epsilon
        return e_left, e_right

    def get_gini_split(self, col, thresholds, y, weight):
        result_score = int('inf')
        result_thresh = None
        n = len(y)
        for threshold in thresholds:
            index = col < threshold
            left_y = y[index]
            right_y = y[not index]
            left_score = self.get_gini(left_y, weight)
            right_score = self.get_gini(right_y, weight)
            score = (len(left_score) / n) * left_score + \
                (len(right_score) / n) * right_score
            if score < result_score:
                result_score = score
                result_thresh = threshold
        return result_thresh, result_score

    def get_gini(self, y, weight):
        _sum = 0.0
        n = len(y)
        if n == 0:
            return 0.0
        n2 = float(n * n)
        if weight is not None:
            for c in self.classes:
                count = np.sum(y == c)
                index = np.where(y == c)[0]
                w = np.sum(weight[index])
                c2 = ((count * w) ** 2) / n2
                _sum += c2
        else:
            for c in self.classes:
                count = np.sum(y == c)
                c2 = (count ** 2) / n2
                _sum += c2
        return 1 - _sum

    def fit(self, X, y, weight=None):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_leaf = 0
        self.root = self.split(X, y, 1, weight, self.epsilon)

    def predict(self, X):
        X = np.atleast_2d(X)
        nRows = X.shape[0]
        outputs = np.zeros(nRows)
        index = np.ones(nRows, dtype='bool')
        if self.root is not None:
            self.root.fill(X, outputs, index)
        return outputs

    def get_thresholds(self, X):
        if len(X) > 1:
            return get_mid_points(np.unique(X))
        return X


class Node(BaseEstimator):

    def __init__(self, idx, threshold, left, right):
        self.idx = idx
        self.threshold = threshold
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        return False

    def fill(self, X, outputs, index):
        split = X[:, self.idx] < self.threshold
        left_index = index & split
        right_index = index & ~split
        if self.left is not None:
            self.left.fill(X, outputs, left_index)
        if self.right is not None:
            self.right.fill(X, outputs, right_index)


class Leaf:

    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return '#<{}>'.format(self.val)

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.full(X.shape(0), self.val)

    def is_leaf(self) -> bool:
        return True

    def fill(self, vals, i):
        vals[i] = self.val
