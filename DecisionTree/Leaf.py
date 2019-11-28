import numpy as np


class Leaf:
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return '#<{}>'.format(self.val)

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.full(X.shape(0), self.val)
