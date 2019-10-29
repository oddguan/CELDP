import numpy as np


class Leaf:
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return '#<{}>'.format(self.val)

    def predict(self, X):
        X = np.atleast_2d(X)
        output = np.zeros(X.shape[0])
        output[:] = self.val
        return output

    def setPredict(self, X, output, index):
        output[index] = self.val
