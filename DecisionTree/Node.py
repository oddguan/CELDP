from sklearn.base import BaseEstimator


class Node(BaseEstimator):
    def __init__(self, featureIdx, threshold, leftTree, rightTree):
        self.featureIdx = featureIdx
        self.threshold = threshold
        self.leftTree = leftTree
        self.rightTree = rightTree

    def setPredict(self, X, outputs, index):
        split = X[:, self.featureIdx] < self.threshold
        leftIndex = index & split
        rightIndex = index & ~split
        if self.leftTree is not None:
            self.leftTree.setPredict(X, outputs, leftIndex)
        if self.rightTree is not None:
            self.rightTree.setPredict(X, outputs, rightIndex)
