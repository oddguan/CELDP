import math

import numpy as np
from sklearn.base import BaseEstimator

from Leaf import Leaf
from Node import Node
from util import midPoints, majority, laplace


class DecisionTree(BaseEstimator):
    def __init__(self, epsilon, min_sample=1, depth=6):
        self.root = None
        self.classes = None
        self.epsilon = epsilon
        self.min_sample = min_sample
        self.depth = depth
        self.n_leaf = 0

    def getThresholds(self, x):
        if len(x) > 1:
            return midPoints(np.unique(x))
        return x

    def splitData(self, X, y, h, sampleWeight, epsilon):
        nRows, nFeatures = X.shape
        if nRows <= self.min_sample or h > self.depth:
            self.n_leaf += 1
            leaf = majority(y, self.classes)
            if epsilon > 0.0:
                leaf += math.floor(laplace(epsilon))
            return Leaf(leaf)
        elif np.all(y == y[0]):
            self.n_leaf += 1
            return Leaf(y[0])
        else:
            bestSplitScore = int('inf')
            bestFeatureIndex = None
            bestThreshold = None
            for i in range(nFeatures):
                colVector = X[:, i]
                featureVector = self.getThresholds(colVector)
                threshold, totalScore = self.getGiniSplit(
                    colVector, featureVector, y, sampleWeight)
                if threshold is not None:
                    if totalScore < bestSplitScore:
                        bestSplitScore = totalScore
                        bestFeatureIndex = i
                        bestThreshold = threshold
                else:
                    break
            if bestFeatureIndex is not None:
                if epsilon > 0.0:
                    bestThreshold += math.floor(laplace(0.5 * epsilon))
                leftBranch = X[:, bestFeatureIndex] < bestThreshold
                rightBranch = ~leftBranch
                leftData = X[leftBranch, :]
                rightData = X[rightBranch, :]
                leftLabels = y[leftBranch]
                rightLabels = y[rightBranch]

                epsilonLeft = epsilonRight = 0
                if leftData.shape[0] == 0 and rightData.shape[0] != 0:
                    epsilonRight = 0.5 * epsilon
                elif leftData.shape[0] != 0 and rightData.shape[0] == 0:
                    epsilonLeft = 0.5 * epsilon
                else:
                    epsilonLeft = 0.25 * epsilon
                    epsilonRight = epsilonLeft
                del y
                del X
                del leftBranch
                del rightBranch
                leftTree = self.splitData(
                    leftData, leftLabels, h+1, sampleWeight, epsilonLeft)
                rightTree = self.splitData(
                    rightData, rightLabels, h+1, sampleWeight, epsilonRight)
                return Node(bestFeatureIndex, bestThreshold, leftTree, rightTree)

    def getGiniSplit(self, colVector, thresholds, y, sampleWeight):
        resultScore = int('inf')
        resultThresh = None
        n = len(y)
        for threshold in thresholds:
            index = colVector < threshold
            leftLabels = y[index]
            rightLabels = y[~index]
            leftScore = self.Gini(leftLabels, sampleWeight)
            rightScore = self.Gini(rightLabels, sampleWeight)
            score = (len(leftScore) / n) * leftScore + \
                (len(rightScore) / n) * rightScore
            if score < resultScore:
                resultScore = score
                resultThresh = threshold
        return resultThresh, resultScore

    def Gini(self, y, sampleWeight):
        _sum = 0.0
        n = len(y)
        if n == 0:
            return 0.0
        n2 = float(n * n)
        if sampleWeight is not None:
            for c in self.classes:
                count = np.sum(y == c)
                index = np.where(y == c)[0]
                w = np.sum(sampleWeight[index])
                c2 = ((count * w) ** 2) / n2
                _sum += c2
        else:
            for c in self.classes:
                count = np.sum(y == c)
                c2 = (count ** 2) / n2
                _sum += c2
        return 1 - _sum

    def fit(self, X, y, sampleWeight=None):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_leaf = 0
        self.root = self.splitData(X, y, 1, sampleWeight, self.epsilon)

    def predict(self, X):
        X = np.atleast_2d(X)
        nRows = X.shape[0]
        outputs = np.zeros(nRows)
        index = np.ones(nRows, dtype='bool')
        if self.root is not None:
            self.root.setPredict(X, outputs, index)
        return outputs
