from DecisionTree import DecisionTree
from Ensemble import Ensemble


def RandomForest(X, y, epsilon, num_learners, depth, min_sample):
    dt = DecisionTree(epsilon, min_sample, depth)
    result = Ensemble(dt, num_learners, 0.8)
    result.fit(X, y)
    return result
