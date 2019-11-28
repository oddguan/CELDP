import numpy as np
from sklearn.base import clone, BaseEstimator


class Ensemble(BaseEstimator):

    def __init__(self, model, num_learners, feat_percent):
        self.model = model
        self.num_learners = num_learners
        self.feat_percent = feat_percent
        self.learners = []
        self.weights = np.ones(self.num_learners) / self.num_learners

    def init_fit(self, X, y):
        self.classes = np.unique(y)
        self.classlist = list(self.classes)

    def fit(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)
        m, n = X.shape
        self.init_fit(X, y)
        if self.feat_percent < 1:
            sub_features = int(self.feat_percent * n)
            self.feature_subset = []
        else:
            sub_features = n
            self.feature_subset = None

        for _ in range(self.num_learners):
            idx = np.random.random_integers(0, m - 1, m)
            X_subset = X[idx, :]
            if sub_features < n:
                features_idx = np.random.permutation(n)[:sub_features]
                self.feature_subset.append(features_idx)
                X_subset = X_subset[:, features_idx]
            y_subset = y[idx]
            learner = clone(self.model)
            learner.fit(X_subset, y_subset)
            self.learners.append(learner)

    def predict(self, X):
        X = np.atleast_2d(X)
        n_rows, _ = X.shape
        n_classes = len(self.classes)
        votes = np.zeros([n_rows, n_classes])

        for i, learner in enumerate(self.learners):  # G->gt
            w = self.weights[i]
            if self.feature_subset is not None:
                features_ind = self.feature_subset[i]
                X_subset = X[:, features_ind]
                pred_y = learner.predict(X_subset)
            else:
                pred_y = learner.predict(X)

            for c in self.classes:
                c_ind = self.classlist.index(c)
                votes[pred_y == c, c_ind] += w

        majority_index = np.argmax(votes, axis=1)
        return np.array([self.classlist[i] for i in majority_index])
