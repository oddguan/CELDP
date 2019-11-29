import math
import sys
import numpy as np

from CRFSDP import RandomForest
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from util import load_dataset, get_weight, horizontal_split_data, get_weight


class CRFSDP(object):
    def __init__(self,
                 epsilon,
                 num_learners,
                 depth,
                 train,
                 test,
                 part=5):
        self.epsilon = epsilon
        self.num_learners = num_learners
        self.depth = depth
        self.part = part
        self.train = train
        self.test = test

    def single_RFsDP(self, X, y, epsilon, num_learner, depth, n_rows):
        _lambda = float(X.shape[0] / n_rows)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        rfs_prediction = RandomForest(X_train,
                                      y_train,
                                      epsilon,
                                      num_learner,
                                      depth,
                                      1)
        predRF_p = rfs_prediction.predict(X_test)
        accuracy = accuracy_score(y_test, predRF_p, normalize=True)
        lambda_accuracy = math.exp(_lambda) * accuracy
        return lambda_accuracy, rfs_prediction

    def parallel(self):
        X, y = load_dataset(self.train)
        n = X.shape[0]
        X, y = horizontal_split_data(X, y, self.part)
        X_test, y_test = load_dataset(self.test)
        lambda_accuracy = np.zeros(self.part)
        F = np.zeros(self.part)
        random_forests = []
        for i in self.num_learners:
            for j in self.depth:
                test_prediction = np.zeros(y_test.shape[0], dtype=float)
                for p in range(self.part):
                    lambda_accuracy[p], random_forest_prediction = self.single_RFsDP(
                        X[p], y[p], self.epsilon[p], i, j, n)
                    random_forests.append(random_forest_prediction)
                F = get_weight(lambda_accuracy)
                for p in range(self.part):
                    test_prediction += random_forests[p].predict(X_test) * F[p]
                test_prediction = [0 if pred <= 0.5 else 1 for pred in test_prediction]
                outputFile = open("crfsdp_output.txt", 'a')
                outputFile.write(
                    f1_score(y_test, test_prediction, average="micro") + '\n')
                outputFile.write(roc_auc_score(y_test, test_prediction) + '\n')
                outputFile.close()
