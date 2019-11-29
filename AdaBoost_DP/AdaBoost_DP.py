import numpy as np
import sys
import math
from util import load_dataset, horizontal_split_data, get_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from AdaBoost import Adaboost_Classifier

class Collaborative_AdaBoost_DP(object):
    def __init__(self, number, depth, epsilon, train_set, test_set, part):
        self.number_of_learners = number
        self.height = depth
        self.privacy_epsilon = epsilon
        self.train_set = train_set
        self.test_set = test_set
        self.part = part

    def AdaBoost_DP(self, X, y, epsilon, depth, number_of_learners, rows):
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2)
        lambda_ = float(X.shape[0] / rows)
        adaboost_instance = Adaboost_Classifier(number_of_learners, depth, epsilon)
        adaboost_instance.fit(X_train, y_train)
        adaboost_predict_instance = adaboost_instance.predict(X_test)
        lambda_acc = math.exp(lambda_) * (accuracy_score(y_test, adaboost_predict_instance, normalize=True))
        return lambda_acc, adaboost_instance

    def parallel(self):
        X_train, y_train = load_dataset(self.train_set)
        X_test, y_test = load_dataset(self.test_set)
        rows = X_train.shape[0]
        hor_X, hor_y = horizontal_split_data(X_train, y_train, self.part)
        lambda_accurancy = np.zeros(self.part)

        weights = np.zeros(self.part)

        for i in self.number_of_learners:
            for j in self.height:
                predict_of_test = np.zeros(y_test.shape[0])
                adaboost_set = []
                for k in hor_X:
                    lambda_accurancy[k], adaboost_instance = self.AdaBoost_DP(hor_X[k], hor_y[k], self.privacy_epsilon[k], j, i, rows)
                    adaboost_set.append(adaboost_instance)

                weights = get_weight(lambda_accurancy)
                for k in range(self.part):
                    predict_of_test += weights[k] * adaboost_set[k].predict(X_test)
                predict_of_test = [0.0 if predict <= 0.5 else 1.0 for predict in predict_of_test]
                outputFile = open("adaboostdp_output.txt", 'a')
                outputFile.write(
                    f1_score(y_test, predict_of_test, average="micro") + '\n')
                outputFile.write(roc_auc_score(y_test, predict_of_test) + '\n')
                outputFile.close()