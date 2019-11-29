import numpy as np
from sklearn.base import clone
from DecisionTree import DecisionTree

class Adaboost_Classifier(object):

    def __init__(self, number, depth, epsilon, min_sample=1):
        self.learners = []
        self.number_of_learners = number
        self.weight_of_learners = np.zeros(self.number_of_learners)
        self.base_tree = DecisionTree(epsilon=epsilon / number, min_sample=min_sample, depth=depth)

    def boost(self, weight_of_sample, X, y):
        tree = clone(self.base_tree)
        decision_tree_classifier = tree
        decision_tree_classifier.fit(X, y, weight_of_sample)
        predictions_y = decision_tree_classifier.predict(X)
        predictions_y = [0.0 if predict <= 0.0 else 1.0 for predict in predictions_y]
        misclassification = [predictions_y != y][0] * np.ones(X.shape[0])
        tau_t = np.dot(weight_of_sample, misclassification) / np.sum(weight_of_sample)
        n_t = 0.5 * np.log((1 - tau_t) / max(tau_t, 1e-16))
        new_weight = weight_of_sample * np.exp(n_t * misclassification)
        return tree, new_weight, n_t

    def fit(self, X, y):
        weight_of_sample = np.ones(X.shape[0]) / X.shape[0]
        for index in range(self.number_of_learners):
            learner, weight_of_sample, n_t = self.boost(weight_of_sample, X, y)
            self.learners.append(learner)
            self.weight_of_learners[index] = n_t

    def predict(self, X):
        predictions = []
        for learner in self.learners:
            prediction = learner.predict(X)
            prediction[prediction == 0] = -1
            predictions.append(prediction)
        predictions = np.array(predictions)
        final_predictions = np.sign(np.dot(self.weight_of_learners, predictions))
        final_predictions[final_predictions == -1] = 0
        return final_predictions