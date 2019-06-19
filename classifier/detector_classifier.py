""""Classifier that replaces the current classifier with a new one when a change is detected in accuracy.
"""
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


class DetectorClassifier(BaseEstimator):
    def __init__(self, clf, detection_method, classes):
        if not hasattr(clf, "partial_fit"):
            raise TypeError("Choose incremental classifier")
        self.clf = clf
        self.detection_method = detection_method
        self.classes = classes
        self.change_detected = 0
        self.elems = 0
        self.detected_elements = []

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def partial_fit(self, X, y):
        pre_y = self.clf.predict(X)
        if self.detection_method.set_input(accuracy_score(pre_y, y)):
            self.change_detected += 1
            self.clf = clone(self.clf)
            self.clf.partial_fit(X, y, classes=self.classes)
            # register the concept drift element
            self.detected_elements .append(self.elems)
        else:
            self.clf.partial_fit(X, y)
        self.elems += 1

    def predict(self, X):
        return self.clf.predict(X)
