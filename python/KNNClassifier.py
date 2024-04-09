import numpy as np

'''
Python implementation of the K-Nearest Neighbor (KNN) algorithm for classification.
The KNN classifier predicts the class of a given test observation by identifying 
training observations that are nearest to it in features space, as defined by some
valid distance function. The relative scale of each feature matters. 
'''

class KNNClassifier:
    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("Tuning parameter 'k' must be a positive integer.")
        self.k = k
        pass

    def fit(self, X_train, y_train, f: function = None):
        # if self.k > n(X_train) then raise ValueError

        # fit the model

        # if f then use it as user-defined distance function 
        # else use f_dist

        pass

    def f_dist(x1, x2):
        # Euclidean distance between x1 and x2
        pass
