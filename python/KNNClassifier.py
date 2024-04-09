import numpy as np

'''
Python implementation of the K-Nearest Neighbor (KNN) algorithm for classification.
The KNN classifier predicts the class of a given test observation by identifying 
training observations that are nearest to it in features space, as defined by some
valid distance function. The relative scale of each feature matters. 
'''

class KNNClassifier:
    def __init__(self, k: int = 1):
        """Constructor for KNNClassifier object.

        Args:
            k: Calculate distances to each of 'k' nearest neighbors in feature space.

        Raises:
            ValueError: If the provided 'k' is not positive.
        """

        if k <= 0:
            raise ValueError("Tuning parameter 'k' must be a positive integer.")
        
        self.k = k

    def fit(self, X: np.array, y: np.array):
        """Fitter method for KNNClassifier object.

        Input data 'X' contains all predictive features for each of the training
        observations. The response data 'y' contains encoded representation of the
        member class of each training observation. These objects are stored respectively
        as 'X_train' and 'y_train' to the classifier object.

        Args:
            X: (n by p) array of input data, consisting of 'p' features and 'n' 
                observations.
            y: (n by q) array of encoded responses, for 'q' output classes encoded to
                0 (not a member of class) or 1 (member of class), and 'n' observations.

        Raises:
            ValueError: If tuning parameter 'k' is larger than number of observations.
        """
        self.X_train = X
        self.y_train = y
        # if self.k > n(X_train) then raise ValueError
        pass

    def f_dist(x1, x2):
        # Euclidean distance between x1 and x2
        pass

    def predict(self, X_test, y_test, f: function = None):
        # fit the model

        # if f then use it as user-defined distance function 
        # else use f_dist

        # Make a prediction for new observations
        pass