"""
Python implementation of the K-Nearest Neighbor (KNN) algorithm for classification.
The KNN classifier predicts the class of a given test observation by identifying 
training observations that are nearest to it in features space, as defined by some
valid distance function. The relative scale of each feature matters, so data should
be appropriately centered and scaled before feeding through this algorithm. 
"""
import numpy as np

class KNNClassifier:
    def __init__(self, k = 1):
        """Constructor for KNNClassifier object.

        Args:
            k (int): Calculate distances to each of 'k' nearest neighbors in feature space.

        Raises:
            ValueError: If the provided 'k' is not positive.
        """

        if k <= 0:
            raise ValueError("Tuning parameter 'k' must be a positive integer.")
        
        self.k = k

    def fit(self, X, y):
        """Fitter method for KNNClassifier object.

        Input data 'X' contains all predictive features for each of the training
        observations. The response data 'y' contains encoded representation of the
        member class of each training observation. These objects are stored respectively
        as 'X_train' and 'y_train' to the classifier object. Predictive features are 
        scaled using robust scaling that uses the 5th and 95th percentiles rather than 
        min and max, respectively. 

        Args:
            X (numeric array): (n by p) array of input data, consisting of 'p' features and 'n' 
                observations.
            y (numeric array): (n by q) array of encoded responses, for 'q' output classes encoded to
                0 (not a member of class) or 1 (member of class), and 'n' observations.
        Raises:
            ValueError: If tuning parameter 'k' is larger than number of observations.
            TypeError: If 'y' has only one dimension it is certainly not one-hot encoded appropriately.
        """
        if self.k > X.shape[0]:
            raise ValueError("Tuning parameter 'k' must be not be greater than number of observations.")
        
        self.X_train = X
        self.y_train = y
            
    def _f_dist(self, x):
        """Base Euclidean distance function

        The distance between each observation in X_train and the observation 'x' is calculated.
       
        Args:
            x (numeric array): New observation existing in training data feature space.
        """
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        return distances

    def predict(self, X):
        """Make predictions for new data

        For each new observation in 'X', predict class membership based on the most common
        membership of its 'k' nearest neighbors, where nearness is measured by a distance
        function f. If this function is not supplied, Euclidean Distance is used.

        Args:
            X (numeric array): (m by p) Predictive features for each of 'm' new observations.
            f (callable): Valid distance function, or 'None'.
        """

        # helper function to predict for a single observation
        def _predict_single(x):

            # calculate the distances for each training observation
            distances = self._f_dist(x)
            
            # sort the training observations by distance
            knn_indices = np.argsort(distances)[:self.k]

            # average the class membership of the 'k' training indices chosen
            knn_labels = self.y_train[knn_indices]
            neighbor_membership = np.bincount(knn_labels)
            predicted_class = np.where(neighbor_membership == max(neighbor_membership))[0] # outputs a tuple, we want first element
            # pull randomly if there are ties
            if len(predicted_class) > 1:
                predicted_class = np.random.choice(predicted_class, 1)
            
            return predicted_class

        # Make a prediction for all new observations
        y_pred = [_predict_single(x) for x in X]
        return np.array(y_pred)