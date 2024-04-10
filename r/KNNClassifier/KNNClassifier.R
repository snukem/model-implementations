# R implementation of the K-Nearest Neighbor (KNN) algorithm for classification.
# The KNN classifier predicts the class of a given test observation by identifying 
# training observations that are nearest to it in features space, as defined by some
# valid distance function. The relative scale of each feature matters, so data should
# be appropriately centered and scaled before feeding through this algorithm. 
KNNClassifier <- function(k = 1) {
  
  if (!is.numeric(k)) stop("Function argument 'k' must be a positive integer.")
  
  # return a function that has takes training data as input and returns as 
  # predictor function that allows for new data to be classified 
  function(X, y) {
    
    # response classes y should be encoded as factor labels in a vector.
    if (!is.null(dim(y))) stop("Response classes should be encoded as labels in a vector.") 
    if (!is.factor(y)) stop("Reponse classes should be encoded as factors.")
    
    # return predictor function that takes in test observations
    function(X_test, y_test, ...) {
      
      # calculate (Euclidean, by default) distances from test observations to training observations
      all_X <- rbind(X, X_test)
      n <- nrow(X)
      m <- nrow(X_test)
      D <- as.matrix(dist(all_X, ...))
      
      y_hat <- factor(character(m), levels = levels(y))
      # the last m rows of the distance matrix correspond to the distances 
      for (i in (n+1):(n+m)) {
        distances <- D[1:n, i]
        indices <- order(distances)[1:k]
        knn_labels <- y[indices]
        knn_membership <- table(knn_labels)
        predicted_class <- names(knn_membership[knn_membership == max(knn_membership)])
        
        # if there is a tie for most common class among neighbors, choose randomly
        if (length(predicted_class) > 1) {
          predicted_class <- sample(predicted_class, 1)
        }
        y_hat[i-n] <- predicted_class
      }
      
      list(y_hat = y_hat, 
    }
  }
}