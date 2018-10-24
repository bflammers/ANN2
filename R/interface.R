#' @title Train a Neural Network
#'
#' @description
#' Train a Multilayer Neural Network using Stohastic Gradient
#' Descent with optional batch learning. Functions \code{autoencoder}
#' and \code{replicator} are special cases of this general function.
#'
#' @details
#' A genereric function for training Neural Networks for classification and
#' regression problems. Various types of activation and cost functions are
#' supported, as well as  L1 and L2 regularization. Additional options are
#' early stopping, momentum and the specification of a learning rate schedule.
#' See function \code{example_NN} for some visualized examples on toy data.
#'
#' @references LeCun, Yann A., et al. "Efficient backprop." Neural networks:
#' Tricks of the trade. Springer Berlin Heidelberg, 2012. 9-48.
#'
#' @param X matrix with explanatory variables
#' @param y matrix with dependent variables
#' @param hidden.layers vector specifying the number of nodes in each layer. Set
#' to \code{NA} for a Network without any hidden layers
#' @param lossFunction which loss function should be used. Options are "log",
#' "quadratic", "absolute", "huber" and "pseudo-huber"
#' @param dHuber used only in case of loss functions "huber" and "pseudo-huber".
#' This parameter controls the cut-off point between quadratic and absolute loss.
#' @param regression logical indicating regression or classification
#' @param standardize logical indicating if X and y should be standardized before
#' training the network. Recommended to leave at \code{TRUE} for faster
#' convergence.
#' @param learnRate the size of the steps made in gradient descent. If set too large,
#' optimization can become unstable. Is set too small, convergence will be slow.
#' @param maxEpochs the maximum number of epochs (one iteration through training
#' data).
#' @param batchSize the number of observations to use in each batch. Batch learning
#' is computationally faster than stochastic gradient descent. However, large
#' batches might not result in optimal learning, see Le Cun for details.
#' @param momentum numeric value specifying how much momentum should be
#' used. Set to zero for no momentum, otherwise a value between zero and one.
#' @param L1 L1 regularization. Non-negative number. Set to zero for no regularization.
#' @param L2 L2 regularization. Non-negative number. Set to zero for no regularization.
#' @param validLoss logical indicating if loss should be monitored during training.
#' If \code{TRUE}, a validation set of proportion \code{validProp} is randomly
#' drawn from full training set. Use function \code{plot} to assess convergence.
#' @param validProp proportion of training data to use for validation
#' @param verbose logical indicating if additional information (such as lifesign)
#' @return An \code{ANN} object. Use function \code{plot(<object>)} to assess
#' loss on training and optionally validation data during training process. Use
#' function \code{predict(<object>, <newdata>)} for prediction.
#' @examples
#' # Example on iris dataset:
#' randDraw <- sample(1:nrow(iris), size = 100)
#' train    <- iris[randDraw,]
#' test     <- iris[setdiff(1:nrow(iris), randDraw),]
#'
#' plot(iris[,1:4], pch = as.numeric(iris$Species))
#'
#' NN <- neuralnetwork(train[,-5], train$Species, hiddenLayers = c(5, 5),
#'                     momentum = 0.8, learnRate = 0.001, verbose = FALSE)
#' plot(NN)
#' pred <- predict(NN, newdata = test[,-5])
#' plot(test[,-5], pch = as.numeric(test$Species),
#'      col = as.numeric(test$Species == pred$predictions)+2)
#'
#' #For other examples see function example_NN()
#'
#' @export
neuralnetwork <- function(X, Y, hidden.layers, regression = FALSE, 
                          loss.type = "log", huber.delta = 1, 
                          activ.functions = "tanh", step.H = 5, step.k = 100,
                          optim.type = "sgd", n.epochs = 1000, 
                          learn.rates = 1e-04, L1 = 0, L2 = 0, sgd.momentum = 0.9,
                          rmsprop.decay = 0.9, adam.beta1 = 0.9, adam.beta2 = 0.999,
                          batch.size = 32, standardize = TRUE, drop.last = TRUE,
                          val.prop = 0.1, verbose = TRUE) {
  
  # Store function call
  NN_call <- match.call()
  
  # Perform checks on data, set meta data
  data <- setData(X, Y, regression)
  meta <- setMeta(data, hidden.layers, regression)
  
  # Set and check parameters
  net_param   <- setNetworkParams(hidden.layers, standardize, verbose, meta)
  activ_param <- setActivParams(activ.functions, step.H, step.k, meta)
  optim_param <- setOptimParams(optim.type, learn.rates, L1, L2, sgd.momentum, 
                                rmsprop.decay, adam.beta1, adam.beta2, meta)
  loss_param  <- setLossParams(loss.type, huber.delta, meta)
  
  # Initialize new ANN object
  Rcpp_ANN <- new(ANN, data, net_param, optim_param, loss_param, activ_param)
  
  # Set and check training parameters
  train_param <- setTrainParams(n.epochs, batch.size, val.prop, drop.last, data)
  
  # Call train method
  Rcpp_ANN$train(data, train_param)
  
  # Create ANN object
  ANN <- list(Rcpp_ANN = Rcpp_ANN)
  class(ANN) <- 'ANN'
  attr(ANN, 'autoencoder') <- FALSE

  return(ANN)
}

#' @title Train an Autoencoding Neural Network
#'
#' @description
#' Trains an Autoencoder by setting explanatory variables X as dependent variables
#' in training. The number of nodes in the middle layer should be smaller than
#' the number of variables in X. During training, the networks will learn a
#' generalised representation of the data (generalised since the middle layer
#' functions as a bottleneck, resulting in reproduction of only the most
#' important features of the data).
#'
#' @details
#' A function for training Autoencoders. To be used in conjunction with function
#' \code{reproduce(<object>, <newdata>)}.
#'
#' @param X matrix with explanatory variables
#' @param hiddenLayers vector specifying the number of nodes in each layer. Set
#' to \code{NA} for a Network without any hidden layers
#' @param lossFunction which loss function should be used. Options are "log",
#' "quadratic", "absolute", "huber" and "pseudo-huber"
#' @param dHuber used only in case of loss functions "huber" and "pseudo-huber".
#' This parameter controls the cut-off point between quadratic and absolute loss.
#' @param H numeric integer specifying number of steps of the step function
#' @param k numeric indicating the smoothness of the step function.
#' Smaller values result in smoother steps. Recommended to keep below 50 for
#' stability. If set to high, the derivative of the stepfunction will also be large
#' @param standardize logical indicating if X and y should be standardized before
#' training the network. Recommended to leave at \code{TRUE} for faster
#' convergence.
#' @param learnRate the size of the steps made in gradient descent. If set too large,
#' optimization can become unstable. Is set too small, convergence will be slow.
#' @param maxEpochs the maximum number of epochs (one iteration through training
#' data).
#' @param batchSize the number of observations to use in each batch. Batch learning
#' is computationally faster than stochastic gradient descent. However, large
#' batches might not result in optimal learning, see Le Cun for details.
#' @param momentum numeric value specifying how much momentum should be
#' used. Set to zero for no momentum, otherwise a value between zero and one.
#' @param L1 L1 regularization. Non-negative number. Set to zero for no regularization.
#' @param L2 L2 regularization. Non-negative number. Set to zero for no regularization.
#' @param validLoss logical indicating if loss should be monitored during training.
#' If \code{TRUE}, a validation set of proportion \code{validProp} is randomly
#' drawn from full training set. Use function \code{plot} to assess convergence.
#' @param validProp proportion of training data to use for validation
#' @param verbose logical indicating if additional information (such as lifesign)
#' should be printed to console during training.
#' @param robErrorCov logical indicating if robust covariance should be estimated in 
#' order to assess Mahalanobis distances of reconstruction errors
#' @return An \code{ANN} object. Use function \code{plot(<object>)} to assess
#' loss on training and optionally validation data during training process. Use
#' function \code{predict(<object>, <newdata>)} for prediction.
#' @examples
#' # Autoencoder
#' aeNN <- autoencoder(faithful, hiddenLayers = c(4,1,4), batchSize = 5,
#'                     learnRate = 1e-5, momentum = 0.5, L1 = 1e-3, L2 = 1e-3,
#'                     robErrorCov = TRUE)
#' plot(aeNN)
#'
#' rX <- reconstruct(aeNN, faithful)
#' plot(rX, alpha = 0.05)
#' plot(faithful, col = (rX$mah_p < 0.05)+1, pch = 16)
#' @export
autoencoder <- function(X, hidden.layers, loss.type = "squared", 
                        delta.huber = 1, activ.functions = "tanh", H = 5, k = 100,
                        optim.type = "sgd", n.epochs = 1000, 
                        learn.rates = 1e-04, momentum = 0.2, L1 = 0, L2 = 0, 
                        batch.size = 32, standardize = TRUE, drop.last = TRUE,
                        val.prop = 0.1, verbose = TRUE) {
  
  # Store function call
  NN_call <- match.call()
  
  # Perform checks on data, set meta data
  data <- setData(X, X, regression = TRUE)
  meta <- setMeta(data, hidden.layers, regression = TRUE)
  
  # Set and check parameters
  net_param   <- setNetworkParams(hidden.layers, standardize, verbose, meta)
  activ_param <- setActivParams(activ.functions, H, k, meta)
  optim_param <- setOptimParams(optim.type, learn.rates, momentum, L1, L2, meta)
  loss_param  <- setLossParams(loss.type, delta.huber, meta)
  
  # Initialize new ANN object
  Rcpp_ANN <- new(ANN, data, net_param, optim_param, loss_param, activ_param)
  
  # Set and check training parameters
  train_param <- setTrainParams(n.epochs, batch.size, val.prop, drop.last, data)
  
  # Call train method
  Rcpp_ANN$train(data, train_param)
  
  # Create ANN object
  ANN <- list(Rcpp_ANN = Rcpp_ANN)
  class(ANN) <- 'ANN'
  attr(ANN, 'autoencoder') <- TRUE
  
  return(ANN)
}

#' @title Continue training of a Neural Network
#'
#' @description
#' Train a Multilayer Neural Network using Stohastic Gradient
#' Descent with optional batch learning. Functions \code{autoencoder}
#' and \code{replicator} are special cases of this general function.
#'
#' @details
#' A genereric function for training Neural Networks for classification and
#' regression problems. Various types of activation and cost functions are
#' supported, as well as  L1 and L2 regularization. Additional options are
#' early stopping, momentum and the specification of a learning rate schedule.
#' See function \code{example_NN} for some visualized examples on toy data.
#'
#' @references LeCun, Yann A., et al. "Efficient backprop." Neural networks:
#' Tricks of the trade. Springer Berlin Heidelberg, 2012. 9-48.
#'
#' @param object object of class \code{ANN}
#' @param X matrix with explanatory variables
#' @param y matrix with dependent variables
#' @param learnRate the size of the steps made in gradient descent. If set too large,
#' optimization can become unstable. Is set too small, convergence will be slow.
#' @param maxEpochs the maximum number of epochs (one iteration through training
#' data).
#' @param batchSize the number of observations to use in each batch. Batch learning
#' is computationally faster than stochastic gradient descent. However, large
#' batches might not result in optimal learning, see Le Cun for details.
#' @param momentum numeric value specifying how much momentum should be
#' used. Set to zero for no momentum, otherwise a value between zero and one.
#' @param L1 L1 regularization. Non-negative number. Set to zero for no regularization.
#' @param L2 L2 regularization. Non-negative number. Set to zero for no regularization.
#' @param validLoss logical indicating if loss should be monitored during training.
#' If \code{TRUE}, a validation set of proportion \code{validProp} is randomly
#' drawn from full training set. Use function \code{plot} to assess convergence.
#' @param validProp proportion of training data to use for validation
#' @param verbose logical indicating if additional information (such as lifesign)
#' should be printed to console during training.
#' @return An \code{ANN} object. Use function \code{plot(<object>)} to assess
#' loss on training and optionally validation data during training process. Use
#' function \code{predict(<object>, <newdata>)} for prediction.
#' @examples
#' # Example on iris dataset:
#' randDraw <- sample(1:nrow(iris), size = 100)
#' train    <- iris[randDraw,]
#' test     <- iris[setdiff(1:nrow(iris), randDraw),]
#'
#' plot(iris[,1:4], pch = as.numeric(iris$Species))
#'
#' NN <- neuralnetwork(train[,-5], train$Species, hiddenLayers = c(5, 5),
#'                     momentum = 0.8, learnRate = 0.001, verbose = FALSE)
#' plot(NN)
#' pred <- predict(NN, newdata = test[,-5])
#' plot(test[,-5], pch = as.numeric(test$Species),
#'      col = as.numeric(test$Species == pred$predictions)+2)
#'
#' #For other examples see function example_NN()
#'
#' @export
train <- function(object, X, Y, n.epochs = 500, batch.size = 32, 
                  drop.last = TRUE, val.prop = 0.1, verbose = TRUE) {
  
  # Extract meta from object
  meta <- object$Rcpp_ANN$getMeta()
  
  # Perform checks on data, set meta data
  data <- setData(X, Y, meta$regression, meta$y_names)
  
  # Set and check training parameters
  train_param <- setTrainParams(n.epochs, batch.size, val.prop, drop.last, data)
  
  # Call train method
  object$Rcpp_ANN$train(data, train_param)

}

#' @title Reconstruct data using trained Autoencoder or Replicator object
#'
#' @description
#' \code{reconstruct} takes new data as input and reconstructs the observations using
#' a trained replicator or autoencoder object.
#'
#' @details
#' A genereric function for training neural nets
#'
#' @param object Object of class \code{ANN} created with autoencoder()
#' @param X data (matrix) to reconstruct
#' @param mahalanobis logical indicating if Mahalanobis distance should be calculated
#' @return Reconstructed observations and optional Mahalanobis distances
#' @export
reconstruct <- function(object, X, mahalanobis = TRUE) {

  # Extract meta
  meta <- object$Rcpp_ANN$getMeta()
  
  # Convert X to matrix
  X <- as.matrix(X)
  
  # Reconstruct only relevant for NNs of type autoencoder
  if ( !attr(object, 'autoencoder') ) {
    stop("Object is not of type autoencoder")
  }
  
  # (ERROR) missing values in X
  if ( any(is.na(X)) ) {
    stop('X contain missing values', call. = FALSE)
  }
  
  # (ERROR) matrix X all numeric columns
  if ( !all(apply(X, 2, is.numeric)) ) {
    stop('X should be numeric', call. = FALSE)
  }
  
  # (ERROR) incorrect number of columns of input data
  if ( ncol(X) != meta$n_in ) {
    stop('Input data incorrect number of columns', call. = FALSE)
  }
  
  # Make reconstruction, calculate errors
  fit <- object$Rcpp_ANN$predict(X)
  colnames(fit) <- meta$names
  err <- rowSums( (fit - X)^2 ) / meta$n_out
  
  # Construct function output
  return( list(reconstructed = fit, errors = err) )
  
}


#' @title Make predictions for new data
#' @description \code{predict} Predict class or value for new data
#' @details A genereric function for training neural nets
#' @param object Object of class \code{ANN}
#' @param newdata Data to make predictions on
#' @param ... further arguments (not in use)
#' @return A list with predicted classes for classification and fitted probabilities
#' @method predict ANN
#' @export
predict.ANN <- function(object, newdata, ...) {
  
  # Extract meta
  meta <- object$Rcpp_ANN$getMeta()
  
  # Convert X to matrix
  X <- as.matrix(newdata)
  
  # (ERROR) missing values in X
  if ( any(is.na(X)) ) {
    stop('newdata contain missing values', call. = FALSE)
  }
  
  # (ERROR) matrix X all numeric columns
  if ( !all(apply(X, 2, is.numeric)) ) {
    stop('newdata should be numeric', call. = FALSE)
  }
  
  # Predict and set column names
  fit <- object$Rcpp_ANN$predict(X)
  colnames(fit) <- paste0("class_", meta$y_names)
  
  # For regression return fitted values
  if ( meta$regression ) {
    return( list(predictions = fit) )
  }
  
  # For classification return predicted classes and probabilities (fit)
  predictions <- meta$y_names[apply(fit, 1, which.max)]
  return( list(predictions = predictions, probabilities = fit) )
}

#' @title Plot mahalanobis distances of reconstructed errors
#' @description \code{plot} Generate plots of the mahalanobis distances for each reconstructed observation
#' @details A genereric function for training neural nets
#' @param x Object of class \code{rX}
#' @param alpha significance level for determining cut-off point of squared
#' Mahalanobis distanced using the chi-square distribution
#' @param ... further arguments to be passed to plot
#' @return Plots
#' @method plot rANN
#' @export
plot.rANN <- function(x, alpha = 0.05, ...) {
  nObs    <- length(x$mah_sq)
  maxMah  <- max(x$mah_sq)
  stats::qqplot(stats::qchisq(p = stats::ppoints(500), df = x$dfChiSq), x$mah_sq,
                main = "Chi-Square QQ-plot Mahalanobis Squared",
                xlab = "Theoretical Quantiles", ylab = "Observed Quantiles", ...)
  graphics::lines(c(0, maxMah), c(0, maxMah), col = "blue")
  invisible(readline(prompt = "Press [enter] for next plot..."))
  ChiSqBound <- stats::qchisq(p = 1 - alpha, df = x$dfChiSq)
  graphics::plot(seq.int(1, nObs), x$mah_sq, col = (x$mah_p <= alpha) + 1,
      main = "Mahalanobis Distance Squared", xlab = "Index", ylab = "Distance", ...)
  graphics::abline(h = ChiSqBound, col = "darkgrey")
}

#' @title Print ANN
#' @description Print info on trained Neural Network
#' @param x Object of class \code{ANN}
#' @param ... Further arguments
#' @method print ANN
#' @export
print.ANN <- function(x, ...){
  x$Rcpp_ANN$print( TRUE )
}

#' @title Encoding step 
#' @description Compress data according to trained replicator or autoencoder.
#' Outputs are the activations of the nodes in the middle layer for each 
#' observation in \code{newdata}
#' @param object Object of class \code{ANN}
#' @param newdata Data to compress
#' @param compression.layer Integer specifying which hidden layer is the 
#' compression layer. If NULL this parameter is inferred from the structure 
#' of the network (hidden layer with smallest number of nodes)
#' @export
encode <- function(object, newdata, compression.layer = NULL) {
  
  if ( !attr(object, 'autoencoder') ) {
    warning("Object is not an autoencoder")
  }
  
  # Extract meta, hidden_layers
  meta <- object$Rcpp_ANN$getMeta()
  hidden_layers <- meta$num_nodes[2:(1+meta$n_hidden)]
  
  # Convert X to matrix
  X <- as.matrix(newdata)
  
  # (ERROR) missing values in X
  if ( any(is.na(X)) ) {
    stop('newdata contain missing values', call. = FALSE)
  }
  
  # (ERROR) matrix X all numeric columns
  if ( !all(apply(X, 2, is.numeric)) ) {
    stop('newdata should be numeric', call. = FALSE)
  }
  
  # (ERROR) incorrect number of columns of input data
  if ( ncol(X) != meta$n_in ) {
    stop('Input data incorrect number of columns', call. = FALSE)
  }
  
  # Determine compression layer
  if ( is.null(compression.layer) ) {
    
    # Compression layer is hidden layer with minimum number of nodes
    compression.layer <- which.min( hidden_layers )
    
    # (ERROR) Ambiguous compression layer
    if ( sum( hidden_layers[compression.layer] == hidden_layers) > 1 ) {
      stop('Ambiguous compression layer, specify compression.layer', call. = FALSE)
    } 
  }
  
  # Predict and set column names
  compressed <- object$Rcpp_ANN$partialForward(X, 0, compression.layer)
  colnames(compressed) <- paste0("node_", 1:NCOL(compressed))
  
  return( compressed )
}


#' @title Decoding step 
#' @description Decompress low-dimensional representation resulting from the nodes
#' of the middle layer. Output are the reconstructed inputs to function \code{encode}
#' @param object Object of class \code{ANN}
#' @param compressed Compressed data
#' @param compression.layer Integer specifying which hidden layer is the 
#' compression layer. If NULL this parameter is inferred from the structure 
#' of the network (hidden layer with smallest number of nodes)
#' @export
decode <- function(object, compressed, compression.layer = NULL) {
  
  if ( !attr(object, 'autoencoder') ) {
    warning("Object is not an autoencoder")
  }
  
  # Extract meta, hidden_layers vector
  meta <- object$Rcpp_ANN$getMeta()
  hidden_layers <- meta$hidden_layers[2:(1+meta$n_hidden)]
  
  # Convert X to matrix
  X <- as.matrix(compressed)
  
  # (ERROR) missing values in X
  if ( any(is.na(X)) ) {
    stop('compressed contain missing values', call. = FALSE)
  }
  
  # (ERROR) matrix X all numeric columns
  if ( !all(apply(X, 2, is.numeric)) ) {
    stop('compressed should be numeric', call. = FALSE)
  }
  
  # Determine compression layer
  if ( is.null(compression.layer) ) {
    
    # Compression layer is hidden layer with minimum number of nodes
    compression.layer <- which.min( hidden_layers )
    
    # (ERROR) Ambiguous compression layer
    if ( sum( hidden_layers[compression.layer] == hidden_layers) > 1 ) {
      stop('Ambiguous compression layer, specify compression.layer', call. = FALSE)
    } 
  }
  
  # Predict and set column names
  fit <- object$Rcpp_ANN$partialForward(X, compression.layer, meta$n_hidden + 1)
  colnames(fit) <- meta$y_names
  
  return( fit )
}

