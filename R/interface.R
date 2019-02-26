#' @title Train a Neural Network
#'
#' @description
#' Construct and train a Multilayer Neural Network for regression or 
#' classification
#'
#' @details
#' A genereric function for training Neural Networks for classification and
#' regression problems. Various types of activation and loss functions are
#' supported, as well as  L1 and L2 regularization. Possible optimizer include
#' SGD (with or without momentum), RMSprop and Adam. 
#'
#' @references LeCun, Yann A., et al. "Efficient backprop." Neural networks:
#' Tricks of the trade. Springer Berlin Heidelberg, 2012. 9-48.
#'
#' @param X matrix with explanatory variables
#' @param y matrix with dependent variables. For classification this should be 
#' a one-columns matrix containing the classes - classes will be one-hot encoded.
#' @param hidden.layers vector specifying the number of nodes in each layer. The
#' number of hidden layers in the network is implicitly defined by the length of
#' this vector. Set \code{hidden.layers} to \code{NA} for a network with no hidden 
#' layers
#' @param regression logical indicating regression or classification
#' @param standardize logical indicating if X and Y should be standardized before
#' training the network. Recommended to leave at \code{TRUE} for faster
#' convergence.
#' @param loss.type which loss function should be used. Options are "log",
#' "quadratic", "absolute", "huber" and "pseudo-huber"
#' @param huber.delta used only in case of loss functions "huber" and "pseudo-huber".
#' This parameter controls the cut-off point between quadratic and absolute loss.
#' @param activ.functions character vector of activation functions to be used in 
#' each hidden layer. Possible options are 'tanh', 'sigmoid', 'relu', 'linear', 
#' 'ramp' and 'step'. Should be either the size of the number of hidden layers
#' or equal to one. If a single avtivation type is specified, this type will be 
#' broadcasted across the hidden layers. 
#' @param step.H number of steps of the step activation function. Only applicable 
#' if activ.functions includes 'step'
#' @param step.k parameter controlling the smoothness of the step activation 
#' function. Larger values lead to a less smooth step function. Only applicable 
#' if activ.functions includes 'step'.
#' @param optim.type type of optimizer to use for updating the parameters. Options 
#' are 'sgd', 'rmsprop' and 'adam'. SGD is implemented with momentum.
#' @param learn.rates the size of the steps to make in gradient descent. If set 
#' too large, the optimization might not converge to optimal values. If set too 
#' small, convergence will be slow. Should be either the size of the number of 
#' hidden layers plus one or equal to one. If a single learn rate is specified, 
#' this learn rate will be broadcasted across the layers. 
#' @param L1 L1 regularization. Non-negative number. Set to zero for no regularization.
#' @param L2 L2 regularization. Non-negative number. Set to zero for no regularization.
#' @param sgd.momentum numeric value specifying how much momentum should be
#' used. Set to zero for no momentum, otherwise a value between zero and one.
#' @param rmsprop.decay level of decay in the rms term. Controls the strength
#' of the exponential decay of the squared gradients in the term that scales the
#' gradient before the parameter update. Common values are 0.9, 0.99 and 0.999
#' @param adam.beta1 level of decay in the first moment estimate (the mean). 
#' The recommended value is 0.9
#' @param adam.beta2 level of decay in the second moment estimate (the uncentered
#' variance). The recommended value is 0.999
#' @param n.epochs the number of epochs to train. This parameter largely determines
#' the training time (one epoch is a single iteration through the training data).
#' @param batch.size the number of observations to use in each batch. Batch learning
#' is computationally faster than stochastic gradient descent. However, large
#' batches might not result in optimal learning, see Efficient Backprop by Le Cun 
#' for details.
#' @param drop.last logical. Only applicable if the size of the training set is not 
#' perfectly devisible by the batch size. Determines if the last chosen observations
#' should be discarded (in the current epoch) or should constitute a smaller batch. 
#' Note that a smaller batch leads to a noisier approximation of the gradient.
#' @param val.prop proportion of training data to use for tracking the loss on a 
#' validation set during training. Useful for assessing the training process and
#' identifying possible overfitting. Set to zero for only tracking the loss on the 
#' training data.
#' @param verbose logical indicating if additional information should be printed
#' @param random.seed optional seed for the random number generator
#' @return An \code{ANN} object. Use function \code{plot(<object>)} to assess
#' loss on training and optionally validation data during training process. Use
#' function \code{predict(<object>, <newdata>)} for prediction.
#' @examples
#' # Example on iris dataset
#' # Prepare test and train sets
#' random_draw <- sample(1:nrow(iris), size = 100)
#' X_train     <- iris[random_draw, 1:4]
#' y_train     <- iris[random_draw, 5]
#' X_test      <- iris[setdiff(1:nrow(iris), random_draw), 1:4]
#' y_test      <- iris[setdiff(1:nrow(iris), random_draw), 5]
#' 
#' # Train neural network on classification task
#' NN <- neuralnetwork(X = X_train, y = y_train, hidden.layers = c(5, 5),
#'                     optim.type = 'adam', learn.rates = 0.01, val.prop = 0)
#' 
#' # Plot the loss during training
#' plot(NN)
#' 
#' # Make predictions
#' y_pred <- predict(NN, newdata = X_test)
#' 
#' # Plot predictions
#' correct <- (y_test == y_pred$predictions)
#' plot(X_test, pch = as.numeric(y_test), col = correct + 2)
#' 
#' @export
neuralnetwork <- function(X, y, hidden.layers, regression = FALSE, 
                          standardize = TRUE, loss.type = "log", huber.delta = 1, 
                          activ.functions = "tanh", step.H = 5, step.k = 100,
                          optim.type = "sgd", learn.rates = 1e-04, L1 = 0, L2 = 0, 
                          sgd.momentum = 0.9, rmsprop.decay = 0.9, adam.beta1 = 0.9, 
                          adam.beta2 = 0.999, n.epochs = 100, batch.size = 32, 
                          drop.last = TRUE, val.prop = 0.1, verbose = TRUE, 
                          random.seed = NULL) {
  
  # Perform checks on data, set meta data
  data <- setData(X, y, regression)
  meta <- setMeta(data, hidden.layers, regression, autoencoder = FALSE)
  
  # Set and check parameters
  net_param   <- setNetworkParams(hidden.layers, standardize, verbose, meta)
  activ_param <- setActivParams(activ.functions, step.H, step.k, meta)
  optim_param <- setOptimParams(optim.type, learn.rates, L1, L2, sgd.momentum, 
                                rmsprop.decay, adam.beta1, adam.beta2, meta)
  loss_param  <- setLossParams(loss.type, huber.delta, meta)
  
  # Set and check training parameters - not used during object initialization
  # Sets the random seed so must be called before initializing ANN object 
  train_param <- setTrainParams(n.epochs, batch.size, val.prop, drop.last, 
                                random.seed, data)
  
  # Initialize new ANN object
  Rcpp_ANN <- new(ANN, data, net_param, optim_param, loss_param, activ_param)
  
  # Train the network
  Rcpp_ANN$train(data, train_param)
  
  # Create ANN object
  ANN <- list(Rcpp_ANN = Rcpp_ANN)
  class(ANN) <- 'ANN'

  return(ANN)
}

#' @title Train an Autoencoding Neural Network
#'
#' @description
#' Construct and train an Autoencoder by setting the target variables equal 
#' to the input variables. The number of nodes in the middle layer should be 
#' smaller than the number of input variables in X in order to create a 
#' bottleneck layer. 
#'
#' @details
#' A function for training Autoencoders. During training, the network will learn a
#' generalised representation of the data (generalised since the middle layer
#' acts as a bottleneck, resulting in reproduction of only the most
#' important features of the data). As such, the network models the normal state 
#' of the data and therefore has a denoising property. This property can be 
#' exploited to detect anomalies by comparing input to reconstruction. If the 
#' difference (the reconstruction error) is large, the observation is a possible 
#' anomaly. 
#'
#' @param X matrix with explanatory variables
#' @param hidden.layers vector specifying the number of nodes in each layer. The
#' number of hidden layers in the network is implicitly defined by the length of
#' this vector. Set \code{hidden.layers} to \code{NA} for a network with no hidden 
#' layers
#' @param standardize logical indicating if X and Y should be standardized before
#' training the network. Recommended to leave at \code{TRUE} for faster
#' convergence.
#' @param loss.type which loss function should be used. Options are "log",
#' "quadratic", "absolute", "huber" and "pseudo-huber"
#' @param huber.delta used only in case of loss functions "huber" and "pseudo-huber".
#' This parameter controls the cut-off point between quadratic and absolute loss.
#' @param activ.functions character vector of activation functions to be used in 
#' each hidden layer. Possible options are 'tanh', 'sigmoid', 'relu', 'linear', 
#' 'ramp' and 'step'. Should be either the size of the number of hidden layers
#' or equal to one. If a single avtivation type is specified, this type will be 
#' broadcasted across the hidden layers. 
#' @param step.H number of steps of the step activation function. Only applicable 
#' if activ.functions includes 'step'
#' @param step.k parameter controlling the smoothness of the step activation 
#' function. Larger values lead to a less smooth step function. Only applicable 
#' if activ.functions includes 'step'.
#' @param optim.type type of optimizer to use for updating the parameters. Options 
#' are 'sgd', 'rmsprop' and 'adam'. SGD is implemented with momentum.
#' @param learn.rates the size of the steps to make in gradient descent. If set 
#' too large, the optimization might not converge to optimal values. If set too 
#' small, convergence will be slow. Should be either the size of the number of 
#' hidden layers plus one or equal to one. If a single learn rate is specified, 
#' this learn rate will be broadcasted across the layers. 
#' @param L1 L1 regularization. Non-negative number. Set to zero for no regularization.
#' @param L2 L2 regularization. Non-negative number. Set to zero for no regularization.
#' @param sgd.momentum numeric value specifying how much momentum should be
#' used. Set to zero for no momentum, otherwise a value between zero and one.
#' @param rmsprop.decay level of decay in the rms term. Controls the strength
#' of the exponential decay of the squared gradients in the term that scales the
#' gradient before the parameter update. Common values are 0.9, 0.99 and 0.999
#' @param adam.beta1 level of decay in the first moment estimate (the mean). 
#' The recommended value is 0.9
#' @param adam.beta2 level of decay in the second moment estimate (the uncentered
#' variance). The recommended value is 0.999
#' @param n.epochs the number of epochs to train. This parameter largely determines
#' the training time (one epoch is a single iteration through the training data).
#' @param batch.size the number of observations to use in each batch. Batch learning
#' is computationally faster than stochastic gradient descent. However, large
#' batches might not result in optimal learning, see Efficient Backprop by Le Cun 
#' for details.
#' @param drop.last logical. Only applicable if the size of the training set is not
#' perfectly devisible by the batch size. Determines if the last chosen observations
#' should be discarded (in the current epoch) or should constitute a smaller batch. 
#' Note that a smaller batch leads to a noisier approximation of the gradient.
#' @param val.prop proportion of training data to use for tracking the loss on a 
#' validation set during training. Useful for assessing the training process and
#' identifying possible overfitting. Set to zero for only tracking the loss on the 
#' training data.
#' @param verbose logical indicating if additional information should be printed
#' @param random.seed optional seed for the random number generator
#' @return An \code{ANN} object. Use function \code{plot(<object>)} to assess
#' loss on training and optionally validation data during training process. Use
#' function \code{predict(<object>, <newdata>)} for prediction.
#' @examples
#' # Autoencoder example
#' X <- USArrests
#' AE <- autoencoder(X, c(10,2,10), loss.type = 'pseudo-huber',
#'                   activ.functions = c('tanh','linear','tanh'),
#'                   batch.size = 8, optim.type = 'adam',
#'                   n.epochs = 1000, val.prop = 0)
#' 
#' # Plot loss during training
#' plot(AE)
#' 
#' # Make reconstruction and compression plots
#' reconstruction_plot(AE, X)
#' compression_plot(AE, X)
#' 
#' # Reconstruct data and show states with highest anomaly scores
#' recX <- reconstruct(AE, X)
#' sort(recX$anomaly_scores, decreasing = TRUE)[1:5]
#' 
#' @export
autoencoder <- function(X, hidden.layers, standardize = TRUE, 
                        loss.type = "squared", huber.delta = 1, 
                        activ.functions = "tanh", step.H = 5, step.k = 100,
                        optim.type = "sgd", learn.rates = 1e-04, L1 = 0, L2 = 0, 
                        sgd.momentum = 0.9, rmsprop.decay = 0.9, adam.beta1 = 0.9, 
                        adam.beta2 = 0.999, n.epochs = 100, batch.size = 32, 
                        drop.last = TRUE, val.prop = 0.1, verbose = TRUE, 
                        random.seed = NULL) {
  
  # Perform checks on data, set meta data
  data <- setData(X, X, regression = TRUE)
  meta <- setMeta(data, hidden.layers, regression = TRUE, autoencoder = TRUE)
  
  # Set and check parameters
  net_param   <- setNetworkParams(hidden.layers, standardize, verbose, meta)
  activ_param <- setActivParams(activ.functions, step.H, step.k, meta)
  optim_param <- setOptimParams(optim.type, learn.rates, L1, L2, sgd.momentum, 
                                rmsprop.decay, adam.beta1, adam.beta2, meta)
  loss_param  <- setLossParams(loss.type, huber.delta, meta)
  
  # Set and check training parameters - not used during object initialization
  # Sets the random seed so must be called before initializing ANN object 
  train_param <- setTrainParams(n.epochs, batch.size, val.prop, drop.last, 
                                random.seed, data)
  
  # Initialize new ANN object
  Rcpp_ANN <- new(ANN, data, net_param, optim_param, loss_param, activ_param)
  
  # Train network
  Rcpp_ANN$train(data, train_param)
  
  # Create ANN object
  ANN <- list(Rcpp_ANN = Rcpp_ANN)
  class(ANN) <- 'ANN'
  
  return(ANN)
}

#' @title Continue training of a Neural Network
#'
#' @description
#' Continue training of a neural network object returned by \code{neuralnetwork()} 
#' or \code{autoencoder()}
#'
#' @details
#' A new validation set is randomly chosen. This can result in irregular jumps
#' in the plot given by \code{plot.ANN()}.
#'
#' @references LeCun, Yann A., et al. "Efficient backprop." Neural networks:
#' Tricks of the trade. Springer Berlin Heidelberg, 2012. 9-48.
#'
#' @param object object of class \code{ANN} produced by \code{neuralnetwork()} 
#' or \code{autoencoder()}
#' @param X matrix with explanatory variables
#' @param y matrix with dependent variables. Not required if object is an autoencoder
#' @param n.epochs the number of epochs to train. This parameter largely determines
#' the training time (one epoch is a single iteration through the training data).
#' @param batch.size the number of observations to use in each batch. Batch learning
#' is computationally faster than stochastic gradient descent. However, large
#' batches might not result in optimal learning, see Efficient Backprop by Le Cun 
#' for details.
#' @param drop.last logical. Only applicable if the size of the training set is not 
#' perfectly devisible by the batch size. Determines if the last chosen observations
#' should be discarded (in the current epoch) or should constitute a smaller batch. 
#' Note that a smaller batch leads to a noisier approximation of the gradient.
#' @param val.prop proportion of training data to use for tracking the loss on a 
#' validation set during training. Useful for assessing the training process and
#' identifying possible overfitting. Set to zero for only tracking the loss on the 
#' training data.
#' @param random.seed optional seed for the random number generator
#' @return An \code{ANN} object. Use function \code{plot(<object>)} to assess
#' loss on training and optionally validation data during training process. Use
#' function \code{predict(<object>, <newdata>)} for prediction.
#' @examples 
#' # Train a neural network on the iris dataset
#' X <- iris[,1:4]
#' y <- iris$Species
#' NN <- neuralnetwork(X, y, hidden.layers = 10, sgd.momentum = 0.9, 
#'                     learn.rates = 0.01, val.prop = 0.3, n.epochs = 100)
#' 
#' # Plot training and validation loss during training
#' plot(NN)
#' 
#' # Continue training for 1000 epochs
#' train(NN, X, y, n.epochs = 200, val.prop = 0.3)
#' 
#' # Again plot the loss - note the jump in the validation loss at the 100th epoch
#' # This is due to the random selection of a new validation set
#' plot(NN)
#' @export
train <- function(object, X, y = NULL, n.epochs = 100, batch.size = 32, 
                  drop.last = TRUE, val.prop = 0.1, random.seed = NULL) {
  
  # Extract meta from object
  meta <- object$Rcpp_ANN$getMeta()
  
  # Checks for different behavior autoencoder and normal neural net
  if ( meta$autoencoder ) {
    
    # Autoencoder but also Y specified
    if ( !is.null(y) ) {
      stop('Object of type autoencoder but y is given', call. = FALSE)
    }
    
    # Set Y equal to X
    y = X
    
  } else {
    
    # Not an autoencoder but no Y given
    if ( is.null(y) ) {
      stop('y matrix of dependent variables needed', call. = FALSE)
    }
    
  }
  
  # Perform checks on data, set meta data
  data <- setData(X, y, meta$regression, meta$y_names)
  
  # Set and check training parameters
  train_param <- setTrainParams(n.epochs, batch.size, val.prop, drop.last, 
                                random.seed, data)
  
  # Call train method
  object$Rcpp_ANN$train(data, train_param)

}

#' @title Reconstruct data using trained ANN object of type autoencoder
#'
#' @description
#' \code{reconstruct} takes new data as input and reconstructs the observations using
#' a trained replicator or autoencoder object.
#'
#' @details
#' A genereric function for training neural nets
#'
#' @param object Object of class \code{ANN} created with \code{autoencoder()}
#' @param X data matrix to reconstruct
#' @return Reconstructed observations and anomaly scores (reconstruction errors)
#' @export
reconstruct <- function(object, X) {

  # Extract meta
  meta <- object$Rcpp_ANN$getMeta()
  
  # Convert X to matrix
  X <- as.matrix(X)
  
  # Reconstruct only relevant for NNs of type autoencoder
  if ( !meta$autoencoder ) {
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
  
  # Make reconstruction, calculate the anomaly scores
  fit <- object$Rcpp_ANN$predict(X)
  colnames(fit) <- meta$names
  err <- rowSums( (fit - X)^2 ) / meta$n_out
  
  # Construct function output
  return( list(reconstructed = fit, anomaly_scores = err) )
  
}


#' @title Make predictions for new data
#' @description \code{predict} Predict class or value for new data
#' @details A genereric function for training neural nets
#' @method predict ANN
#' @param object Object of class \code{ANN}
#' @param newdata Data to make predictions on
#' @param \dots further arguments (not in use)
#' @return A list with predicted classes for classification and fitted probabilities
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
  
  # Predict 
  fit <- object$Rcpp_ANN$predict(X)
  
  # Set column names
  if (meta$regression) {
    colnames(fit) <- meta$y_names
  } else {
    colnames(fit) <- paste0("class_", meta$y_names)
  }
  
  # For regression return fitted values
  if ( meta$regression ) {
    return( list(predictions = fit) )
  }
  
  # For classification return predicted classes and probabilities (fit)
  predictions <- meta$y_names[apply(fit, 1, which.max)]
  return( list(predictions = predictions, probabilities = fit) )
}

#' @title Print ANN
#' @description Print info on trained Neural Network
#' @method print ANN
#' @param x Object of class \code{ANN}
#' @param \dots Further arguments
#' @export
print.ANN <- function(x, ...){
  x$Rcpp_ANN$print( TRUE )
}

#' @title Write ANN object to file
#' @description Serialize ANN object to binary file
#' @param object Object of class \code{ANN}
#' @param file character specifying file path
#' @export
write_ANN <- function(object, file){
  object$Rcpp_ANN$write(file)
}

#' @title Read ANN object from file
#' @description Deserialize ANN object from binary file
#' @param file character specifying file path
#' @return Object of class ANN
#' @export
read_ANN <- function(file){
  
  # Initialize new S4 object and call read method 
  Rcpp_ANN <- new(ANN)
  Rcpp_ANN$read(file)
  
  # Create ANN object
  ANN <- list(Rcpp_ANN = Rcpp_ANN)
  class(ANN) <- 'ANN'
  
  return(ANN)
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
#' @param \dots arguments to be passed down
#' @export
encode <- function(object, ...) UseMethod("encode")

#' @rdname encode
#' @method encode ANN
#' @export
encode.ANN <- function(object, newdata, compression.layer = NULL, ...) {
  
  # Extract meta, hidden_layers
  meta <- object$Rcpp_ANN$getMeta()
  hidden_layers <- meta$num_nodes[2:(1+meta$n_hidden)]
  
  # Check if autoencoder
  if ( !meta$autoencoder ) {
    warning("Object is not an autoencoder")
  }
  
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
#' of the middle layer. Output are the reconstructed inputs to function \code{encode()}
#' @param object Object of class \code{ANN}
#' @param compressed Compressed data
#' @param compression.layer Integer specifying which hidden layer is the 
#' compression layer. If NULL this parameter is inferred from the structure 
#' of the network (hidden layer with smallest number of nodes)
#' @param \dots arguments to be passed down
#' @export
decode <- function(object, ...) UseMethod("decode")

#' @rdname decode
#' @method decode ANN
#' @export
decode.ANN <- function(object, compressed, compression.layer = NULL, ...) {
  
  # Extract meta, hidden_layers vector
  meta <- object$Rcpp_ANN$getMeta()
  hidden_layers <- meta$num_nodes[2:(1+meta$n_hidden)]
  
  if ( !meta$autoencoder ) {
    warning("Object is not an autoencoder")
  }
  
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

