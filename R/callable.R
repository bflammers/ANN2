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
#' @param hiddenLayers vector specifying the number of nodes in each layer. Set
#' to \code{NA} for a Network without any hidden layers
#' @param lossFunction which loss function should be used. Options are "log",
#' "quadratic", "absolute", "huber" and "pseudo-huber"
#' @param dHuber used only in case of loss functions "huber" and "pseudo-huber".
#' This parameter controls the cut-off point between quadratic and absolute loss.
#' @param rectifierLayers vector or integer specifying which layers should have
#' rectifier activation in its nodes
#' @param sigmoidLayers vector or integer specifying which layers should have
#' sigmoid activation in its nodes
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
#' should be printed to console during training.
#' @param earlyStop logical indicating if early stopping should be used based on
#' the loss on a validation set. Only possible with \code{validLoss} set to \code{TRUE}
#' @param earlyStopEpochs after how many epochs without sufficient improvement
#' (as specified by \code{earlyStopTol}) should training be stopped.
#' @param earlyStopTol numerical value specifying tolerance for early stopping.
#' Can be either positive or negative. When set negative, training will be stopped
#' if improvements are made but improvements are smaller than tolerance.
#' @param lrSched logical indicating if a schedule for the learning rate should
#' be used. If \code{TRUE}, schedule as specified by \code{lrSchedEpochs} and
#' \code{lrSchedLearnRates} .
#' @param lrSchedLearnRates vector with elements specifying the learn rate to be used
#' after epochs determined by lrSchedEpochs.
#' @param lrSchedEpochs vector with elements specifying the epoch after which the
#' corresponding learn rate from vector \code{lrSchedLearnRates}. Length of vector
#' shoud be the same as length of \code{learnSchedLearnRates}.
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
#'                     momentum = 0.8, learnRate = 0.001)
#' plot(NN)
#' pred <- predict(NN, newdata = test[,-5])
#' plot(test[,-5], pch = as.numeric(test$Species),
#'      col = as.numeric(test$Species == pred$predictions)+2)
#'
#' #For other examples see function example_NN()
#'
#' @export
neuralnetwork <- function(X, y, hiddenLayers, lossFunction = "log", dHuber = 1, rectifierLayers = NA, sigmoidLayers = NA,
                          regression = FALSE, standardize = TRUE, learnRate = 1e-04, maxEpochs = 1000, batchSize = 32,
                          momentum = 0.2, L1 = 1e-07, L2 = 1e-04, validLoss = TRUE, validProp = 0.2, verbose = TRUE,
                          earlyStop = TRUE, earlyStopEpochs = 50, earlyStopTol = -1e-07, lrSched = FALSE,
                          lrSchedLearnRates = 1e-05, lrSchedEpochs = 400) {
  
  NN_call <- match.call()
  X       <- as.matrix(X)
  y       <- as.matrix(y)
  if (regression) {
    if (!all(apply(X, 2, is.numeric))) {
      stop("X should be numeric")
    }
    if (!all(apply(y, 2, is.numeric))) {
      stop("y should be numeric")
    }
    if (!(lossFunction %in% c("quadratic", "huber", "pseudo-huber", "absolute"))) {
      lossFunction <- "quadratic"
      warning("Regression: using \"quadratic\" loss function \n")
    }
  } else {
    if (ncol(y)!=1) {
      stop("Classification: y should be a vector or one-column matrix containing classes.")
    }
    if (lossFunction != "log") {
      warning("Log loss recommended for classification.")
    }
    if (length(unique(y)) <= 1) {
      stop("Dependent variable y contains less than two classes.")
    }
  }
  
  if (!validLoss) {
    validProp <- 0
  }
  nColX   <- ncol(X)
  nColY   <- ifelse(regression, ncol(y), length(unique(y)))
  nTot    <- nrow(X)
  nTrain  <- ceiling(nTot * (1 - validProp))
  nVal    <- nTot - nTrain
  
  checkParameters(lossFunction = lossFunction, dHuber = dHuber, hiddenLayers = hiddenLayers, stepLayers = NA,
                  rampLayers = NA, rectifierLayers = rectifierLayers, sigmoidLayers = sigmoidLayers,
                  maxEpochs = maxEpochs, batchSize = batchSize, momentum = momentum, L1 = L1, L2 = L2,
                  validLoss = validLoss, validProp = validProp, earlyStop = earlyStop, earlyStopEpochs = earlyStopEpochs,
                  lrSched = lrSched, lrSchedEpochs = lrSchedEpochs, lrSchedLearnRates = lrSchedLearnRates, nTrain = nTrain,
                  nSteps = 0, smoothSteps = 0, autoencoder = FALSE, nColX = nColX)
  
  dataList <- prepData(X = X, y = y, nColX = nColX, nColY = nColY, standardize = standardize, 
                       regression = regression, nTot = nTot, nTrain = nTrain, nVal = nVal)
  
  startVal <- init(hiddenLayers = hiddenLayers, lossFunction = lossFunction, regression = regression,
                   stepLayers = NA, rampLayers = NA, rectifierLayers = rectifierLayers,
                   sigmoidLayers = sigmoidLayers, nColX = nColX, nColY = nColY, verbose = verbose)
  
  NNfit <- stochGD(dataList = dataList, nTrain = nTrain, standardize = standardize, activTypes = startVal$activTypes, 
                   lossType = lossFunction, dHuber = dHuber, nSteps = 0, smoothSteps = 0, batchSize = batchSize,
                   maxEpochs = maxEpochs, learnRate = learnRate, momentum = momentum, L1 = L1, L2 = L2,
                   earlyStop = earlyStop, earlyStopEpochs = earlyStopEpochs, earlyStopTol = earlyStopTol,
                   lrSched = lrSched, lrSchedEpochs = lrSchedEpochs, lrSchedLearnRates = lrSchedLearnRates,
                   fpOut = list(NA), bpOut = list(NA), upOut = startVal$upOut, validLoss = validLoss,
                   verbose = verbose, regression = regression, plotExample = FALSE)
  
  NN <- list(X       = X, trainInd = dataList$trainInd, valInd = dataList$valInd, hiddenLayers = hiddenLayers,
             print   = list(call = NN_call, overview = startVal$overview, nEpochs = NNfit$NN_plot$nEpochs),
             y_val   = dataList$y_val, pred = NNfit$NN_pred, plot = NNfit$NN_plot, reconstruct = FALSE)
  class(NN) <- "ANN"
  return(NN)
}

#' @title Train a Replicator Neural Network
#'
#' @description
#' Train a Replicator Neural Network using Stohastic Gradient
#' descent with optional batch learning. See Hawkins et al. (2002) for details on
#' Replicator Neural Networks.
#'
#' @details
#' A function for training an Replicator Neural Network.
#'
#' @references #' Hawkins, Simon, et al. "Outlier detection using replicator neural
#' networks." DaWaK. Vol. 2454. 2002.
#'
#' @param X matrix with explanatory variables
#' @param hiddenLayers vector specifying the number of nodes in each layer. Set
#' to \code{NA} for a Network without any hidden layers
#' @param lossFunction which loss function should be used. Options are "log",
#' "quadratic", "absolute", "huber" and "pseudo-huber"
#' @param dHuber used only in case of loss functions "huber" and "pseudo-huber".
#' This parameter controls the cut-off point between quadratic and absolute loss.
#' @param stepLayers vector or integer specifying which layers should have
#' stepwise activation in its nodes
#' @param nSteps numeric integer specifying how many steps the step function should
#' have on the interval [0, 1]
#' @param smoothSteps numeric indicating the smoothness of the step function.
#' Smaller values result in smoother steps. Recommended to keep below 50 for
#' stability. If set to high, the derivative of the stepfunction will also be large
#' @param rampLayers vector or integer specifying which layers should have
#' ramplike activation in its nodes. This is equivalent to a stepfunction
#' with an infinite number of steps (limit of step function when nSteps and
#' smoothSteps go to infinity) but more efficient than using step function layer
#' with a large number for nSteps.
#' @param rectifierLayers vector or integer specifying which layers should have
#' rectifier activation in its nodes
#' @param sigmoidLayers vector or integer specifying which layers should have
#' sigmoid activation in its nodes
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
#' @param earlyStop logical indicating if early stopping should be used based on
#' the loss on a validation set. Only possible with \code{validLoss} set to \code{TRUE}
#' @param earlyStopEpochs after how many epochs without sufficient improvement
#' (as specified by \code{earlyStopTol}) should training be stopped.
#' @param earlyStopTol numerical value specifying tolerance for early stopping.
#' Can be either positive or negative. When set negative, training will be stopped
#' if improvements are made but improvements are smaller than tolerance.
#' @param lrSched logical indicating if a schedule for the learning rate should
#' be used. If \code{TRUE}, schedule as specified by \code{lrSchedEpochs} and
#' \code{lrSchedLearnRates} .
#' @param lrSchedLearnRates vector with elements specifying the learn rate to be used
#' after epochs determined by lrSchedEpochs.
#' @param lrSchedEpochs vector with elements specifying the epoch after which the
#' corresponding learn rate from vector \code{lrSchedLearnRates}. Length of vector
#' shoud be the same as length of \code{learnSchedLearnRates}.
#' @return An \code{ANN} object. Use function \code{plot(<object>)} to assess
#' loss on training and optionally validation data during training process. Use
#' function \code{predict(<object>, <newdata>)} for prediction.
#' @examples
#' # Replicator
#' repNN <- replicator(faithful, hiddenLayers = c(4,1,4), batchSize = 5,
#'                     learnRate = 1e-5, momentum = 0.5, L1 = 1e-3, L2 = 1e-3)
#' plot(repNN)
#'
#' rX <- reconstruct(repNN, faithful)
#' plot(rX, alpha = 0.05)
#' plot(faithful, col = (rX$mah_p < 0.05)+1, pch = 16)
#' @export
replicator <- function(X, hiddenLayers = c(10, 5, 10), lossFunction = "pseudo-huber", dHuber = 1, stepLayers = 2,
                       nSteps = 5, smoothSteps = 25, rampLayers = NA, rectifierLayers = NA, sigmoidLayers = NA,
                       standardize = TRUE, learnRate = 1e-06, maxEpochs = 1000, batchSize = 32, momentum = 0.2,
                       L1 = 1e-07, L2 = 1e-04, validLoss = TRUE, validProp = 0.2, verbose = TRUE, earlyStop = TRUE,
                       earlyStopEpochs = 50, earlyStopTol = -1e-07, lrSched = FALSE, lrSchedEpochs = NA,
                       lrSchedLearnRates = NA) {
  NN_call <- match.call()
  X <- y <- as.matrix(X)
  if (!all(apply(X, 2, is.numeric))){
    stop("X should be numeric")
  }
  if (all(is.na(hiddenLayers)) || is.null(hiddenLayers)) {
    stop("Replicator NN should have at least one hidden layer")
  }
  if ((all(is.na(rampLayers)) || is.null(rampLayers)) && (all(is.na(stepLayers)) || is.null(stepLayers))){
    stop("Replicator NN should have one layer with step function activation or ramp activation")
  }
  if (!validLoss) {
    validProp <- 0
  }
  nColX  <- nColY <- ncol(X)
  nTot   <- nrow(X)
  nTrain <- ceiling(nTot * (1 - validProp))
  nVal   <- nTot - nTrain
  
  checkParameters(lossFunction = lossFunction, dHuber = dHuber, hiddenLayers = hiddenLayers, stepLayers = stepLayers,
                  rampLayers = rampLayers, rectifierLayers = rectifierLayers, sigmoidLayers = sigmoidLayers,
                  maxEpochs = maxEpochs, batchSize = batchSize, momentum = momentum, L1 = L1, L2 = L2,
                  validLoss = validLoss, validProp = validProp, earlyStop = earlyStop, earlyStopEpochs = earlyStopEpochs,
                  lrSched = lrSched, lrSchedEpochs = lrSchedEpochs, lrSchedLearnRates = lrSchedLearnRates, nTrain = nTrain,
                  nSteps = nSteps, smoothSteps = smoothSteps, autoencoder = TRUE, nColX = nColX)
  
  if (!(lossFunction %in% c("quadratic", "huber", "pseudo-huber", "absolute"))) {
    warning("Loss function not one of \"huber\", \"pseudo-huber\", \"quadratic\", \"absolute\". Using pseudo-huber loss function.\n")
    lossFunction <- "pseudo-huber"
  }
  
  dataList <- prepData(X = X, y = y, nColX = nColX, nColY = nColY, standardize = standardize, 
                       regression = TRUE, nTot = nTot, nTrain = nTrain, nVal = nVal)
  
  startVal <- init(hiddenLayers = hiddenLayers, lossFunction = lossFunction, regression = TRUE,
                   stepLayers = stepLayers, rampLayers = rampLayers, rectifierLayers = rectifierLayers,
                   sigmoidLayers = sigmoidLayers, nColX = nColX, nColY = nColY, verbose = verbose)
  
  NNfit <- stochGD(dataList = dataList, nTrain = nTrain, standardize = standardize, activTypes = startVal$activTypes, 
                   lossType = lossFunction, dHuber = dHuber, nSteps = nSteps, smoothSteps = smoothSteps, 
                   batchSize = batchSize, maxEpochs = maxEpochs, learnRate = learnRate, momentum = momentum, L1 = L1, 
                   L2 = L2, earlyStop = earlyStop, earlyStopEpochs = earlyStopEpochs, earlyStopTol = earlyStopTol,
                   lrSched = lrSched, lrSchedEpochs = lrSchedEpochs, lrSchedLearnRates = lrSchedLearnRates,
                   fpOut = list(NA), bpOut = list(NA), upOut = startVal$upOut, validLoss = validLoss,
                   verbose = verbose, regression = TRUE, plotExample = FALSE)
  
  
  errX <- X - predictC(NNfit$NN_pred, as.matrix(X), standardize)
  MCD  <- robustbase::covMcd(errX)
  
  NN <- list(X       = X, trainInd = dataList$trainInd, valInd = dataList$valInd, hiddenLayers = hiddenLayers,
             print   = list(call = NN_call, overview = startVal$overview, nEpochs = NNfit$NN_plot$nEpochs),
             pred    = NNfit$NN_pred, plot = NNfit$NN_plot, reconstruct = TRUE,
             rec     = list(MCDcenter = MCD$center, MCDcov = MCD$cov, standardize = standardize,
                            replicator = TRUE, nSteps = nSteps, smoothSteps = smoothSteps))
  class(NN) <- "ANN"
  return(NN)
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
#' @param rectifierLayers vector or integer specifying which layers should have
#' rectifier activation in its nodes
#' @param sigmoidLayers vector or integer specifying which layers should have
#' sigmoid activation in its nodes
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
#' @param earlyStop logical indicating if early stopping should be used based on
#' the loss on a validation set. Only possible with \code{validLoss} set to \code{TRUE}
#' @param earlyStopEpochs after how many epochs without sufficient improvement
#' (as specified by \code{earlyStopTol}) should training be stopped.
#' @param earlyStopTol numerical value specifying tolerance for early stopping.
#' Can be either positive or negative. When set negative, training will be stopped
#' if improvements are made but improvements are smaller than tolerance.
#' @param lrSched logical indicating if a schedule for the learning rate should
#' be used. If \code{TRUE}, schedule as specified by \code{lrSchedEpochs} and
#' \code{lrSchedLearnRates} .
#' @param lrSchedLearnRates vector with elements specifying the learn rate to be used
#' after epochs determined by lrSchedEpochs.
#' @param lrSchedEpochs vector with elements specifying the epoch after which the
#' corresponding learn rate from vector \code{lrSchedLearnRates}. Length of vector
#' shoud be the same as length of \code{learnSchedLearnRates}.
#' @return An \code{ANN} object. Use function \code{plot(<object>)} to assess
#' loss on training and optionally validation data during training process. Use
#' function \code{predict(<object>, <newdata>)} for prediction.
#' @examples
#' # Autoencoder
#' aeNN <- autoencoder(faithful, hiddenLayers = c(4,1,4), batchSize = 5,
#'                     learnRate = 1e-5, momentum = 0.5, L1 = 1e-3, L2 = 1e-3)
#' plot(aeNN)
#'
#' rX <- reconstruct(aeNN, faithful)
#' plot(rX, alpha = 0.05)
#' plot(faithful, col = (rX$mah_p < 0.05)+1, pch = 16)
#' @export
autoencoder <- function(X, hiddenLayers = c(10, 5, 10), lossFunction = "pseudo-huber", dHuber = 1,
                        rectifierLayers = NA, sigmoidLayers = NA, standardize = TRUE, learnRate = 1e-06, maxEpochs = 1000,
                        batchSize = 32, momentum = 0.2, L1 = 1e-07, L2 = 1e-04, validLoss = TRUE, 
                        validProp = 0.2, verbose = TRUE, earlyStop = TRUE, earlyStopEpochs = 50, earlyStopTol = -1e-07,
                        lrSched = FALSE, lrSchedEpochs = NA, lrSchedLearnRates = NA) {
  NN_call <- match.call()
  X <- y <- as.matrix(X)
  if (!all(apply(X, 2, is.numeric))){
    stop("X should be numeric")
  }
  if (all(is.na(hiddenLayers)) || is.null(hiddenLayers)) {
    stop("Autoencoder should have at least one hidden layer")
  }
  if (!validLoss) {
    validProp <- 0
  }
  
  nColX  <- nColY <- ncol(X)
  nTot   <- nrow(X)
  nTrain <- ceiling(nTot * (1 - validProp))
  nVal   <- nTot - nTrain
  
  checkParameters(lossFunction = lossFunction, dHuber = dHuber, hiddenLayers = hiddenLayers, stepLayers = NA,
                  rampLayers = NA, rectifierLayers = rectifierLayers, sigmoidLayers = sigmoidLayers,
                  maxEpochs = maxEpochs, batchSize = batchSize, momentum = momentum, L1 = L1, L2 = L2,
                  validLoss = validLoss, validProp = validProp, earlyStop = earlyStop, earlyStopEpochs = earlyStopEpochs,
                  lrSched = lrSched, lrSchedEpochs = lrSchedEpochs, lrSchedLearnRates = lrSchedLearnRates, nTrain = nTrain,
                  nSteps = 1, smoothSteps = 1, autoencoder = TRUE, nColX = nColX)
  
  if (!(lossFunction %in% c("quadratic", "huber", "pseudo-huber", "absolute"))) {
    warning("Loss function not one of \"huber\", \"pseudo-huber\", \"quadratic\", \"absolute\". Using pseudo-huber loss function.\n")
    lossFunction <- "pseudo-huber"
  }
  
  dataList <- prepData(X = X, y = y, nColX = nColX, nColY = nColY, standardize = standardize, 
                       regression = TRUE, nTot = nTot, nTrain = nTrain, nVal = nVal)
  
  startVal <- init(hiddenLayers = hiddenLayers, lossFunction = lossFunction, regression = TRUE,
                   stepLayers = NA, rampLayers = NA, rectifierLayers = rectifierLayers,
                   sigmoidLayers = sigmoidLayers, nColX = nColX, nColY = nColY, verbose = verbose)
  
  NNfit <- stochGD(dataList = dataList, nTrain = nTrain, standardize = standardize, 
                   activTypes = startVal$activTypes, lossType = lossFunction,
                   dHuber = dHuber, nSteps = 0, smoothSteps = 0, batchSize = batchSize,
                   maxEpochs = maxEpochs, learnRate = learnRate, momentum = momentum, L1 = L1, L2 = L2,
                   earlyStop = earlyStop, earlyStopEpochs = earlyStopEpochs, earlyStopTol = earlyStopTol,
                   lrSched = lrSched, lrSchedEpochs = lrSchedEpochs, lrSchedLearnRates = lrSchedLearnRates,
                   fpOut = list(NA), bpOut = list(NA), upOut = startVal$upOut, validLoss = validLoss,
                   verbose = verbose, regression = TRUE, plotExample = FALSE)
  
  errX <- X - predictC(NNfit$NN_pred, as.matrix(X), standardize)
  MCD  <- robustbase::covMcd(errX)
  
  NN <- list(X       = X, trainInd = dataList$trainInd, valInd = dataList$valInd, hiddenLayers = hiddenLayers,
             print   = list(call = NN_call, overview = startVal$overview, nEpochs = NNfit$NN_plot$nEpochs),
             pred    = NNfit$NN_pred, plot = NNfit$NN_plot, reconstruct = TRUE,
             rec     = list(MCDcenter = MCD$center, MCDcov = MCD$cov, standardize = standardize,
                            replicator = FALSE, nSteps = NA, smoothSteps = NA))
  class(NN) <- "ANN"
  return(NN)
}

#' @title Visual examples of training a Neural Network
#'
#' @description
#' Some examples of training a neural network using simple randomly generated data.
#' The training process is visualized through plots. Most parameters can be adjusted
#' so that the effect of changes can be assessed by inspecting the plots.
#'
#' @details
#' One regression example and three classification examples are included. More
#' examples will be added in future versions of \code{ANN}.
#'
#' @param example_type which example to use. Possible values are \code{surface},
#' \code{polynomial}, \code{nested}, \code{linear}, \code{disjoint} and \code{multiclass}
#' @param example_n number of observations to generate
#' @param example_sdnoise standard deviation of random normal noise to be added to data
#' @param example_nframes number of frames to be plotted
#' @param hiddenLayers vector specifying the number of nodes in each layer. Set
#' to \code{NA} for a Network without any hidden layers
#' @param lossFunction which loss function should be used. Options are "log",
#' "quadratic", "absolute", "huber" and "pseudo-huber"
#' @param dHuber used only in case of loss functions "huber" and "pseudo-huber".
#' This parameter controls the cut-off point between quadratic and absolute loss.
#' @param rectifierLayers vector or integer specifying which layers should have
#' rectifier activation in its nodes
#' @param sigmoidLayers vector or integer specifying which layers should have
#' sigmoid activation in its nodes
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
#' @export
example_NN <- function(example_type = "nested", example_n = 500, example_sdnoise = 1, example_nframes = 30,
                       hiddenLayers = c(5, 5), lossFunction = "log", dHuber = 1, rectifierLayers = NA,
                       sigmoidLayers = NA, regression = FALSE, standardize = TRUE, learnRate = 1e-3, maxEpochs = 2000,
                       batchSize = 10, momentum = 0.3, L1 = 1e-07, L2 = 1e-04) {
  if(example_type == "surface"){ regression <- TRUE; if(lossFunction == "log") lossFunction <- "quadratic"}
  XYdata <- genrResponse(example_n, example_type, example_sdnoise)
  X      <- as.matrix(XYdata$X)
  y      <- as.matrix(XYdata$y)
  
  if (regression) {
    if (!(lossFunction %in% c("quadratic", "huber", "pseudo-huber", "absolute"))) {
      stop("Use loss functions \"huber\", \"pseudo-huber\", \"quadratic\" and \"absolute\" for regression.\n")
    }
  } else {
    if (lossFunction != "log")
      warning("Log loss recommended for classification.")
  }

  nColX  <- ncol(X)
  nColY  <- ifelse(regression, ncol(y), length(unique(y)))
  nTot   <- nrow(X)
  nTrain <- nTot 
  nVal   <- 0
  
  checkParameters(lossFunction = lossFunction, dHuber = dHuber, hiddenLayers = hiddenLayers, stepLayers = NA,
                  rampLayers = NA, rectifierLayers = rectifierLayers, sigmoidLayers = sigmoidLayers,
                  maxEpochs = maxEpochs, batchSize = batchSize, momentum = momentum, L1 = L1, L2 = L2,
                  validLoss = FALSE, validProp = 0, earlyStop = FALSE, earlyStopEpochs = 0,
                  lrSched = FALSE, lrSchedEpochs = 0, lrSchedLearnRates = 0, nTrain = nTrain,
                  nSteps = 0, smoothSteps = 0, autoencoder = FALSE, nColX = nColX)
  
  dataList <- prepData(X = X, y = y, nColX = nColX, nColY = nColY, standardize = standardize, 
                       regression = regression, nTot = nTot, nTrain = nTrain, nVal = nVal)
  
  startVal <- init(hiddenLayers = hiddenLayers, lossFunction = lossFunction, regression = regression,
                   stepLayers = NA, rampLayers = NA, rectifierLayers = rectifierLayers,
                   sigmoidLayers = sigmoidLayers, nColX = nColX, nColY = nColY, verbose = FALSE)
  
  fpOut <- bpOut <- list(NA)
  upOut <- startVal$upOut
  exMaxEpochs <- ceiling(maxEpochs/example_nframes)
  for (i in 1:example_nframes) {
    cur_epoch <- exMaxEpochs * i
    cur_maxEpoch <- min(exMaxEpochs, maxEpochs - exMaxEpochs * (i - 1))
    if (cur_maxEpoch <= 0)
      (break)()
    NNfit <- stochGD(dataList = dataList, nTrain = nTrain, standardize = standardize,
                     activTypes = startVal$activTypes, lossType = lossFunction, dHuber = dHuber, 
                     nSteps = 0, smoothSteps = 0, batchSize = batchSize, maxEpochs = cur_maxEpoch, learnRate = learnRate,
                     momentum = momentum, L1 = L1, L2 = L2, earlyStop = FALSE, earlyStopEpochs = 0,
                     earlyStopTol = 0, lrSched = FALSE, lrSchedEpochs = 0, lrSchedLearnRates = 0, fpOut = fpOut,
                     bpOut = bpOut, upOut = upOut, validLoss = FALSE, verbose = FALSE, regression = regression,
                     plotExample = TRUE)
    
    fpOut <- NNfit$NN_example$fpOut
    bpOut <- NNfit$NN_example$bpOut
    upOut <- NNfit$NN_example$upOut
    
    if (regression) {
      plot_regress(NNfit$NN_pred, XYdata$X, XYdata$y, epoch = min(cur_epoch, maxEpochs),
                   standardize = standardize)
    } else {
      plot_classif(NNfit$NN_pred, XYdata$X, XYdata$y, epoch = min(cur_epoch, maxEpochs),
                   standardize = standardize, example_type)
    }
    Sys.sleep(0.05)
  }
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
#' @param object Object of class \code{ANN}
#' @param X data (matrix) to reconstruct
#' @param mahalanobis logical indicating if Mahalanobis distance should be calculated
#' @return Reconstructed observations and optional Mahalanobis distances
#' @export
reconstruct <- function(object, X, mahalanobis = TRUE) {
  if (!object$reconstruct) {
    stop("Object is not of type autoencoder or replicator")
  }
  X       <- as.matrix(X)
  NN_rec  <- object$rec
  NN_pred <- object$pred
  recX    <- predictC(NN_pred, X, NN_rec$standardize)
  errX    <- recX - X
  if (!mahalanobis) {
    rANN <- list(reconstructed = recX, reconstruction_errors = errX)
  } else {
    dfChiSq <- ncol(X)
    mah_sq  <- stats::mahalanobis(errX, center = NN_rec$MCDcenter, cov = NN_rec$MCDcov)
    mah_p   <- 1 - stats::pchisq(mah_sq, df = dfChiSq)
    rANN    <- list(reconstructed = recX, reconstruction_errors = errX,
                    mah_sq = mah_sq, mah_p = mah_p, dfChiSq = dfChiSq)
  }
  class(rANN) <- "rANN"
  return(rANN)
}

#' @title Plot the step function used in \code{replicator}
#'
#' @description
#' Plot the projections of observations onto the stepfunction in the low-dimensional 
#' space of the compression layer 
#'
#' @param object Trained replicator neural network object
#' @param X Data to be plotted as points on steps
#' @param hidden_node Number of the node to plot
#' @param color Color of points. Replace with a vector indicating class-membership
#' to give points in classes different colors. 
#' @param derivative logical indicating if the derivative of the step function should be plotted.
#' @param ... further arguments to be passed to plot
#' @export
plotStepFunction <- function(object = NULL, X, hidden_node = 1, color = "red", 
                             derivative = FALSE, ...){
  NN_rec <- object$rec
  if (!NN_rec$replicator) {
    stop("Object not of type replicator")
  }
  comprLayerIO <- encode(object, X, returnInputs = TRUE)
  x     <- comprLayerIO$input[,hidden_node]
  y     <- comprLayerIO$activation[,hidden_node]
  x_lim <- c(min(-0.1, x), max(1.1, x))
  x_seq <- matrix(seq(from = (x_lim[1]-0.1), to = (x_lim[2]+0.1) , by = 0.001), ncol = 1)
  steps <- stepFun(x_seq, NN_rec$nSteps, NN_rec$smoothSteps)
  graphics::plot(x = x_seq, y = steps, type = "l", xlab = "input", ylab = "activation",
                 main = paste0("Step function, node ", hidden_node), xlim = x_lim)
  graphics::points(x = x, y = y, col = color)
  if (derivative) {
    dSteps <- stepGradFun(x_seq, NN_rec$nSteps,  NN_rec$smoothSteps)
    graphics::lines(x_seq, dSteps, col = "blue")
  }
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
  NN_pred <- object$pred
  newdata <- as.matrix(newdata)
  rawPred <- predictC(NN_pred, newdata, NN_pred$standardize)
  if (NN_pred$regression) {
    colnames(rawPred) <- NN_pred$y_names
    return(list(prediction = rawPred, probabilities = "Not applicable"))
  } else {
    colnames(rawPred) <- paste0("class_", NN_pred$y_names)
    predictions <- NN_pred$y_names[apply(rawPred, 1, which.max)]
    return(list(predictions = predictions, probabilities = rawPred))
  }
}

#' @title Plot training and validation loss
#' @description \code{plot} Generate plots of the loss against epochs
#' @details A genereric function for training neural nets
#' @param x Object of class \code{ANN}
#' @param ... further arguments to be passed to plot
#' @return Plots
#' @method plot ANN
#' @export
plot.ANN <- function(x, ...) {
  NN_plot <- x$plot
  ddMat <- NN_plot$descentDetails
  x_seq <- seq.int(1, NN_plot$nEpochs)
  graphics::plot(x = x_seq, y = ddMat[x_seq,1], type = "l", col = "red", xlab = "Epoch", ylab = "Loss", ...)
  if (NN_plot$validLoss) {
    graphics::lines(x_seq, ddMat[x_seq, 2], col = "blue")
    graphics::abline(v = which(ddMat[x_seq, 4] == 1), col = "darkgrey")
    graphics::legend('topright',c("Training","Validation"), lty = 1, col=c('red','blue'),bty ="n")
  }
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
  NN_print <- x$print
  cat("\nCall:\n", deparse(NN_print$call), "\n")
  cat("\nOverview:\n", NN_print$overview, "\n")
  cat("\nNumber of Epochs:\n", NN_print$nEpochs)
}

#' @title Encoding step 
#' @description Compress data according to trained replicator or autoencoder.
#' Outputs are the activations of the nodes in the middle layer for each 
#' observation in \code{newdata}
#' @param object Object of class \code{ANN}
#' @param newdata Data to compress
#' @param returnInputs Logical indicating whether the inputs to the compression
#' layer should be returned. Mainly used for plotting the activations. 
#' @export
encode <- function(object, newdata, returnInputs = FALSE){
  if (!object$reconstruct) {
    stop("Object is not of type autoencoder or replicator")
  }
  newdata       <- as.matrix(newdata)
  nHidden       <- length(object$hiddenLayers)
  middleLayer   <- ceiling(nHidden/2)
  NN_pred       <- object$pred
  middleLayerIO <- partialForward(NN_pred, newdata, NN_pred$standardize, FALSE, 0, middleLayer)
  if (returnInputs) {
    return(middleLayerIO)
  }
  compressed    <- middleLayerIO$activation
  colnames(compressed) <- paste0("hidden_Node", 1:NCOL(compressed))
  return(compressed)
}


#' @title Decoding step 
#' @description Decompress low-dimensional representation resulting from the nodes
#' of the middle layer. Output are the reconstructed inputs to function \code{encode}
#' @param object Object of class \code{ANN}
#' @param compressed Data to decompress
#' @export
decode <- function(object, compressed){
  if (!object$reconstruct) {
    stop("Object is not of type autoencoder or replicator")
  }
  compressed   <- as.matrix(compressed)
  nHidden      <- length(object$hiddenLayers)
  middleLayer  <- ceiling(nHidden/2)
  finalLayer   <- nHidden + 1
  NN_pred      <- object$pred
  finalLayerIO <- partialForward(NN_pred, compressed, FALSE, NN_pred$standardize, middleLayer, finalLayer)
  decompressed <- finalLayerIO$activation
  colnames(decompressed) <- NN_pred$y_names
  return(decompressed)
}

