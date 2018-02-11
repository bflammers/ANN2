checkParameters <- function(lossFunction, dHuber, hiddenLayers, stepLayers, rampLayers, rectifierLayers, linearLayers,
                            sigmoidLayers, maxEpochs, batchSize, momentum, L1, L2, validLoss, validProp, earlyStop,
                            earlyStopEpochs, lrSched, lrSchedEpochs, lrSchedLearnRates, nTrain, nSteps, smoothSteps,
                            autoencoder, nColX) {
  noHiddenLayers <- (all(is.na(hiddenLayers)) || is.null(hiddenLayers))
  
  ## ERRORS
  if (!(lossFunction %in% c("log", "huber", "pseudo-huber", "quadratic", "absolute"))) {
    stop("Loss function not one of \"log\", \"huber\", \"pseudo-huber\", \"quadratic\", \"absolute\".")
  }
  if ((lossFunction %in% c("huber", "pseudo-huber")) && dHuber <= 0) {
    stop("dHuber should be positive and non-zero.")
  }
  if (nTrain < batchSize) {
    stop(paste0("Batchsize should be between zero and number of observations in trainset: ", nTrain))
  }
  if (momentum >= 1 || momentum < 0) {
    stop("Momentum should be smaller than one and larger than or equal to zero.")
  }
  if (L1 < 0 || L2 < 0) {
    stop("L1 and L2 should be larger than or equal to zero.")
  }
  if (validLoss) {
    if (validProp > 1 || validProp < 0) {
      stop("validProp should be between zero and one")
    }
    if (earlyStop) {
      if (earlyStopEpochs > maxEpochs) {
        stop("earlyStopEpochs should be smaller than maxEpochs")
      }
    }
  }
  if (lrSched) {
    if (any(is.na(lrSchedEpochs)) || is.null(lrSchedEpochs)) {
      stop("lrSchedEpochs should be an integer or vector of integers for learning rate schedule")
    }
    if (any(is.na(lrSchedLearnRates)) || is.null(lrSchedLearnRates)) {
      stop("lrSchedLearnRates should be numeric or vector of numerics for learning rate schedule")
    }
    if (length(lrSchedEpochs) != length(lrSchedLearnRates)) {
      stop("Parametervectors lrSchedEpochs and lrSchedLearnRates should be same sized")
    }
  }
  if (!noHiddenLayers && !is.vector(hiddenLayers) ) {
    stop("hiddenLayers should be an integer or a (numeric) vector of integers")
  }
  if (!noHiddenLayers && !all(hiddenLayers > 0)) {
    stop("All elements of hiddenLayers should be non-zero and positive.")
  }
  
  noLinLayers  <- (all(is.na(linearLayers)) || is.null(linearLayers))
  noStepLayers <- (all(is.na(stepLayers)) || is.null(stepLayers))
  noRampLayers <- (all(is.na(rampLayers)) || is.null(rampLayers))
  noRectLayers <- (all(is.na(rectifierLayers)) || is.null(rectifierLayers))
  noSigmLayers <- (all(is.na(sigmoidLayers)) || is.null(sigmoidLayers))
  allLayers    <- !c(noStepLayers, noRampLayers, noRectLayers, noSigmLayers)
  
  if(!noStepLayers){
    if(nSteps<0) stop("nSteps cannot be negative")
    if(smoothSteps<=0) stop("smoothSteps cannot be non-positive")
  }
  
  if (noHiddenLayers) {
    if(any(allLayers)){
      stop("Cannot use ", paste0(c("step", "ramp", "rectifier",
                                   "sigmoid")[allLayers], collapse = ", "), " layers with no hidden layers")
    }
  } else {
    seqLayers <- seq.int(1, length(hiddenLayers))
    if (!noLinLayers && any(!(linearLayers %in% linearLayers))) {
      stop("Cannot set hidden layer to linear layer. Possible layers are: ", paste0(seqLayers, collapse = ", "))
    }
    if (!noStepLayers && any(!(stepLayers %in% seqLayers))) {
      stop("Cannot set hidden layer to step layer. Possible layers are: ", paste0(seqLayers, collapse = ", "))
    }
    if (!noRampLayers && any(!(rampLayers %in% seqLayers))) {
      stop("Cannot set hidden layer to ramp layer. Possible layers are: ", paste0(seqLayers, collapse = ", "))
    }
    if (!noRectLayers && any(!(rectifierLayers %in% seqLayers))) {
      stop("Cannot set hidden layer to rectifier layer. Possible layers are: ", paste0(seqLayers, collapse = ", "))
    }
    if (!noSigmLayers && any(!(sigmoidLayers %in% seqLayers))) {
      stop("Cannot set hidden layer to sigmoid layer. Possible layers are: ", paste0(seqLayers, collapse = ", "))
    }
    
    activList <- list(linearLayers, stepLayers, rampLayers, rectifierLayers, sigmoidLayers)
    activList[is.na(activList)] <- NULL
    if(length(activList)>0){
      for(i in 1:length(activList)){
        if(any(activList[[i]] %in% unlist(activList[-i]))) {
          stop("linearLayers, stepLayers, rampLayers, rectifierLayers and sigmoidLayers cannot overlap")
        }
      }
    }
  }
  
  ## WARNINGS
  if (validLoss && validProp == 0) {
    warning("Cannot evaluate validation loss with validProp equal to zero.")
  }
  if (earlyStop && (earlyStopEpochs > maxEpochs)) {
    warning("Cannot do early stopping when earlyStopEpochs > maxEpochs")
  }
  if (lrSched && (lrSchedEpochs > maxEpochs)) {
    warning("Cannot do learning rate schedule when lrSchedEpochs > maxEpochs")
  }
  
  # AUTOENCODER ERRORS AND WARINGS
  if (autoencoder) {
    nHidden     <- length(hiddenLayers)
    middleLayer <- ceiling(nHidden/2)
    if (nHidden %% 2 == 0) {
      stop("Number of layers should be odd such that there is a middle layer")
    }
    if (middleLayer != which.min(hiddenLayers)) {
      warning("Hidden layer not the layer with the minimal number of nodes")
    }
    if (hiddenLayers[middleLayer] > nColX) {
      warning("No data compression in middle layer since the number of nodes is larger than the number of input variables")
    }
  }
}


init <- function(hiddenLayers, lossFunction, regression, stepLayers, rampLayers,
                 rectifierLayers, linearLayers, sigmoidLayers, nColX, nColY, verbose) {
  
  # Define structure of network
  noHiddenLayers <- (all(is.na(hiddenLayers)) || is.null(hiddenLayers))
  if (noHiddenLayers) {
    n_Hidden <- 0
    n_Hid_Out <- n_Hidden + 1L
    seqHidden <- seq.int(1L, n_Hidden)
    seqHid_Out <- seq.int(1L, n_Hid_Out)
    strucLayers <- c(nColX, nColY)
    strucWeights <- strucLayers[seqHid_Out] * strucLayers[seqHid_Out + 1L]
    activTypes <- ifelse(regression, "linear", "softMax")
    if (verbose) {
      # Overview layers
      cat("Neural Network: \n Input layer:   ", nColX, "nodes \n", "Output layer:  ",
          nColY, "nodes -", activTypes, "(", lossFunction, "loss ) \n")
    }
  } else {
    n_Hidden <- length(hiddenLayers)
    n_Hid_Out <- n_Hidden + 1L
    seqHidden <- seq.int(1L, n_Hidden)
    seqHid_Out <- seq.int(1L, n_Hid_Out)
    strucLayers <- c(nColX, hiddenLayers, nColY)
    strucWeights <- strucLayers[seqHid_Out] * strucLayers[seqHid_Out + 1L]
    activTypes <- rep("tanh", n_Hidden)
    activTypes[seqHidden %in% sigmoidLayers]   <- "sigmoid"
    activTypes[seqHidden %in% rectifierLayers] <- "rectifier"
    activTypes[seqHidden %in% rampLayers]      <- "ramp"
    activTypes[seqHidden %in% stepLayers]      <- "step"
    activTypes[seqHidden %in% linearLayers]    <- "linear"
    activTypes <- c(activTypes, ifelse(regression, "linear", "softMax"))
    
    
    # Overview layers
    overviewNetwork <-  paste0("Neural Network: \n Input layer: ", strrep(" ", 10 - nchar(nColX)), nColX, " nodes\n",
                               paste0(" hidden layer ", seqHidden, ": ", strrep(" ", 7 - nchar(hiddenLayers)), hiddenLayers,
                                      " nodes - ", activTypes[-n_Hid_Out], "\n", collapse = ""), " Output layer: ",
                               strrep(" ", 9 - nchar(nColY)), nColY, " nodes - ", activTypes[n_Hid_Out], " (", lossFunction, " loss) \n")
    if (verbose) cat(overviewNetwork)
  }
  
  # Initialize weights, biases, momentums, etc...
  biasVecs <- biasMomVecs <- sapply(strucLayers[-1L], function(n_nodes) rep(0, n_nodes), simplify = FALSE)
  weightMats <- sapply(seq.int(1L, n_Hidden + 1L), function(w) {
    matrix(stats::rnorm(strucWeights[w])/sqrt(strucLayers[w]), 
           nrow = strucLayers[w + 1L], ncol = strucLayers[w])
  }, simplify = FALSE)
  weightMomMats <- weightGradMats <- sapply(seq.int(1L, n_Hidden + 1L), function(w) {
    matrix(0, nrow = strucLayers[w + 1L], ncol = strucLayers[w])
  }, simplify = FALSE)
  
  return(list(activTypes = activTypes, overview = overviewNetwork,
              upOut = list(weightMats = weightMats, weightMomMats = weightMomMats,
                           biasVecs = biasVecs, biasMomVecs = biasMomVecs)))
  
}

prepData <- function(X, y, nColX, nColY, standardize, regression, nTot, nTrain, nVal) {
  if (regression) {
    if (standardize) {
      # X
      scaledX  <- scaleData(X)
      sX       <- scaledX$scaled
      X_center <- scaledX$center
      X_scale  <- scaledX$scale
      # Y
      scaledY  <- scaleData(y)
      sY       <- scaledY$scaled
      y_center <- scaledY$center
      y_scale  <- scaledY$scale
    } else {
      sX       <- X
      sY       <- y
      y_center <- X_center <- 0
      y_scale  <- X_scale  <- 1
    }
    y_names <- colnames(y)
  } else {
    # Classification
    y_names  <- sort(unique(y))
    sY       <- t(sapply(y, function(c_y) ifelse(c_y == y_names, 1L, 0L)))
    y_vec    <- FALSE
    y_center <- 0
    y_scale  <- 1
    if (standardize) {
      # X
      scaledX  <- scaleData(X)
      sX       <- scaledX$scaled
      X_center <- scaledX$center
      X_scale  <- scaledX$scale
    } else {
      sX       <- X
      X_center <- 0
      X_scale  <- 1
    }
  }
  
  if (is.null(y_names)) {
    y_names <- paste0("y", seq.int(1, nColY))
  }
  # Make validationset and trainset
  seqObs   <- seq.int(1, nTot)
  trainInd <- sample(seqObs, size = nTrain)
  valInd   <- setdiff(seqObs, trainInd)
  
  X_val   <- matrix(sX[valInd, ],   nrow = nVal,   ncol = nColX)
  X_train <- matrix(sX[trainInd, ], nrow = nTrain, ncol = nColX)
  y_val   <- matrix(sY[valInd, ],   nrow = nVal,   ncol = nColY)
  y_train <- matrix(sY[trainInd, ], nrow = nTrain, ncol = nColY)
  
  return(list(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, 
              X_center = X_center, X_scale = X_scale, y_center = y_center, y_scale = y_scale, 
              y_names = y_names, trainInd = trainInd, valInd = valInd))
}

plot_classif <- function(NN, X, y, epoch = "", standardize, example_type) {
  if (!requireNamespace("reshape2", quietly = TRUE)) {
    stop("Package \"reshape2\" needed for plotting classification example. Please install.", call. = FALSE)
  }
  x.seq <- seq(min(X), max(X), by = 0.1)
  xy.plot <- as.matrix(expand.grid(x.seq, x.seq))
  xy.pred <- data.frame(xy.plot, apply(predictC(NN, xy.plot, standardize), 1, which.max))
  #xy.pred <- data.frame(xy.plot, predict.NN(object = NN, newdata = xy.plot)$predictions)
  colnames(xy.pred) <- c("X1", "X2", "P")
  df.plot <- reshape2::dcast(xy.pred, X1 ~ X2, value.var = "P")
  graphics::image(x = x.seq, y = x.seq, data.matrix(df.plot[, -1]), 
                  col = c("#FFDEA6", "#B7E7FF", "#EEEEEE"),
                  main = paste0("Epoch: ", epoch), xlab = "x", ylab = "y")
  graphics::points(X, col = y + 1)
  # Add decision boundaries
  if (example_type == "linear") {
    graphics::lines(c(0, 1), c(0, 1), col = "darkgrey")
    graphics::lines(c(0, 1/3), c(1, 1/3), col = "darkgrey")
  } else if (example_type == "polynomial") {
    x.seq <- seq(-3, 7, by = 0.2)
    graphics::lines(x.seq, 2/3 * x.seq^2 - 1/9 * x.seq^3, col = "darkgrey")
  } else if (example_type == "nested") {
    c.seq <- seq(0, 2 * pi + 0.1, by = 0.1)
    r1 <- 17
    r2 <- 5
    loc <- c(2, 2)
    graphics::lines(sqrt(r1) * cos(c.seq) + loc[2], sqrt(r1) * sin(c.seq) + loc[2], col = "darkgrey")
    graphics::lines(sqrt(r2) * cos(c.seq) + loc[2], sqrt(r2) * sin(c.seq) + loc[2], col = "darkgrey")
  } else if (example_type == "disjoint") {
    c.seq <- seq(0, 2 * pi + 0.1, by = 0.1)
    r1 <- 6
    r2 <- 4
    loc1 <- c(-0.5, 0)
    loc2 <- c(4, 4)
    graphics::lines(sqrt(r1) * cos(c.seq) + loc1[1], sqrt(r1) * sin(c.seq) + loc1[2], col = "darkgrey")
    graphics::lines(sqrt(r2) * cos(c.seq) + loc2[1], sqrt(r2) * sin(c.seq) + loc2[2], col = "darkgrey")
  }
}

plot_regress <- function(NN, X, y, epoch = "", standardize) {
  if (!requireNamespace("rgl", quietly = TRUE)) {
    stop("Package \"rgl\" needed for plotting regression example. Please install.", call. = FALSE)
  }
  x.seq   <- seq(min(X), max(X), by = 0.2)
  xy.plot <- as.matrix(expand.grid(x.seq, x.seq))
  xy.pred <- data.frame(xy.plot, predictC(NN, xy.plot, standardize))
  colnames(xy.pred) <- c("X1", "X2", "Y")
  df.plot <- reshape2::dcast(xy.pred, X1 ~ X2, value.var = "Y")
  rgl::plot3d(cbind(X, y), main = paste0("Epoch: ", epoch), xlab = "", ylab = "", zlab = "")
  rgl::persp3d(x = x.seq, y = x.seq, z = as.matrix(df.plot[, -1]), col = c("red"), alpha = 0.4, add = TRUE)
}


genrResponse <- function(n, type, sd.noise) {
  # 'linear' 'polynomial' 'disjoint' 'multiclass' 'nested' 'surface'
  if (type == "linear") {
    X <- matrix(stats::runif(2 * n), ncol = 2, nrow = n)
    y <- rep(0, n)
    y[-2 * X[, 1] + 1 > (X[, 2] + stats::rnorm(nrow(X), sd = sd.noise))] <- 1
    y[X[, 1] > (X[, 2] + stats::rnorm(nrow(X), sd = sd.noise))] <- 2
  } else if (type == "polynomial") {
    X <- matrix(stats::runif(2 * n, -3, 7), ncol = 2, nrow = n)
    y <- 2/3 * X[, 1]^2 - 1/9 * X[, 1]^3 > (X[, 2] + 
                                              stats::rnorm(nrow(X), sd = sd.noise))
  } else if (type == "nested") {
    X <- matrix(stats::runif(2 * n, -4, 8), ncol = 2, nrow = n)
    loc <- c(2, 2)
    y <- rep(0, nrow(X))
    y[(X[, 1] - loc[1])^2 + (X[, 2] - loc[2])^2 < 17 + 
        stats::rnorm(nrow(X), sd = sd.noise)] <- 1
    y[(X[, 1] - loc[1])^2 + (X[, 2] - loc[2])^2 < 5 + 
        stats::rnorm(nrow(X), sd = sd.noise)] <- 2
  } else if (type == "disjoint") {
    X <- matrix(stats::runif(2 * n, -4, 8), ncol = 2, nrow = n)
    loc1 <- c(-0.5, 0)
    loc2 <- c(4, 4)
    y <- rep(0, nrow(X))
    y[(X[, 1] - loc1[1])^2 + (X[, 2] - loc1[2])^2 < 6 + 
        stats::rnorm(nrow(X), sd = sd.noise)] <- 1
    y[(X[, 1] - loc2[1])^2 + (X[, 2] - loc2[2])^2 < 4 + 
        stats::rnorm(nrow(X), sd = sd.noise)] <- 2
  } else if (type == "surface") {
    X <- matrix(stats::runif(2 * n, -4, 6), ncol = 2, nrow = n)
    y <- X[, 1] * sin(X[, 2]) - X[, 2] * cos(X[, 1]) + 
      stats::rnorm(nrow(X), mean = 3, sd = sd.noise)
  } else if (type == "yin-yang") {
    r  <- seq(0.05, 0.8,   length.out = n) 
    t1 <- seq(0,    2.6, length.out = n) + stats::rnorm(n, sd = sd.noise) 
    c1 <- cbind(r*sin(t1)-0.1 , r*cos(t1)+0.1) 
    t2 <- seq(9.4,  12, length.out = n) + stats::rnorm(n, sd = sd.noise) 
    c2 <- cbind(r*sin(t2)+0.1 , r*cos(t2)-0.1) 
    X  <- rbind(c1, c2)
    y  <- as.numeric(1:(2*n)>n)+1
  } else print(paste0("Type ", type, " not supported."))
  return(list(X = X, y = as.numeric(y)))
}
