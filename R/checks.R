
# Set and check data
setData <- function(X, Y, regression) {
  
  # Convert X to matrix
  X <- as.matrix(X)

  # (ERROR) missing values in X and Y
  miss_X <- any(is.na(X))
  miss_Y <- any(is.na(Y))
  if ( miss_X && miss_Y ) {
    stop('X and Y have missing values')
  } else if ( miss_X ) {
    stop('X has missing values')
  } else if ( miss_Y ) {
    stop('Y has missing values')
  }
  
  # (ERROR) matrix X all numeric columns
  if ( !all(apply(X, 2, is.numeric)) ) {
    stop('X should be numeric')
  }
  
  # Checks on Y for regression and classification
  if ( regression ) {
    
    # Convert Y to matrix
    Y <- as.matrix(Y)
    
    # (ERROR) matrix Y all numeric columns
    if ( !all(apply(Y, 2, is.numeric)) ) {
      stop('Y should be numeric for regression')
    }
    
    # Set classes to NULL (only relevant for classification)
    classes <- NULL
    
  } else {
    
    # (ERROR) matrix Y not single column
    if ( NCOL(Y) != 1 ) {
      stop('Y should be a vector or single-column matrix containing classes for classification')
    }
    
    # (ERROR) not enough classes for classification in Y
    if ( length(unique(Y)) < 1 ) {
      stop('Y contains one or less classes, minimum of two required for classification')
    }
    
    # One-hot encode Y and store classes
    classes <- sort(unique(Y))
    Y <- 1 * outer(c(Y), classes, '==')
    
    # Convert Y to matrix
    Y <- as.matrix(Y)
  }
  
  # (ERROR) not same number of observations in X and Y
  if ( nrow(X) != nrow(Y) ) {
    stop('Unequal numbers of observations in X and Y')
  }
  
  # Collect parameters in list
  return ( list(X = X, Y = Y, classes = classes) )
}

# Set and check network parameter list
setNetworkParams <- function(data, hidden.layers, standardize, regression) {
  
  # Booleans for standardizing X and Y
  standardize_X <- standardize
  standardize_Y <- ifelse(regression, standardize, FALSE)
  
  # No hidden layers, only input and output layers
  if ( all(is.na(hidden.layers)) || is.null(hidden.layers) ) {
    params <- list(hidden_layers = NULL, 
                   standardize_X = standardize_X, 
                   standardize_Y = standardize_Y)
    return ( params )
  }
  
  # (ERROR) missing in hidden.layers
  if ( any(is.na(hidden.layers)) && !all(is.na(hidden.layers)) ) {
    stop('hidden.layers contains NA')
  }
  
  # (ERROR) non-integers in hidden.layers
  if ( !all(hidden.layers %% 1 == 0) ) {
    stop('hidden.layers contains non-integers')
  }
  
  # Set vector with structure (number of nodes) in the network
  num_nodes <- as.integer( c(ncol(data$X), hidden.layers, ncol(data$Y)) )
  
  # Collect parameters in list
  return (
    list(num_nodes     = num_nodes, 
         standardize_X = standardize_X, 
         standardize_Y = standardize_Y)
    )
}

# Set and check loss parameter list
setLossParams <- function(loss.type, delta.huber, regression) {
  allowed_types <- c('log', 'squared', 'absolute', 'huber', 'pseudo-huber')
  
  # (ERROR) loss.type one of allowed types
  if ( !(loss.type %in% allowed_types) ) {
    stop('loss.type not one of ', paste(allowed_types, collapse = ', '))
  }
  
  # (WARN) do not use log loss for regression
  if ( regression && loss.type == 'log' ) {
    warning('loss.type "log" not recommended for regression')
  }
  
  # (WARN) use log loss for classification
  if ( !regression && loss.type != 'log' ) {
    warning('loss.type "log" recommended for classification')
  }
  
  # Checks on parameters specific to huber and pseudo-huber
  if ( loss.type %in% c('huber', 'pseudo-huber') ) {
    
    # (ERROR) delta should be non-negative
    if ( delta.huber < 0 ) {
      stop('delta.huber smaller than zero')
    }
    
    # (WARN) use absolute loss if delta equals zero
    if ( delta.huber == 0 ) {
      warning ('delta.huber equal to zero, this is equivalent to using absolute loss but less efficient')
    }
  }
  
  # Collect parameters in list
  return ( list(type = loss.type, delta_huber = delta.huber) )
}

# Set and check activation parameter list
setActivParams <- function(activ.functions, regression, hidden.layers) {
  allowed_types <- c('tanh', 'sigmoid', 'relu', 'linear', 'ramp', 'step')
  
  # No hidden layers: return parameter list with types set to NULL
  if ( all(is.na(hidden.layers)) || is.null(hidden.layers) ) {
    return ( list(types = NULL) )
  }
  
  # (ERROR) activ.functions not of allowed types
  if ( !(all(activ.functions %in% allowed_types)) ) {
    stop('activ.functions not all one of types ', paste(allowed_types, collapse = ', '))
  }
  
  # Single activ.function specified by user
  if ( length(activ.functions) != length(hidden.layers) ) {
    
    # (ERROR) length of activ.functions either 1 (broadcasting) or equal to the 
    #         number of hidden layers
    if ( length(activ.functions) != 1 ) {
      stop('length of activ.functions should be one or equal to number of hidden layers')
    }
    
    # Broadcast single activ.function over all hidden layers
    types <- rep(activ.functions, length(hidden.layers))
  }
  
  # Append activation type of output layer (and 'input', not used)
  output_type <- ifelse(regression, 'linear', 'softMax')
  types  <- c('input', activ.functions, output_type)
  
  # Collect parameters in list
  return ( list(types = types) )
}

# Set and check optimizer parameter list
setOptimParams <- function(optim.type, learn.rates, momentum, L1, L2, hidden.layers) {
  allowed_types <- c('sgd')
  
  # (ERROR) optim.type not of allowed type
  if ( !(optim.type %in% allowed_types) ) {
    stop('currently only optim.type "sgd" supported')
    # stop('optim.type not one of types ', paste(allowed_types, collapse = ', '))
  }
  
  # (ERROR) learn.rate incorrect value
  if ( any(learn.rates <= 0) ) {
    stop('learn rates should be larger than 0')
  }
  
  # Single learn.rate specified by user
  if ( length(learn.rates) != length(hidden.layers) + 1 ) {
    
    # (ERROR) length of learn.rates should be either 1 (broadcasting) or
    #         equal to the number of hidden layers + 1
    if ( length(learn.rates) != 1 ) {
      stop('length of learn.rates should be one or equal to number of hidden layers + 1')
    }
    
    # Broadcast single activ.function over all layers
    learn.rates <- rep(learn.rates, length(hidden.layers) + 1)
  }
  
  # Push learn rate for input layer to front (not used but easier for init)
  learn.rates <- c(0, learn.rates)
  
  # (ERROR) momentum incorrect value
  if ( momentum >= 1 || momentum < 0 ) {
    stop('momentum larger than or equal to one or smaller than zero')
  }
  
  # (ERROR) L1 and/or L2 incorrect value(s)
  if ( L1 < 0 || L2 < 0 ) {
    stop('L1 and/or L2 negative')
  }
  
  # Collect parameters in list
  return ( 
    list(type        = optim.type, 
         learn_rates = learn.rates, 
         m           = momentum, 
         L1          = L1, 
         L2          = L2)
    )
}

setTrainParams <- function (data, n.epochs, batch.size, val.prop, drop.last, verbose) {
  
  # (ERROR) n.epochs not positive
  if ( n.epochs <= 0 ) {
    stop('n.epochs should be larger than zero')
  }
  
  # (ERROR) n.epochs not whole number
  if ( n.epochs %% 1 != 0 ){
    stop('n.epochs not an integer')
  }
  
  # (ERROR) batch.size not positive
  if ( batch.size <= 0 ) {
    stop('batch.size should be larger than zero')
  }
  
  # (ERROR) batch.size not whole number
  if ( batch.size %% 1 != 0 ){
    stop('batch.size not an integer')
  }
  
  # (ERROR) val.prop incorrect value
  if ( val.prop < 0 || val.prop >= 1){
    stop('val.prop should be smaller than one and larger than or equal to zero')
  }
  
  # Size of training and validation sets
  n_obs   <- nrow(data$X)
  n_train <- ceiling( n_obs * (1 - val.prop) )
  n_val   <- n_obs - n_train
  
  # (ERROR) batch.size larger than n_obs
  if ( batch.size > n_obs ){
    stop('batch.size larger than total number of observations in X and Y')
  }
  
  # (ERROR) batch.size larger than n_train
  if ( n_val > 0 && batch.size > n_train ){
    stop('batch.size larger than size of training data (after training/validation subsetting)')
  }
  
  # (WARN) small validation set
  if ( val.prop > 0 && n_val < 25 ){
    warning('small validation set, only ', n_val, ' observations')
  }
  
  # Collect parameters in list
  return (
    list(n_epochs   = n.epochs, 
         batch_size = batch.size, 
         val_prop   = val.prop, 
         drop_last  = drop.last, 
         verbose    = verbose)
  )
  
}

