
# Set network meta info
setMeta <- function(data, hidden.layers, regression) {
  
  # Check if network has no hidden layers, set n_hidden layers accordingly
  if ( is.null(hidden.layers) ) {
    no_hidden <- TRUE
    n_hidden  <- 0
  } else {
    if ( length(hidden.layers) == 1 && all(is.na(hidden.layers)) ) {
      no_hidden <- TRUE
      n_hidden  <- 0
    } else {
      no_hidden <- FALSE
      n_hidden  <- length(hidden.layers)
    }
  }
  
  # Set number of nodes in input and output layer
  n_in  <- ncol(data$X)
  n_out <- ncol(data$Y)
  
  # Set number of observations
  n_obs <- nrow(data$X)
  
  return( list(no_hidden = no_hidden, n_hidden = n_hidden, n_in = n_in, 
               n_out = n_out, n_obs = n_obs, regression = regression) )
}


# Set and check data
setData <- function(X, Y, regression) {
  
  # Convert X to matrix
  X <- as.matrix(X)

  # (ERROR) missing values in X and Y
  miss_X <- any(is.na(X))
  miss_Y <- any(is.na(Y))
  if ( miss_X && miss_Y ) {
    stop('X and Y contain missing values', call. = FALSE)
  } else if ( miss_X ) {
    stop('X contain missing values', call. = FALSE)
  } else if ( miss_Y ) {
    stop('Y contain missing values', call. = FALSE)
  }
  
  # (ERROR) matrix X all numeric columns
  if ( !all(apply(X, 2, is.numeric)) ) {
    stop('X should be numeric', call. = FALSE)
  }
  
  # Checks on Y for regression and classification
  if ( regression ) {
    
    # Convert Y to matrix
    Y <- as.matrix(Y)
    
    # (ERROR) matrix Y all numeric columns
    if ( !all(apply(Y, 2, is.numeric)) ) {
      stop('Y should be numeric for regression', call. = FALSE)
    }
    
    # Set classes to NULL (only relevant for classification)
    classes <- NULL
    
  } else {
    
    # (ERROR) matrix Y not single column
    if ( NCOL(Y) != 1 ) {
      stop('Y should be a vector or single-column matrix containing classes for classification', 
           call. = FALSE)
    }
    
    # (ERROR) not enough classes for classification in Y
    if ( length(unique(Y)) < 1 ) {
      stop('Y contains one or less classes, minimum of two required for classification',
           call. = FALSE)
    }
    
    # One-hot encode Y and store classes
    classes <- sort(unique(Y))
    Y <- 1 * outer(c(Y), classes, '==')
    
    # Convert Y to matrix
    Y <- as.matrix(Y)
  }
  
  # (ERROR) not same number of observations in X and Y
  if ( nrow(X) != nrow(Y) ) {
    stop('Unequal numbers of observations in X and Y', call. = FALSE)
  }
  
  # Collect parameters in list
  return ( list(X = X, Y = Y, classes = classes) )
}

# Set and check network parameter list
setNetworkParams <- function(hidden.layers, standardize, meta) {
  
  # Booleans for standardizing X and Y
  stand_X <- standardize
  stand_Y <- ifelse(meta$regression, standardize, FALSE)
  
  # Check if hidden layers NULL
  if ( meta$no_hidden ) {
    
    # Only input and output layers
    num_nodes <- as.integer( c(meta$n_in, meta$n_out) )
    
  } else {
    
    # (ERROR) missing in hidden.layers
    if ( any(is.na(hidden.layers)) ) {
      stop('hidden.layers contains NA', call. = FALSE)
    }
    
    # (ERROR) non-integers in hidden.layers
    if ( !all(hidden.layers %% 1 == 0) ) {
      stop('hidden.layers contains non-integers', call. = FALSE)
    }
      
    # Set vector with structure (number of nodes) in the network
    num_nodes <- as.integer( c(meta$n_in, hidden.layers, meta$n_out) )
    
  }

  # Collect parameters in list
  return ( list(num_nodes = num_nodes, stand_X = stand_X, stand_Y = stand_Y) )
}

# Set and check activation parameter list
setActivParams <- function(activ.functions, H, k, meta) {
  allowed_types <- c('tanh', 'sigmoid', 'relu', 'linear', 'ramp', 'step')
  
  # Determine output type
  output_type <- ifelse(meta$regression, 'linear', 'softmax')
  
  # Check if no hidden layers
  if ( meta$no_hidden ) {
    
    # (WARN) no hidden layers so activ.functions will not be used
    if ( !is.null(activ.functions) && !all(is.na(activ.functions)) ) {
      warning('network with no hidden layers, specified activ.functions not used', 
              call. = FALSE)
    }
    
    # Append activation type of output layer (and 'input', not used)
    types  <- c('input', output_type)
    
  } else {
    
    # (ERROR) activ.functions not of allowed types
    if ( !(all(activ.functions %in% allowed_types)) ) {
      stop('activ.functions not all one of types ', 
           paste(allowed_types, collapse = ', '), call. = FALSE)
    }
    
    # Single activ.function specified by user (or incorrect number of values)
    if ( length(activ.functions) != meta$n_hidden ) {
      
      # (ERROR) length of activ.functions either 1 (broadcasting) or equal to the 
      #         number of hidden layers
      if ( length(activ.functions) != 1 ) {
        stop('length of activ.functions should be either one (for broadcasting) or equal to number of hidden layers', 
             call. = FALSE)
      }
      
      # Broadcast single activ.function over all hidden layers
      types <- c('input', rep(activ.functions, meta$n_hidden), output_type)
      
    } else {
      
      # No broadcasting
      types  <- c('input', activ.functions, output_type)
    }
  }
  
  # Check parameters H and k, only for step activation function
  if ( 'step' %in% activ.functions ) {
    
    # (ERROR) number of steps H should be a whole number larger than zero
    if ( H %% 1 != 0  || H < 1 ) {
      stop('H should be an integer larger than zero', call. = FALSE)
    }
    
    # (ERROR) smoothing parameter k should be and integer larger than zero
    if ( k %% 1 != 0  || k < 1 ) {
      stop('k should be an integer larger than zero', call. = FALSE)
    }
  }
  
  # Collect parameters in list
  return ( list(types = types, H = H, k = k) )
}

# Set and check optimizer parameter list
setOptimParams <- function(optim.type, learn.rates, momentum, L1, L2, meta) {
  allowed_types <- c('sgd')
  
  # (ERROR) optim.type not of allowed type
  if ( !(optim.type %in% allowed_types) ) {
    stop('currently only optim.type "sgd" supported', call. = FALSE)
    # stop('optim.type not one of types ', paste(allowed_types, collapse = ', '))
  }
  
  # (ERROR) momentum incorrect value
  if ( momentum >= 1 || momentum < 0 ) {
    stop('momentum larger than or equal to one or smaller than zero', call. = FALSE)
  }
  
  # (ERROR) L1 and/or L2 incorrect value(s)
  if ( L1 < 0 || L2 < 0 ) {
    stop('L1 and/or L2 negative', call. = FALSE)
  }
  
  # (ERROR) missing learn.rates
  if ( is.null(learn.rates) || any(is.na(learn.rates)) ) {
    stop('missing value in learn.rates or learn.rates is NULL', call. = FALSE)
  } 
  
  # (ERROR) learn.rate incorrect value
  if ( any(learn.rates <= 0) ) {
    stop('learn rates should be larger than 0', call. = FALSE)
  }
  
  # No hidden layers: return parameter list with only output type
  if ( meta$no_hidden ) {
    
    # (ERROR) no learn rate for output layer
    if ( length(learn.rates) != 1 ) {
      stop('no hidden layers, length of learn.rates should be one', call. = FALSE)
    }
    
    # Set learn rate for input layer (0, not used) and output layer
    rates <- c(0, learn.rates)
    
  } else {
    
    # Single learn.rate specified by user
    if ( length(learn.rates) != meta$n_hidden + 1 ) {
      
      # (ERROR) length of learn.rates should be either 1 (broadcasting) or
      #         equal to the number of hidden layers + 1
      if ( length(learn.rates) != 1 ) {
        stop('length of learn.rates should be one or equal to number of hidden layers + 1',
             call. = FALSE)
      }
      
      # Broadcast single activ.function over all layers
      rates <- c(0, rep(learn.rates, meta$n_hidden + 1))
    } else {
      
      # No broadcasting
      rates <- c(0, learn.rates)
    }
  }
  
  # Collect parameters in list
  return ( list(type = optim.type, learn_rates = rates, m = momentum, L1 = L1, L2 = L2) )
}

# Set and check loss parameter list
setLossParams <- function(loss.type, delta.huber, meta) {
  allowed_types <- c('log', 'squared', 'absolute', 'huber', 'pseudo-huber')
  
  # (ERROR) loss.type one of allowed types
  if ( !(loss.type %in% allowed_types) ) {
    stop('loss.type not one of ', paste(allowed_types, collapse = ', '), 
         call. = FALSE)
  }
  
  # (WARN) do not use log loss for regression
  if ( meta$regression && loss.type == 'log' ) {
    warning('loss.type "log" not recommended for regression', call. = FALSE)
  }
  
  # (WARN) use log loss for classification
  if ( !meta$regression && loss.type != 'log' ) {
    warning('loss.type "log" recommended for classification', call. = FALSE)
  }
  
  # Checks on parameters specific to huber and pseudo-huber
  if ( loss.type %in% c('huber', 'pseudo-huber') ) {
    
    # (ERROR) delta should be non-negative
    if ( delta.huber < 0 ) {
      stop('delta.huber smaller than zero', call. = FALSE)
    }
    
    # (WARN) use absolute loss if delta equals zero
    if ( delta.huber == 0 ) {
      warning ('delta.huber equal to zero, this is equivalent to using absolute loss but less efficient', 
               call. = FALSE)
    }
  }
  
  # Collect parameters in list
  return ( list(type = loss.type, delta_huber = delta.huber) )
}

# Set training parameters
setTrainParams <- function (n.epochs, batch.size, val.prop, drop.last, verbose, meta) {
  
  # (ERROR) n.epochs not positive
  if ( n.epochs <= 0 ) {
    stop('n.epochs should be larger than zero', call. = FALSE)
  }
  
  # (ERROR) n.epochs not whole number
  if ( n.epochs %% 1 != 0 ) {
    stop('n.epochs not an integer', call. = FALSE)
  }
  
  # (ERROR) batch.size not positive
  if ( batch.size <= 0 ) {
    stop('batch.size should be larger than zero', call. = FALSE)
  }
  
  # (ERROR) batch.size not whole number
  if ( batch.size %% 1 != 0 ) {
    stop('batch.size not an integer', call. = FALSE)
  }
  
  # (ERROR) val.prop incorrect value
  if ( val.prop < 0 || val.prop >= 1) {
    stop('val.prop should be smaller than one and larger than or equal to zero',
         call. = FALSE)
  }
  
  # Size of training and validation sets
  n_train <- ceiling( meta$n_obs * (1 - val.prop) )
  n_val   <- meta$n_obs - n_train
  
  # (ERROR) batch.size larger than n_obs
  if ( batch.size > meta$n_obs ) {
    stop('batch.size larger than total number of observations in X and Y', 
         call. = FALSE)
  }
  
  # (ERROR) batch.size larger than n_train
  if ( n_val > 0 && batch.size > n_train ) {
    stop('batch.size larger than size of training data (after training/validation subsetting)',
         call. = FALSE)
  }
  
  # (WARN) small validation set
  if ( val.prop > 0 && n_val < 25 ) {
    warning('small validation set, only ', n_val, ' observations', call. = FALSE)
  }
  
  # Collect parameters in list
  return ( list(n_epochs = n.epochs, batch_size = batch.size, val_prop = val.prop, 
                drop_last = drop.last, verbose = verbose) )
}

