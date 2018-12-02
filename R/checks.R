#' @title Sets list with meta info, to be used in other checks
#' @description 
#' Set network meta info
#' @keywords internal
setMeta <- function(data, hidden.layers, regression) {
  
  # Check if network has no hidden layers, set n_hidden layers accordingly
  if ( is.null(hidden.layers) || all(is.na(hidden.layers)) ) {
    no_hidden <- TRUE
    n_hidden  <- 0
  } else {
    no_hidden <- FALSE
    n_hidden  <- length(hidden.layers)
  }
  
  # Set number of nodes in input and output layer
  n_in  <- ncol(data$X)
  n_out <- ncol(data$Y)
  
  # Set number of observations
  n_obs <- nrow(data$X)
  
  return( list(no_hidden = no_hidden, n_hidden = n_hidden, n_in = n_in, 
               n_out = n_out, n_obs = n_obs, regression = regression, 
               classes = data$classes, names = data$names, 
               hidden_layers = hidden.layers) )
}

#' @title Check the input data
#' @description 
#' Set and check data
#' @keywords internal
setData <- function(X, Y, regression, y_names = NULL) {
  
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
    
    # Set names to class names (used in predict.ANN() )
    y_names <- colnames(Y)
    
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
    
    # Convert Y to matrix
    Y <- as.matrix(Y)
    
    # If y_names NULL (first training run) then y_names should be derived
    # from the data. If not NULL, y_names of first training run are used
    if ( is.null(y_names) ) {
      y_names <- sort(unique(Y))
    }
    
    # One-hot encode Y and store classes
    Y <- 1 * outer(c(Y), y_names, '==')
    
  }
  
  # Set number of observations
  n_obs <- nrow(X)
  
  # (ERROR) not same number of observations in X and Y
  if ( n_obs != nrow(Y) ) {
    stop('Unequal numbers of observations in X and Y', call. = FALSE)
  }
  
  # Collect parameters in list
  return ( list(X = X, Y = Y, y_names = y_names, n_obs = n_obs) )
}

#' @title Check user input related to network structure
#' @description 
#' Set and check network parameter list
#' @keywords internal
setNetworkParams <- function(hidden.layers, standardize, verbose, meta) {
  
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
  return ( list(num_nodes = num_nodes, stand_X = stand_X, stand_Y = stand_Y, 
                verbose = verbose, regression = meta$regression) )
}

#' @title Check user input related to activation functions
#' @description 
#' Set and check activation parameter list
#' @keywords internal
setActivParams <- function(activ.functions, step.H, step.k, meta) {
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
      
      # Broadcasting or throw error
      if ( length(activ.functions) == 1 ) {
        
        # Broadcast single activ.function over all hidden layers
        types <- c('input', rep(activ.functions, meta$n_hidden), output_type)
        
      } else {
        
        # (ERROR) length of activ.functions either 1 (broadcasting) or equal to the 
        #         number of hidden layers
        stop('length of activ.functions should be either one (for broadcasting) or equal to number of hidden layers', 
             call. = FALSE)
      }
      
    } else {
      
      # No broadcasting
      types  <- c('input', activ.functions, output_type)
    }
  }
  
  # Check parameters H and k, only for step activation function
  if ( 'step' %in% activ.functions ) {
    
    # (ERROR) number of steps H should be a whole number larger than zero
    if ( step.H %% 1 != 0  || step.H < 1 ) {
      stop('step.H should be an integer larger than zero', call. = FALSE)
    }
    
    # (ERROR) smoothing parameter k should be and integer larger than zero
    if ( step.k %% 1 != 0  || step.k < 1 ) {
      stop('step.k should be an integer larger than zero', call. = FALSE)
    }
  }
  
  # Collect parameters in list
  return ( list(types = types, step_H = step.H, step_k = step.k) )
}

#' @title Check user input related to optimizer
#' @description 
#' Set and check optimizer parameter list
#' @keywords internal
setOptimParams <- function(optim.type, learn.rates, L1, L2, sgd.momentum, 
                           rmsprop.decay, adam.beta1, adam.beta2, meta) {
  # Allowed optimizer types
  allowed_types <- c('sgd', 'rmsprop', 'adam')
  
  # (ERROR) optim.type not of allowed type
  if ( !(optim.type %in% allowed_types) ) {
    stop('optim.type not one of types ', paste(allowed_types, collapse = ', '))
  }
  
  # (ERROR) missing learn.rates
  if ( is.null(learn.rates) || any(is.na(learn.rates)) ) {
    stop('missing value in learn.rates or learn.rates is NULL', call. = FALSE)
  } 
  
  # (ERROR) learn.rate incorrect value
  if ( any(learn.rates <= 0) ) {
    stop('learn rates should be larger than 0', call. = FALSE)
  }
  
  # (ERROR) L1 and/or L2 incorrect value(s)
  if ( L1 < 0 || L2 < 0 ) {
    stop('L1 and/or L2 negative', call. = FALSE)
  }
  
  # Checks specific to SGD optimizer
  if ( optim.type == 'sgd' ) {
   
    # (ERROR) momentum incorrect value
    if ( sgd.momentum >= 1 || sgd.momentum < 0 ) {
      stop('sgd.momentum larger than or equal to one or smaller than zero', call. = FALSE)
    }
  
  # Checks specific to RMSprop optimizer  
  } else if ( optim.type == 'rmsprop' ) {
    
    # (ERROR) rmsprop.decay incorrect value
    if ( rmsprop.decay >= 1 || rmsprop.decay <= 0 ) {
      stop('rmsprop.decay should be between zero and one', call. = FALSE)
    }
    
  # Checks specific to ADAM optimizer
  } else if ( optim.type == 'adam' ) {
    
    # (ERROR) adam.beta1 incorrect value
    if ( adam.beta1 >= 1 || adam.beta1 <= 0 ) {
      stop('adam.beta1 should be between zero and one', call. = FALSE)
    }
    
    # (ERROR) adam.beta2 incorrect value
    if ( adam.beta1 >= 2 || adam.beta2 <= 0 ) {
      stop('adam.beta2 should be between zero and one', call. = FALSE)
    }
    
  } else {
    # Not implemented optimizer
    stop('optim.type not supported')
  }
  
  # Length of learn.rates does not mach number of optimizers
  if ( length(learn.rates) != meta$n_hidden + 1 ) {
    
    # Broadcasting or throw error
    if ( length(learn.rates) == 1 ) {
      
      # Broadcast single activ.function over all layers
      rates <- c(0, rep(learn.rates, meta$n_hidden + 1))
      
    } else {
      
      # (ERROR) length of learn.rates should be either 1 (broadcasting) or
      #         equal to the number of hidden layers + 1
      stop('length of learn.rates should be one or equal to number of hidden layers + 1',
           call. = FALSE)
    }
      
  } else {
      
    # No broadcasting
    rates <- c(0, learn.rates)
  
  }
  
  # Collect parameters in list
  return ( list(type = optim.type, learn_rates = rates, L1 = L1, L2 = L2, 
                sgd_momentum = sgd.momentum, rmsprop_decay = rmsprop.decay, 
                adam_beta1 = adam.beta1, adam_beta2 = adam.beta2) )
}

#' @title Check user input related to loss function
#' @description 
#' Set and check loss parameter list
#' @keywords internal
setLossParams <- function(loss.type, huber.delta, meta) {
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
    if ( huber.delta < 0 ) {
      stop('huber.delta smaller than zero', call. = FALSE)
    }
    
    # (WARN) use absolute loss if delta equals zero
    if ( huber.delta == 0 ) {
      warning ('huber.delta equal to zero, this is equivalent to using absolute loss but less efficient', 
               call. = FALSE)
    }
  }
  
  # Collect parameters in list
  return ( list(type = loss.type, huber_delta = huber.delta) )
}

#' @title Check user input related to training
#' @description 
#' Set training parameters
#' @keywords internal
setTrainParams <- function (n.epochs, batch.size, val.prop, drop.last, data) {
  
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
  n_train <- ceiling( data$n_obs * (1 - val.prop) )
  n_val   <- data$n_obs - n_train
  
  # (ERROR) batch.size larger than n_obs
  if ( batch.size > data$n_obs ) {
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
                drop_last = drop.last) )
}

