

context("Test checks on user input")

test_that("the checks for neuralnetwork() work correctly",
{
  
  nnCall <- function(...) {
    
    default_args <- list(
      'X' = matrix(runif(400), 100, 4),
      'y' = sample(0:1, size = 100, replace = TRUE),
      'hidden.layers' = c(10, 10),
      'val.prop' = 0,
      'verbose' = FALSE
    )
    
    args <- list(...)
    
    for (arg_name in names(default_args)) {
      if ( !(arg_name %in% names(args)) ) {
        args[[arg_name]] <- default_args[[arg_name]]
      }
    }
    
    do.call("neuralnetwork", args)
    
  }
  
  # Run tests
  expect_error( nnCall(regression = FALSE, loss.type = "squared") )
  expect_error( nnCall(regression = TRUE, loss.type = "log") )
  expect_error( nnCall(n.epochs = 0) )
  expect_error( nnCall(n.epochs = -1) )
  expect_error( nnCall(batch.size = 101) )
  expect_error( nnCall(val.prop = -1) )
  expect_error( nnCall(val.prop = 1) )
  expect_error( nnCall(learn.rates = -1) )
  expect_error( nnCall(learn.rates = 0) )
  expect_error( nnCall(learn.rates = c(0.1, 0.1, 0)) )
  expect_error( nnCall(learn.rates = c(0.1, 0.1, 0.1, 0.1)) )
  expect_error( nnCall(learn.rates = c(0.1, 0.1)) )
  expect_error( nnCall(activ.functions = c('tanh', 'tanh', 'tanh')) )
  expect_error( nnCall(activ.functions = 'not_supported') )
  expect_error( nnCall(optim.type = 'not_supported') )
  expect_error( nnCall(X = iris[1:100,1:4], y = iris[1:101, 5]) )
  expect_error( nnCall(X = iris[,1:4], y = iris[, 5], regression = TRUE) )
  expect_error( nnCall(L1 = -1) )
  expect_error( nnCall(L2 = -1) )
  expect_error( nnCall(sgd.momentum = 1) )
  expect_error( nnCall(sgd.momentum = -1) )
  expect_error( nnCall(regression = TRUE, loss.type = 'huber', huber.delta = -1) )
  expect_error( nnCall(regression = TRUE, loss.type = 'pseudo-huber', huber.delta = -1) )
  expect_error( nnCall(activ.functions = 'step', step.H = -1) )
  expect_error( nnCall(activ.functions = 'step', step.k = -1) )
  expect_error( nnCall(optim.type = 'rmsprop', rmsprop.decay = -1) )
  expect_error( nnCall(optim.type = 'rmsprop', rmsprop.decay = 0) )
  expect_error( nnCall(optim.type = 'rmsprop', rmsprop.decay = 1) )
  expect_error( nnCall(optim.type = 'adam', adam.beta1 = -1) )
  expect_error( nnCall(optim.type = 'adam', adam.beta1 = 0) )
  expect_error( nnCall(optim.type = 'adam', adam.beta1 = 1) )
  expect_error( nnCall(optim.type = 'adam', adam.beta2 = -1) )
  expect_error( nnCall(optim.type = 'adam', adam.beta2 = 0) )
  expect_error( nnCall(optim.type = 'adam', adam.beta2 = 1) )
  
})

test_that("the checks for autoencoder() work correctly",
{
  
  aeCall <- function(...) {
    
    default_args <- list(
      'X' = matrix(runif(400), 100, 4),
      'hidden.layers' = c(10, 2, 10),
      'val.prop' = 0,
      'verbose' = FALSE
    )
    
    args <- list(...)
    
    for (arg_name in names(default_args)) {
      if ( !(arg_name %in% names(args)) ) {
        args[[arg_name]] <- default_args[[arg_name]]
      }
    }
    
    do.call("autoencoder", args)
    
  }
  
  # Run tests
  expect_error( aeCall(regression = FALSE) )
  expect_error( aeCall(loss.type = "log") )
  expect_error( aeCall(X = iris[,1:5]) )
  
})

