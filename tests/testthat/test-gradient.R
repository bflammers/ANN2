
# See http://cs231n.github.io/neural-networks-3/#gradcheck

# Function to calculate numerical and analytical gradient
calcGradients <- function(X, y, regression, ...) {
  
  # Set epsilon (for calculating numerical gradient), determine dimensions
  epsilon <- 1e-5
  n_rows <- nrow(X)
  n_cols <- ncol(X)
  
  # Construct NN
  NN <- neuralnetwork(X = X, 
                      y = y, 
                      standardize = TRUE,
                      regression = regression,
                      batch.size = n_rows,
                      val.prop = 0, 
                      n.epochs = 100,
                      verbose = FALSE, 
                      step.k = 30, ...)
  
  # Prepare targets for regression/classification
  if (regression) {
    y <- as.matrix(y)
  } else {
    y <- 1 * outer(y, NN$Rcpp_ANN$getMeta()$y_names, '==')
  }
  y <- NN$Rcpp_ANN$scale_y(y, FALSE)
  
  # Prepare features - same for regression and classification
  X <- as.matrix(X)
  X <- NN$Rcpp_ANN$scale_X(X, FALSE)
  
  #### NUMERICAL GRADIENT
  # Construct feature matrices with small perturbations (positive and negative) 
  # at one of the elements per row. For this element the gradients compared
  X_eps_neg <- X_eps_pos <- X
  idx_vec <- sample(n_cols, n_rows, replace = TRUE)
  
  for (i in 1:n_rows) {
    idx <- idx_vec[i]
    X_eps_pos[i,idx] <- X_eps_pos[i,idx] + epsilon
    X_eps_neg[i,idx] <- X_eps_neg[i,idx] - epsilon
  }
  
  # Function to evaluate loss for a feature matrix X and targets y
  eval_loss <- function(X, y) {
    y_fit <- NN$Rcpp_ANN$forwardPass(X)
    return( NN$Rcpp_ANN$evalLoss(y, y_fit) )
  }
  
  # Calculate numerical gradient
  E_num <- eval_loss(X_eps_pos, y) - eval_loss(X_eps_neg, y)
  E_num <- E_num / (2*epsilon)
  E_num <- rowSums(E_num)
  
  #### ANALYTICAL GRADIENT
  y_fit <- NN$Rcpp_ANN$forwardPass(X)
  E <- NN$Rcpp_ANN$backwardPass(y, y_fit)
  E_ana <- sapply(1:n_rows, function(i) E[i, idx_vec[i]])
  
  return ( list(numeric = E_num, analytic = E_ana) )
}

# Function to check equality between observed and expected by inspecting their 
# distance to one another relative to the largest absolute value of the two
relCompare <- function(observed, expected, tolerance) {
  
  # Elementwise maximum of absolute values
  max_elem <- pmax(abs(expected), abs(observed))
  
  # Absolute relative difference
  rel_diff <- abs(observed - expected) / max_elem
  
  # Check equality given tolerance
  check_vec <- (rel_diff < tolerance)
  
  # Check for edge case: both gradient elements zero
  check_vec[max_elem < 1e-8] <- TRUE # Both element close to zero
  
  # Pass check if all numerical grads are equal to their analytical counterparts
  return ( all(check_vec) )
}

context("Full gradient check for regression")

## ACTIVATION FUNCTIONS
test_that("the full gradient is correct for all loss functions",
{
  skip_on_cran()
  
  # Set data and relative tolerance for equality check
  data <- iris[sample(nrow(iris), size = 4),  1:4]
  rel_tol_smooth <- 1e-7
  rel_tol_kinks <- 1e-4
  
  # Function to check gradient for current cases
  gradCheck <- function(loss.type, rel_tol) {
    grads <- calcGradients(X = data[,1:3], 
                           y = data[,4],
                           hidden.layers = c(5,5),
                           activ.functions = "tanh", 
                           regression = TRUE,
                           loss.type = loss.type)
    return( relCompare(grads$analytic, grads$numeric, rel_tol) )

  }
  
  # Run tests on gradient
  # The gradient checks sometimes fail by chance (especially with functions 
  # that contain kinks such as the ramp, step and relu) so do not run on CRAN
  expect_true(gradCheck("squared",      rel_tol_smooth))
  expect_true(gradCheck("absolute",     rel_tol_kinks))
  expect_true(gradCheck("huber",        rel_tol_kinks))
  expect_true(gradCheck("pseudo-huber", rel_tol_kinks))
  
})

## ACTIVATION FUNCTIONS
test_that("the full gradient is correct for all activation functions",
{
  skip_on_cran()
  
  # Set data and relative tolerance for equality check
  data <- iris[sample(nrow(iris), size = 4),  1:4]
  rel_tol_smooth <- 1e-7
  rel_tol_kinks <- 1e-4
  
  # Function to check gradient for current cases
  gradCheck <- function(activ.functions, rel_tol) {
    grads <- calcGradients(X = data[,1:3], 
                           y = data[,4],
                           hidden.layers = c(5,5),
                           activ.functions = activ.functions, 
                           regression = TRUE,
                           loss.type = 'squared')
    return( relCompare(grads$analytic, grads$numeric, rel_tol) )
    
  }
  
  # Run tests on gradient
  # The gradient checks sometimes fail by chance (especially with functions 
  # that contain kinks such as the ramp, step and relu) so do not run on CRAN
  expect_true(gradCheck("tanh",    rel_tol_smooth))
  expect_true(gradCheck("sigmoid", rel_tol_smooth))
  expect_true(gradCheck("linear",  rel_tol_smooth))
  expect_true(gradCheck("ramp",    rel_tol_kinks))
  expect_true(gradCheck("step",    rel_tol_kinks))
  expect_true(gradCheck("relu",    rel_tol_kinks))
  
})


context("Full gradient check for classification")

test_that("the full gradient is correct for all activation functions",
{
  skip_on_cran()
  
  # Set data and relative tolerance for equality check
  data <- iris[sample(nrow(iris), size = 4),]
  rel_tol_smooth <- 1e-7
  rel_tol_kinks <- 1e-4

  # Function to check gradient for current cases
  gradCheck <- function(activ.functions, rel_tol) {
    grads <- calcGradients(X = data[,1:4],
                           y = data[,5],
                           hidden.layers = c(5, 5),
                           activ.functions = activ.functions,
                           regression = FALSE,
                           loss.type = 'log')
    return( relCompare(grads$analytic, grads$numeric, rel_tol) )

  }

  # Run tests on gradients
  # The gradient checks sometimes fail by chance (especially with functions 
  # that contain kinks such as the ramp, step and relu) so do not run on CRAN
  expect_true(gradCheck("tanh",    rel_tol_smooth))
  expect_true(gradCheck("sigmoid", rel_tol_smooth))
  expect_true(gradCheck("linear",  rel_tol_smooth))
  expect_true(gradCheck("ramp",    rel_tol_kinks))
  expect_true(gradCheck("step",    rel_tol_kinks))
  expect_true(gradCheck("relu",    rel_tol_kinks))
  
})


