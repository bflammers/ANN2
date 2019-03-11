context("Full gradient check")

library(ANN2)

calcGradients <- function(X, y, regression, ...) {
  
  eval_loss <- function(X, y) {
    y_fit <- NN$Rcpp_ANN$forwardPass(X)
    return( NN$Rcpp_ANN$evalLoss(y, y_fit) )
  }
  
  delta <- 1e-5
  n_rows <- nrow(X)
  n_cols <- ncol(X)
  
  NN <- neuralnetwork(X = X, 
                      y = y, 
                      standardize = TRUE,
                      regression = regression,
                      batch.size = 5,
                      hidden.layers = c(5), 
                      val.prop = 0, 
                      verbose = FALSE, ...)
  
  if (regression) {
    y <- as.matrix(y)
    y <- NN$Rcpp_ANN$scale_y(y)
  } else {
    y <- 1 * outer(y, NN$Rcpp_ANN$getMeta()$y_names, '==')
  }
  
  X <- as.matrix(X)
  X <- NN$Rcpp_ANN$scale_X(X)
  x_dm <- x_dp <- X
  idx_vec <- sample(n_cols, n_rows, replace = TRUE)
  
  for (i in 1:n_rows) {
    idx <- idx_vec[i]
    x_dp[i,idx] <- x_dp[i,idx] + delta
    x_dm[i,idx] <- x_dm[i,idx] - delta
  }
  
  # Numerical gradient
  E_num <- rowSums( (eval_loss(x_dp, y) - eval_loss(x_dm, y)) / (2*delta) )
  
  # Analytical gradient
  y_fit <- NN$Rcpp_ANN$forwardPass(X)
  E <- NN$Rcpp_ANN$backwardPass(y, y_fit)
  E_ana <- sapply(1:n_rows, function(i) E[i, idx_vec[i]])
  
  return( list(numeric = E_num, analytic = E_ana))
}

relCompare <- function(observed, expected, tolerance) {
  max_elem <- pmax(abs(expected), abs(observed), 1e-3)
  rel_diff <- abs(observed - expected) / max_elem
  return ( all(rel_diff < tolerance) )
}
# 
# test_that("the full gradient is correct for all activations for regression", 
# {
#   data <- iris[sample(nrow(iris), size = 4),  1:4]
#   rel_tol <- 1e-2
#   gradCheck <- function(activ.functions) {
#     grads <- calcGradients(X = data[,1:3], y = data[,4], 
#                            activ.functions = activ.functions, regression = TRUE, 
#                            loss.type = 'squared')
#     return( relCompare(grads$analytic, grads$numeric, rel_tol) )
#     
#   }
#   expect_true(gradCheck("tanh"))
#   expect_true(gradCheck("sigmoid"))
#   expect_true(gradCheck("ramp"))
#   expect_true(gradCheck("linear"))
#   expect_true(gradCheck("step"))
#   expect_true(gradCheck("relu"))
# })
# 
# test_that("the full gradient is correct for all activations for classification", 
# {
#   data <- scale(iris[1:10,1:4])
#   rel_tol <- 1e-2
#   gradCheck <- function(activ.functions) {
#     grads <- calcGradients(X = data[,1:3], y = data[,4], 
#                            activ.functions = activ.functions, regression = TRUE, 
#                            loss.type = 'squared')
#     return( relCompare(grads$analytic, grads$numeric, rel_tol) )
#     
#   }
#   expect_true(gradCheck("tanh"))
#   expect_true(gradCheck("sigmoid"))
#   expect_true(gradCheck("ramp"))
#   expect_true(gradCheck("linear"))
#   expect_true(gradCheck("step"))
#   expect_true(gradCheck("relu"))
# })






