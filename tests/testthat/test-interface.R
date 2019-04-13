
context("User interface")

# Dimensions
n_rows <- 100
n_cols_X <- 4
n_cols_y <- 3
X_names <- c('aa', 'bb', 'cc', 'dd')
y_names <- c('a', 'b', 'c')
NN_dims <- c(10, 10)
AE_dims <- c(10, 2, 10)
n_elem_X <- n_cols_X * n_rows
n_elem_y <- n_cols_y * n_rows

# Generate random data
X <- matrix(runif(n = n_elem_X), nrow = n_rows, ncol = n_cols_X)
colnames(X) <- X_names
y_regr <- matrix(runif(n = n_elem_y), nrow = n_rows, ncol = n_cols_y)
colnames(y_regr) <- y_names
y_class <- sample(y_names, size = n_rows, replace = TRUE)

# Construct ANN2 objects
NN_regr <- neuralnetwork(X = X, 
                         y = y_regr,
                         hidden.layers = NN_dims,
                         regression = TRUE,
                         loss.type = 'squared',
                         val.prop = 0,
                         verbose = FALSE)

NN_class <- neuralnetwork(X = X, 
                          y = y_class,
                          hidden.layers = NN_dims,
                          val.prop = 0,
                          verbose = FALSE)

AE <- autoencoder(X = X, 
                  hidden.layers = AE_dims,
                  val.prop = 0,
                  verbose = FALSE)


test_that("the neuralnetwork() function works correctly",
{
  expect_s3_class(NN_regr, 'ANN')
  expect_s3_class(NN_class, 'ANN')
  expect_s4_class(NN_regr$Rcpp_ANN, 'Rcpp_ANN')
  expect_s4_class(NN_class$Rcpp_ANN, 'Rcpp_ANN')
  
  # Test meta data for regression
  meta <- NN_regr$Rcpp_ANN$getMeta()
  expect_false( meta$no_hidden )
  expect_false( meta$autoencoder )
  expect_true( meta$regression )
  expect_equal( meta$n_hidden, length(NN_dims) )
  expect_equal( meta$n_in, n_cols_X )
  expect_equal( meta$n_out, n_cols_y )
  expect_equal( meta$y_names, y_names )
  expect_equal( meta$num_nodes, c(n_cols_X, NN_dims, n_cols_y) )
  
  # Test meta data for classification
  meta <- NN_class$Rcpp_ANN$getMeta()
  expect_false( meta$no_hidden )
  expect_false( meta$autoencoder )
  expect_false( meta$regression )
  expect_equal( meta$n_hidden, length(NN_dims) )
  expect_equal( meta$n_in, n_cols_X )
  expect_equal( meta$n_out, n_cols_y )
  expect_equal( meta$y_names, y_names )
  expect_equal( meta$num_nodes, c(n_cols_X, NN_dims, n_cols_y) )
  
  # Test set.seed
  NN_class_1 <- neuralnetwork(X = X, 
                              y = y_class,
                              hidden.layers = NN_dims,
                              val.prop = 0,
                              verbose = FALSE)
  NN_class_2 <- neuralnetwork(X = X, 
                              y = y_class,
                              hidden.layers = NN_dims,
                              val.prop = 0,
                              verbose = FALSE, 
                              random.seed = 42)
  NN_class_3 <- neuralnetwork(X = X, 
                              y = y_class,
                              hidden.layers = NN_dims,
                              val.prop = 0,
                              verbose = FALSE, 
                              random.seed = 42)
  p1 <- NN_class_1$Rcpp_ANN$getParams()
  p2 <- NN_class_2$Rcpp_ANN$getParams()
  p3 <- NN_class_3$Rcpp_ANN$getParams()
  expect_false(isTRUE(all.equal(p1, p2)))
  expect_false(isTRUE(all.equal(p1, p3)))
  expect_identical(p2, p3)
})

test_that("the autoencoder() function works correctly",
{
  expect_s3_class(AE, 'ANN')
  expect_s4_class(AE$Rcpp_ANN, 'Rcpp_ANN')
  
  # Test meta data
  meta <- AE$Rcpp_ANN$getMeta()
  expect_false( meta$no_hidden )
  expect_true( meta$autoencoder )
  expect_true( meta$regression )
  expect_equal( meta$n_hidden, length(AE_dims) )
  expect_equal( meta$n_in, n_cols_X )
  expect_equal( meta$n_out, n_cols_X )
  expect_equal( meta$y_names, X_names )
  expect_equal( meta$num_nodes, c(n_cols_X, AE_dims, n_cols_X) )
  
})
  
test_that("the predict() function works correctly",
{
  y_regr_pred <- predict(NN_regr, X)
  y_class_pred <- predict(NN_class, X)
  ae_pred <- predict(AE, X)
  
  # Check predictions for regressions
  expect_identical(names(y_regr_pred), c('predictions'))
  expect_identical(colnames(y_regr_pred$predictions), y_names)
  expect_equal(NROW(y_regr_pred$predictions), n_rows)
  expect_equal(NCOL(y_regr_pred$predictions), n_cols_y)
  expect_true( !any(is.na(y_regr_pred$predictions)) )
  
  # Check predictions for classification
  expect_identical(names(y_class_pred), c('predictions', 'probabilities'))
  expect_identical(colnames(y_class_pred$probabilities), paste0('class_', y_names))
  expect_identical(sort(unique(y_class_pred$predictions)), y_names)
  expect_equal(NROW(y_class_pred$predictions), n_rows)
  expect_equal(NCOL(y_class_pred$predictions), 1)
  expect_equal(NROW(y_class_pred$probabilities), n_rows)
  expect_equal(NCOL(y_class_pred$probabilities), n_cols_y)
  expect_equal( rowSums(y_class_pred$probabilities), rep(1, n_rows))
  expect_true( !any(is.na(y_class_pred$predictions)) )
  expect_true( !any(is.na(y_class_pred$probabilities)) )
  
  # Check predictions for autoencoder
  expect_identical(names(ae_pred), c('predictions'))
  expect_identical(colnames(ae_pred$predictions), X_names)
  expect_equal(NROW(ae_pred$predictions), n_rows)
  expect_equal(NCOL(ae_pred$predictions), n_cols_X)
  expect_true( !any(is.na(ae_pred$predictions)) )
  
})

test_that("the train() function works correctly",
{
  n_epoch <- 25
  
  # Classification
  epoch_nn_class_before <- NN_class$Rcpp_ANN$getTrainHistory()$n_epoch
  train(NN_class, X, y_class, val.prop = 0, n.epochs = n_epoch)
  epoch_nn_class_after <- NN_class$Rcpp_ANN$getTrainHistory()$n_epoch
  expect_equal( epoch_nn_class_after - epoch_nn_class_before, n_epoch )
  
  # Regression
  epoch_nn_regr_before <- NN_regr$Rcpp_ANN$getTrainHistory()$n_epoch
  train(NN_regr, X, y_regr, val.prop = 0, n.epochs = n_epoch)
  epoch_nn_regr_after <- NN_regr$Rcpp_ANN$getTrainHistory()$n_epoch
  expect_equal( epoch_nn_regr_after - epoch_nn_regr_before, n_epoch )
  
  # Autoencoder
  epoch_ae_before <- AE$Rcpp_ANN$getTrainHistory()$n_epoch
  train(AE, X, val.prop = 0, n.epochs = n_epoch)
  epoch_ae_after <- AE$Rcpp_ANN$getTrainHistory()$n_epoch
  expect_equal( epoch_ae_after - epoch_ae_before, n_epoch )
  
})

test_that("the reconstruct() function works correctly",
{
  expect_error(reconstruct(NN_class, X))
  expect_error(reconstruct(NN_regr, X))
  
  X_recr <- reconstruct(AE, X)
  expect_equal( length(X_recr$anomaly_scores), n_rows )
  expect_equal( dim(X_recr$reconstructed), dim(X) )
  expect_equal( colnames(X_recr$reconstructed), X_names )
})

# Needed for encode() / decode() tests
X_enc <- encode(AE, X)
X_dec <- decode(AE, X_enc)

callAE <- function(hidden_layers) {
  AE_ambig <- autoencoder(X = X, 
                          hidden.layers = hidden_layers, 
                          verbose = FALSE, 
                          val.prop = 0)
}

test_that("the encode() function works correctly",
{
  expect_equal( dim(X_enc), c(n_rows, AE_dims[2]) )
  expect_equal( colnames(X_enc), paste0('node_', 1:AE_dims[2]) )
  expect_error( encode(AE, X[,1:3]) ) # Incorrect data dimensions
  
  # Check for error when compression layer is ambiguous
  expect_error( encode( callAE( c(3,3,3) ), X ) )
  expect_error( encode( callAE( c(3,2,2) ), X ) )
})

test_that("the decode() function works correctly",
{
  expect_equal( dim(X_dec), dim(X) )
  expect_equal( colnames(X_dec), X_names )
  expect_error( decode(AE, X_enc[,1]) ) # Incorrect data dimensions
  
  # Check for error when compression layer is ambiguous
  expect_error( decode( callAE( c(3,3,3) ), X_enc ) )
  expect_error( decode( callAE( c(3,2,2) ), X_enc ) )
  
  expect_identical( X_dec, reconstruct(AE, X)$reconstructed )
})
