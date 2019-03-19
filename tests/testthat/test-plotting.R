
context("Plotting")

n_epochs <- 100
X <- iris[,1:4]
y <- iris$Species
NN_dims <- c(10,10)
AE_dims <- c(10,2,10)

NN <- neuralnetwork(X = X,
                    y = y, 
                    hidden.layers = NN_dims, 
                    n.epochs = n_epochs,
                    val.prop = 0.5, 
                    verbose = FALSE)

AE <- autoencoder(X = X, 
                  hidden.layers = AE_dims, 
                  n.epochs = n_epochs,
                  val.prop = 0.5, 
                  verbose = FALSE)

test_that("the plot.ANN() function works correctly",
{
  p_NN <- plot(NN) 
  p_AE <- plot(AE) 
  
  expect_s3_class(p_NN, 'gg')
  expect_s3_class(p_NN, 'ggplot')
  expect_true(is.ggplot(p_NN))
  expect_equal(p_NN$labels$x, 'Epoch')
  expect_equal(p_NN$labels$y, 'Loss')
  expect_equal(p_NN$labels$colour, 'variable')
  expect_equal(levels(p_NN$data$variable), c('Training', 'Validation'))
  
  expect_s3_class(p_AE, 'gg')
  expect_s3_class(p_AE, 'ggplot')
  expect_true(is.ggplot(p_AE))
  expect_equal(p_AE$labels$x, 'Epoch')
  expect_equal(p_AE$labels$y, 'Loss')
  expect_equal(p_AE$labels$colour, 'variable')
  expect_equal(levels(p_AE$data$variable), c('Training', 'Validation'))
  
})

test_that("the reconstruction_plot.ANN() function works correctly",
{
  expect_error(reconstruction_plot(NN, X = X) )
  expect_error(reconstruction_plot(AE) )
  
  p_AE <- reconstruction_plot(AE, X = X) 
  
  expect_s3_class(p_AE, 'gg')
  expect_s3_class(p_AE, 'ggplot')
  expect_true(is.ggplot(p_AE))
  expect_null(p_AE$labels$x)
  expect_null(p_AE$labels$y)
  expect_equal(p_AE$labels$group, 'obs')
  expect_equal(p_AE$labels$colour, 'col')
  expect_equal(levels(p_AE$data$x_dim), sort(colnames(X)))
  expect_equal(levels(p_AE$data$y_dim), sort(colnames(X)))
  
})

test_that("the compression_plot.ANN() function works correctly",
{
  expect_error(compression_plot(AE) )
  
  p_AE <- compression_plot(AE, X = X) 
  
  expect_s3_class(p_AE, 'gg')
  expect_s3_class(p_AE, 'ggplot')
  expect_true(is.ggplot(p_AE))
  expect_null(p_AE$labels$x)
  expect_null(p_AE$labels$y)
  expect_null(p_AE$labels$group)
  expect_equal(p_AE$labels$colour, 'col')
  expect_equal(levels(p_AE$data$x_dim), paste0('node_', 1:AE_dims[2]))
  expect_equal(levels(p_AE$data$y_dim), paste0('node_', 1:AE_dims[2]))
  
})
