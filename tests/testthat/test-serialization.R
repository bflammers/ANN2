
context("Serializing and deserializing")

test_that("the NN is correctly written to and read from disk",
{
  skip_on_os("mac")
  
  # Construct network
  wNN <- neuralnetwork(X = iris[,1:4], 
                       y = iris$Species, 
                       hidden.layers = 10, 
                       val.prop = 0.5,
                       verbose = FALSE)
  
  # Write object to disk and read again
  fp <- "./NN.ANN"
  write_ANN(wNN, fp)
  rNN <- read_ANN(fp)
  
  # Run tests
  expect_equal(rNN, wNN)
  expect_identical(rNN$Rcpp_ANN$getParams(), wNN$Rcpp_ANN$getParams())
  expect_identical(rNN$Rcpp_ANN$getMeta(), wNN$Rcpp_ANN$getMeta())
  expect_identical(rNN$Rcpp_ANN$getTrainHistory(), wNN$Rcpp_ANN$getTrainHistory())
  
  # Remove file from disk
  file.remove(fp)
})

