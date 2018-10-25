\name{ANN2-package}
\alias{ANN2-package}
\alias{ANN2}
\docType{package}
\title{
\packageTitle{ANN2}
}
\description{
\packageDescription{ANN2}
}
\details{
Package for training neural networks. Special options for detecting and plotting 
anomalies using autoencoding neural networks. 
}
\author{
\packageAuthor{ANN2}

Maintainer: \packageMaintainer{ANN2}
}
\references{
Add links to references. Efficient Backprop Le Cun
}
\keyword{ package }
\examples{
# Example on iris dataset:

# Plot full data
plot(iris, pch = as.numeric(iris$Species))

# Prepare test and train sets
random_draw <- sample(1:nrow(iris), size = 100)
X_train     <- iris[random_draw, 1:4]
Y_train     <- iris[random_draw, 5]
X_test      <- iris[setdiff(1:nrow(iris), randDraw), 1:4]
Y_test      <- iris[setdiff(1:nrow(iris), randDraw), 5]

# Train neural network on classification task
NN <- neuralnetwork(X = X_train, Y = Y_train, hidden.layers = c(5, 5),
                    optim.type = 'adam', learn.rates = 0.01, val.prop = 0)

# Plot the loss during training
plot(NN)

# Make predictions
Y_pred <- predict(NN, newdata = X_test)

# Plot predictions
plot(X_test, pch = as.numeric(Y_test), col = (Y_test == Y_pred$predictions) + 2)
}