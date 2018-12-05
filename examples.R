library(ANN2)

# Example on iris dataset:
# Prepare test and train sets
random_draw <- sample(1:nrow(iris), size = 100)
X_train     <- iris[random_draw, 1:4]
y_train     <- iris[random_draw, 5]
X_test      <- iris[setdiff(1:nrow(iris), random_draw), 1:4]
y_test      <- iris[setdiff(1:nrow(iris), random_draw), 5]

# Train neural network on classification task
NN <- neuralnetwork(X = X_train, y = y_train, hidden.layers = c(5, 5),
                    optim.type = 'adam', learn.rates = 0.01, val.prop = 0)

# Plot the loss during training
plot(NN)

# Make predictions
y_pred <- predict(NN, newdata = X_test)

# Plot predictions
correct <- (y_test == y_pred$predictions)
plot(X_test, pch = as.numeric(y_test), col = correct + 2)

################################################################################

# Autoencoder example 
# The step function in the compression layer induces a clustering of the 
# compressions and reconstructions
X <- USArrests
AE <- autoencoder(X, c(10,2,10), loss.type = 'pseudo-huber',
                  activ.functions = c('tanh','step','tanh'),
                  batch.size = 8, optim.type = 'sgd',
                  n.epochs = 100000, val.prop = 0)

# Plot loss during training
plot(AE)

# Make reconstruction and compression plots
reconstruction_plot(AE, X)
compression_plot(AE, X, jitter=TRUE)

# Reconstruct data and show states with highest anomaly scores
recX <- reconstruct(AE, X)
sort(recX$anomaly_scores, decreasing = TRUE)[1:5]










