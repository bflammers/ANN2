library(ANN2)

#### NEURALNETWORK

# Prepare test and train sets
random_idx <- sample(1:nrow(iris), size = 145)
X_train    <- iris[random_idx, 1:4]
y_train    <- iris[random_idx, 5]
X_test     <- iris[setdiff(1:nrow(iris), random_idx), 1:4]
y_test     <- iris[setdiff(1:nrow(iris), random_idx), 5]

# Train neural network on classification task
NN <- neuralnetwork(X = X_train, 
                    y = y_train, 
                    hidden.layers = c(5, 5), 
                    optim.type = 'adam', 
                    n.epochs = 5000)

# Predict the class for new data points
predict(NN, X_test)

# Plot the training and validation loss
plot(NN)

#### AUTOENCODER

# Prepare test and train sets
random_idx <- sample(1:nrow(USArrests), size = 45)
X_train    <- USArrests[random_idx,]
X_test     <- USArrests[setdiff(1:nrow(USArrests), random_idx),]

# Define and train autoencoder
AE <- autoencoder(X = X_train, 
                  hidden.layers = c(10,3,10), 
                  loss.type = 'pseudo-huber',
                  optim.type = 'adam',
                  n.epochs = 5000)

# Plot original points (grey) and reconstructions (colored)
reconstruction_plot(AE, X_train)

# Reconstruct test data
reconstruct(AE, X_test)

