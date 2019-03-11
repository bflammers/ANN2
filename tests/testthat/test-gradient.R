# context("Full gradient check")

library(ANN2)

NN <- neuralnetwork(X = iris[,1:4], 
                    y = iris$Species, 
                    hidden.layers = c(10,10), 
                    val.prop = 0, 
                    verbose = FALSE)
meta <- NN$Rcpp_ANN$getMeta()

delta <- 1e-5
data <- iris[sample(nrow(iris), 5),]
X <- as.matrix(data[,1:4])
y <- 1 * outer(data$Species, meta$y_names, '==')

n_rows <- nrow(X)
n_cols <- ncol(X)
x_dm <- x_dp <- X
idx_vec <- sample(n_cols, n_rows, replace = TRUE)

for (i in 1:n_rows) {
  idx <- idx_vec[i]
  x_dp[i,idx] <- x_dp[i,idx] + delta
  x_dm[i,idx] <- x_dm[i,idx] - delta
}

eval_loss <- function(X, y) {
  y_fit <- NN$Rcpp_ANN$forwardPass(X)
  return( NN$Rcpp_ANN$evalLoss(y, y_fit) )
}

l_dp <- eval_loss(x_dp, y)
l_dm <- eval_loss(x_dm, y)

cE <- rowSums( (l_dp - l_dm) / (2*delta) )

y_fit <- NN$Rcpp_ANN$forwardPass(X)

E <- NN$Rcpp_ANN$backwardPass(y, y_fit)
tE <- sapply(1:n_rows, function(i) E[i, idx_vec[i]])




