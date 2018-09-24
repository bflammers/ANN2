
# Generate data for exampleANN
genrResponse <- function(type) {
  n <- 500; sd.noise <- 0.1
  # 'linear' 'polynomial' 'disjoint' 'multiclass' 'nested' 'surface'
  if (type == "linear") {
    X <- matrix(stats::runif(2 * n), ncol = 2, nrow = n)
    y <- rep(0, n)
    y[-2 * X[, 1] + 1 > (X[, 2] + stats::rnorm(nrow(X), sd = sd.noise))] <- 1
    y[X[, 1] > (X[, 2] + stats::rnorm(nrow(X), sd = sd.noise))] <- 2
  } else if (type == "polynomial") {
    X <- matrix(stats::runif(2 * n, -3, 7), ncol = 2, nrow = n)
    y <- 2/3 * X[, 1]^2 - 1/9 * X[, 1]^3 > (X[, 2] + 
                                              stats::rnorm(nrow(X), sd = sd.noise))
  } else if (type == "nested") {
    X <- matrix(stats::runif(2 * n, -4, 8), ncol = 2, nrow = n)
    loc <- c(2, 2)
    y <- rep(0, nrow(X))
    y[(X[, 1] - loc[1])^2 + (X[, 2] - loc[2])^2 < 17 + 
        stats::rnorm(nrow(X), sd = sd.noise)] <- 1
    y[(X[, 1] - loc[1])^2 + (X[, 2] - loc[2])^2 < 5 + 
        stats::rnorm(nrow(X), sd = sd.noise)] <- 2
  } else if (type == "disjoint") {
    X <- matrix(stats::runif(2 * n, -4, 8), ncol = 2, nrow = n)
    loc1 <- c(-0.5, 0)
    loc2 <- c(4, 4)
    y <- rep(0, nrow(X))
    y[(X[, 1] - loc1[1])^2 + (X[, 2] - loc1[2])^2 < 6 + 
        stats::rnorm(nrow(X), sd = sd.noise)] <- 1
    y[(X[, 1] - loc2[1])^2 + (X[, 2] - loc2[2])^2 < 4 + 
        stats::rnorm(nrow(X), sd = sd.noise)] <- 2
  } else if (type == "surface") {
    X <- matrix(stats::runif(2 * n, -4, 6), ncol = 2, nrow = n)
    y <- X[, 1] * sin(X[, 2]) - X[, 2] * cos(X[, 1]) + 
      stats::rnorm(nrow(X), mean = 3, sd = sd.noise)
  } else if (type == "yin-yang") {
    r  <- seq(0.05, 0.8,   length.out = n) 
    t1 <- seq(0,    2.6, length.out = n) + stats::rnorm(n, sd = sd.noise) 
    c1 <- cbind(r*sin(t1)-0.1 , r*cos(t1)+0.1) 
    t2 <- seq(9.4,  12, length.out = n) + stats::rnorm(n, sd = sd.noise) 
    c2 <- cbind(r*sin(t2)+0.1 , r*cos(t2)-0.1) 
    X  <- rbind(c1, c2)
    y  <- as.numeric(1:(2*n)>n)+1
  } else print(paste0("Type ", type, " not supported."))
  return(list(X = X, y = as.numeric(y)))
}


plot_classif <- function(NN, X, y, epoch = "", standardize, example_type) {
  if (!requireNamespace("reshape2", quietly = TRUE)) {
    stop("Package \"reshape2\" needed for plotting classification example. Please install.", call. = FALSE)
  }
  x.seq <- seq(min(X), max(X), by = 0.1)
  xy.plot <- as.matrix(expand.grid(x.seq, x.seq))
  xy.pred <- data.frame(xy.plot, apply(predictC(NN, xy.plot, standardize), 1, which.max))
  #xy.pred <- data.frame(xy.plot, predict.NN(object = NN, newdata = xy.plot)$predictions)
  colnames(xy.pred) <- c("X1", "X2", "P")
  df.plot <- reshape2::dcast(xy.pred, X1 ~ X2, value.var = "P")
  graphics::image(x = x.seq, y = x.seq, data.matrix(df.plot[, -1]), 
                  col = c("#FFDEA6", "#B7E7FF", "#EEEEEE"),
                  main = paste0("Epoch: ", epoch), xlab = "x", ylab = "y")
  graphics::points(X, col = y + 1)
  # Add decision boundaries
  if (example_type == "linear") {
    graphics::lines(c(0, 1), c(0, 1), col = "darkgrey")
    graphics::lines(c(0, 1/3), c(1, 1/3), col = "darkgrey")
  } else if (example_type == "polynomial") {
    x.seq <- seq(-3, 7, by = 0.2)
    graphics::lines(x.seq, 2/3 * x.seq^2 - 1/9 * x.seq^3, col = "darkgrey")
  } else if (example_type == "nested") {
    c.seq <- seq(0, 2 * pi + 0.1, by = 0.1)
    r1 <- 17
    r2 <- 5
    loc <- c(2, 2)
    graphics::lines(sqrt(r1) * cos(c.seq) + loc[2], sqrt(r1) * sin(c.seq) + loc[2], col = "darkgrey")
    graphics::lines(sqrt(r2) * cos(c.seq) + loc[2], sqrt(r2) * sin(c.seq) + loc[2], col = "darkgrey")
  } else if (example_type == "disjoint") {
    c.seq <- seq(0, 2 * pi + 0.1, by = 0.1)
    r1 <- 6
    r2 <- 4
    loc1 <- c(-0.5, 0)
    loc2 <- c(4, 4)
    graphics::lines(sqrt(r1) * cos(c.seq) + loc1[1], sqrt(r1) * sin(c.seq) + loc1[2], col = "darkgrey")
    graphics::lines(sqrt(r2) * cos(c.seq) + loc2[1], sqrt(r2) * sin(c.seq) + loc2[2], col = "darkgrey")
  }
}

plot_regress <- function(NN, X, y, epoch = "", standardize) {
  if (!requireNamespace("rgl", quietly = TRUE)) {
    stop("Package \"rgl\" needed for plotting regression example. Please install.", call. = FALSE)
  }
  x.seq   <- seq(min(X), max(X), by = 0.2)
  xy.plot <- as.matrix(expand.grid(x.seq, x.seq))
  xy.pred <- data.frame(xy.plot, predictC(NN, xy.plot, standardize))
  colnames(xy.pred) <- c("X1", "X2", "Y")
  df.plot <- reshape2::dcast(xy.pred, X1 ~ X2, value.var = "Y")
  rgl::plot3d(cbind(X, y), main = paste0("Epoch: ", epoch), xlab = "", ylab = "", zlab = "")
  rgl::persp3d(x = x.seq, y = x.seq, z = as.matrix(df.plot[, -1]), col = c("red"), alpha = 0.4, add = TRUE)
}



