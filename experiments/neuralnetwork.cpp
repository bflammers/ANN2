// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
#include "loss.h"
#include "layer.h"
using namespace Rcpp;
using namespace arma;

class ANN 
{
private:
  std::list<layer> layers;
  std::list<layer>::iterator it;
  std::list<layer>::reverse_iterator rit;
  scaler sX, sY;
  loss *L;
  tracker T;
  int epoch;
  
public:
  ANN(mat X_, mat Y_, List net_param_, List optim_param_, List loss_param_, 
      List activ_param_)
    : sX(X_, net_param_["standardize_X"], net_param_),
      sY(Y_, net_param_["standardize_Y"], net_param_),
      epoch(0)
  {

    // Set loss
    lossFactory lFact(loss_param_); 
    L = lFact.createLoss();
    
    // Set iterable vectors for number of nodes, activation type and lambda
    ivec num_nodes = net_param_["num_nodes"];
    StringVector activ_types = activ_param_["types"];
    vec lambdas = optim_param_["lambdas"];
    
    // Set parameter Lists to be passed to layer()
    List activ_param = activ_param_;
    List optim_param = optim_param_;
    
    // Set layers
    for(int i = 1; i!=num_nodes.size(); i++){
      activ_param["type"] = activ_types(i);
      optim_param["lambda"] = lambdas(i);
      layer l(num_nodes(i-1), num_nodes(i), activ_param, optim_param);
      layers.push_back(l);
    }
  }
  
  mat forwardPass (mat X) 
  {
    X = X.t();
    for(it = layers.begin(); it != layers.end(); ++it) {
      X = it->forward(X);
    }
    return X.t();
  }
  
  void backwardPass (mat Y, mat Y_fit) 
  {
    mat E = L->grad(Y, Y_fit);
    for(rit = layers.rbegin(); rit != layers.rend(); ++rit) {
      E = rit->backward(E);
    }
  }
  
  mat partialForward (mat X, int i_start, int i_stop) 
  {
    
    // Set start & stop point iterators
    std::list<layer>::iterator start_it = layers.begin();
    std::advance(start_it, i_start - 1);
    std::list<layer>::iterator stop_it = layers.begin();
    std::advance(stop_it, i_stop - 1);
    
    // If input layer: standardize and transpose
    if ( i_start == 1 ) {
      X = sX.scale(X); // Check if start at 0
      X = X.t();
    }
    
    // Loop from start_it to stop_it
    for(it = start_it; it != stop_it; ++it) {
      X = it->forward(X);
    }
    
    // If output layer: transpose and standardize
    if ( i_stop == layers.size() ) {
      X = X.t();
      X = sY.unscale(X);
    }
    
    return X;
  }
  
  mat predict (mat X) 
  {
    X = sX.scale(X);
    X = forwardPass(X);
    mat Y_pred = sY.unscale(X);
    return Y_pred;
  }
  
  // Evaluate loss, input should be scaled data
  double evalLoss(mat Y, mat X)
  {
    return accu( L->eval(Y, forwardPass(X)) );
  }
  
  void train (mat X_, mat Y_, List train_param)
  {
    // Training parameters
    int n_epochs = train_param["n_epochs"];
    int max_epochs = epoch + n_epochs;

    // Scale data
    mat X = sX.scale(X_);
    mat Y = sY.scale(Y_);

    // Set sampler and tracker
    sampler data(X, Y, train_param);
    int n_new_passes = n_epochs * data.n_batch;
    T.setTracker(n_new_passes, data.validate, train_param);

    for (; epoch != max_epochs; epoch++) {
      
      // Shuffle data
      data.shuffle();
      
      for (int b = 0; b != data.n_batch; b++) {
        
        // Sample new batch
        mat Xb = data.nextXb();
        mat Yb = data.nextYb();
      
        // Forward pass
        mat Yb_fit = forwardPass(Xb);
        
        // Backward pass, also includes update
        backwardPass(Yb, Yb_fit);
        
        // Track loss on scaled data
        double batch_loss = L->eval(Yb, Yb_fit);
        double val_loss = (data.validate) ? evalLoss(data.getYv(), data.getYv()) : 0;
        T.track(batch_loss, val_loss);
        
        // Check for interrupt
        checkUserInterrupt();
      }
    }
  }
  
};

RCPP_MODULE(mod_ANN) {
  class_<ANN>( "ANN" )
  .constructor<mat, mat, List, List, List, List>()
  .method( "forwardPass", &ANN::forwardPass)
  .method( "backwardPass", &ANN::backwardPass)
  .method( "predict", &ANN::predict)
  .method( "evalLoss", &ANN::evalLoss)
  .method( "partialForward", &ANN::partialForward)
  .method( "train", &ANN::train)
  ;
}


/*** R
#l <- new(layer, 5, 5, 'relu', 4, 5)
#m <- matrix(rnorm(10,1,2),5,2)
#l$forward(m)

n <- 200
sd_noise = 0.3
r  <- seq(0.05, 0.8,   length.out = n)
t1 <- seq(0,    2.6, length.out = n) + stats::rnorm(n, sd = sd_noise)
c1 <- cbind(r*sin(t1)-0.1 , r*cos(t1)+0.1)
t2 <- seq(9.4,  12, length.out = n) + stats::rnorm(n, sd = sd_noise)
c2 <- cbind(r*sin(t2)+0.1 , r*cos(t2)-0.1)
X  <- rbind(c1, c2)
y  <- as.numeric(1:(2*n)>n)+1
plot(X, col = y)

Y <- t(sapply(y, function(yy) as.numeric(yy == c(1,2))))

loss_params  <- list(type = "log", dHuber = 1)
activ_params <- list(types = c('input', 'tanh', 'softmax'), H = 5, k = 100)
optim_params <- list(type = 'sgd', lambdas = c(0, 0.2, 0.01), m = 0.9, L1 = 0, L2 = 0)
net_params   <- list(standardize_X = TRUE, standardize_Y = FALSE, num_nodes = c(2,2,2))

a <- new(ANN, X, Y, net_params, optim_params, loss_params, activ_params)
# X <- matrix(1:20, 10, 2)
# Y <- matrix(1:20, 10, 2)

train_param <- list(n_epochs = 1000, batch_size = 32, val_prop = 0.1, drop_last = FALSE, verbose = TRUE)
a$train(X, Y, train_param)
x_seq <- seq(min(X), max(X), by = 0.1)
xy_plot <- as.matrix(expand.grid(x_seq, x_seq))
xy_pred <- data.frame(xy_plot, apply(a$predict(xy_plot), 1, which.max))


#test <- a$partialForward(xy_plot, 0, 102)
colnames(xy_pred) <- c("X1", "X2", "P")
df_plot <- reshape2::dcast(xy_pred, X1 ~ X2, value.var = "P")
graphics::image(x = x_seq, y = x_seq, data.matrix(df_plot[, -1]),
                col = c("#FFDEA6", "#B7E7FF", "#EEEEEE"),
                xlab = "x", ylab = "y")
graphics::points(X, col = y + 1)
*/

