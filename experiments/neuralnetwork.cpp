// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
#include "loss.h"
#include "activations.h"
#include "optimizer.h"
using namespace Rcpp;
using namespace arma;

class layer 
{
private:
  mat W, A_prev, Z;
  vec b;
  activation *g;
  optimizer *O;
  
public:
  layer(int nodes_in_, int nodes_out_, List activ_param_, List optim_param_)
  {
    // Initialize weight matrix and biasvector
    W = randn<mat>(nodes_out_, nodes_in_) / sqrt(nodes_in_);
    b = zeros<vec>(nodes_out_);
    
    // Set optimizer
    optimizerFactory oFact(W, b, optim_param_);
    O = oFact.createOptimizer();
    
    // Set activation function
    activationFactory aFact(activ_param_); 
    g = aFact.createActivation();
    
    Rcout << "\n Layer - "<< nodes_out_ << " nodes - " << 
      as<std::string>(activ_param_["type"]);
  }
  
  mat forward (mat X) 
  {
    A_prev = X; 
    Z = W * X + repColVec(b, X.n_cols);
    return g->eval(Z);
  }
  
  mat backward (mat E) 
  {
    mat D = E % g->grad(Z).t();
    W = O->updateW(W, D, A_prev);
    b = O->updateb(b, D);
    return D * W;
  }
};

class ANN 
{
private:
  std::list<layer> layers;
  std::list<layer>::iterator it;
  std::list<layer>::reverse_iterator rit;
  int epoch;
  loss *L;
  mat y_fit;
  
public:
  ANN(ivec num_nodes_, List optim_param_, List loss_param_, List activ_param_)
  {
    // Set loss
    lossFactory lFact(loss_param_); 
    L = lFact.createLoss();
    
    // Set iterable parameter vectors for activation type and lambda
    List activ_param = activ_param_;
    List optim_param = optim_param_;
    StringVector activ_types = activ_param["types"];
    vec lambdas = optim_param["lambdas"];
    
    // Set layers
    for(int i = 1; i!=num_nodes_.size(); i++){
      activ_param["type"] = activ_types(i);
      optim_param["lambda"] = lambdas(i);
      layer l(num_nodes_(i-1), num_nodes_(i), activ_param, optim_param);
      layers.push_back(l);
    }
  }
  
  void forwardPass (mat X) 
  {
    X = X.t();
    for(it = layers.begin(); it != layers.end(); ++it) {
      X = it->forward(X);
    }
    y_fit = X.t();
  }
  
  void backwardPass (mat y) 
  {
    mat E = L->grad(y, y_fit);
    for(rit = layers.rbegin(); rit != layers.rend(); ++rit) {
      E = rit->backward(E);
    }
  }
  
  mat partialForward (mat X, int i_start, int i_stop) 
  {
    X = X.t();
    
    // Set start & stop point iterators
    std::list<layer>::iterator start_it = layers.begin();
    std::advance(start_it, i_start);
    std::list<layer>::iterator stop_it = layers.begin();
    std::advance(stop_it, i_stop);
    
    // Loop from start_it to stop_it
    for(it = start_it; it != stop_it; ++it) {
      X = it->forward(X);
    }
    return X.t();
  }
  
  mat predict (mat X) 
  {
    X = X.t();
    for(it = layers.begin(); it != layers.end(); ++it) {
      X = it->forward(X);
    }
    return X.t();
  }
  
  double evalLoss(mat y, mat X) 
  {
    return accu( L->eval(y, predict(X)) );
  }
  
};

RCPP_MODULE(mod_ANN) {
  class_<ANN>( "ANN" )
  .constructor<ivec, List, List, List>()
  .method( "forwardPass", &ANN::forwardPass)
  .method( "backwardPass", &ANN::backwardPass)
  .method( "predict", &ANN::predict)
  .method( "evalLoss", &ANN::evalLoss)
  .method( "partialForward", &ANN::partialForward)
  ;
}


/*** R
#l <- new(layer, 5, 5, 'relu', 4, 5)
#m <- matrix(rnorm(10,1,2),5,2)
#l$forward(m)

b_s <- 100
loss_params  <- list(type = "pseudoHuber", dHuber = 1)
activ_params <- list(types = c('input', 'sigmoid', 'sigmoid', 'sigmoid', 'softmax'), H = 5, k = 100)
optim_params <- list(type = 'sgd', lambdas = c(0, 0.9, 0.8, 0.7, 0.6), m = 0.9, L1 = 0, L2 = 0)

a <- new(ANN, c(2,5,5,5,2), optim_params, loss_params, activ_params)
n <- 200
sd_noise = 0.8
r  <- seq(0.05, 0.8,   length.out = n)
t1 <- seq(0,    2.6, length.out = n) + stats::rnorm(n, sd = sd_noise)
c1 <- cbind(r*sin(t1)-0.1 , r*cos(t1)+0.1)
t2 <- seq(9.4,  12, length.out = n) + stats::rnorm(n, sd = sd_noise)
c2 <- cbind(r*sin(t2)+0.1 , r*cos(t2)-0.1)
X  <- rbind(c1, c2)
y  <- as.numeric(1:(2*n)>n)+1
plot(X, col = y)

Y <- t(sapply(y, function(yy) as.numeric(yy == c(1,2))))


for(i in 0:5000) {
  a$forwardPass(X)
  a$backwardPass(Y)
  if (i %% 500 == 0){
    print(a$evalLoss(Y, X))
    
    x_seq <- seq(min(X), max(X), by = 0.1)
    xy_plot <- as.matrix(expand.grid(x_seq, x_seq))
    xy_pred <- data.frame(xy_plot, apply(a$predict(xy_plot), 1, which.max))
    #xy.pred <- data.frame(xy.plot, predict.NN(object = NN, newdata = xy.plot)$predictions)
    colnames(xy_pred) <- c("X1", "X2", "P")
    df_plot <- reshape2::dcast(xy_pred, X1 ~ X2, value.var = "P")
    graphics::image(x = x_seq, y = x_seq, data.matrix(df_plot[, -1]),
                    col = c("#FFDEA6", "#B7E7FF", "#EEEEEE"),
                    main = paste0("Epoch: ", i), xlab = "x", ylab = "y")
    graphics::points(X, col = y + 1)
    Sys.sleep(0.5)
  }
}

*/

