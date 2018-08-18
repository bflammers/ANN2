// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
#include "loss.h"
#include "activations.h"
#include "optimizer.h"
using namespace Rcpp;
using namespace arma;

class layer {
private:
  mat A_prev, Z, D;
  activation *g;
  optimizer *O;
  
public:
  mat W;
  vec b;
  layer(int nodes_in_, int nodes_out_, List activ_param_, List optim_param_) {
    
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
  
  mat forward (mat X) {
    int batch_size = X.n_cols;
    A_prev = X; 
    Z = W * X + repColVec(b, batch_size);
    return g->eval(Z);
  }
  
  mat backward (mat E) {
    D = E % g->grad(Z).t();
    W = O->updateW(W, D, A_prev);
    b = O->updateb(b, D);
    return D * W;
  }
  
  mat predict (mat X) {
    int batch_size = X.n_cols;
    mat Z = W * X + repColVec(b, batch_size);
    return g->eval(Z);
  }

};

class ANN {
private:
  std::list<layer>::iterator it;
  std::list<layer>::reverse_iterator rit;
  loss *L;
  mat y_fit;
  
public:
  std::list<layer> layers;
  StringVector activ_types;
  List activ_param;
  
  ANN(ivec num_nodes_, List optim_param_, List loss_param_, List activ_param_) 
    : activ_param(activ_param_) {
    
    // Set loss
    lossFactory lFact(loss_param_); 
    L = lFact.createLoss();
    
    // Set layers
    activ_types = activ_param_["types"];
    int n_layers = num_nodes_.size();
    for(int i = 1; i!=n_layers; i++){
      activ_param["type"] = activ_types(i);
      layer l(num_nodes_(i-1), num_nodes_(i), activ_param, optim_param_);
      layers.push_back(l);
    }
  }
  
  void forwardPass (mat X) {
    X = X.t();
    for(it = layers.begin(); it != layers.end(); ++it) {
      X = it->forward(X);
    }
    y_fit = X.t();
  }
  
  void backwardPass (mat y) {
    mat E = L->grad(y, y_fit);
    for(rit = layers.rbegin(); rit != layers.rend(); ++rit) {
      E = rit->backward(E);
    }
  }
  
  mat predict (mat X) {
    X = X.t();
    for(it = layers.begin(); it != layers.end(); ++it) {
      X = it->predict(X);
    }
    return X.t();
  }
  
  double evalLoss(mat y, mat X) {
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
  ;
}


/*** R
#l <- new(layer, 5, 5, 'relu', 4, 5)
#m <- matrix(rnorm(10,1,2),5,2)
#l$forward(m)

b_s <- 100
loss_params  <- list(type = "log", dHuber = NA)
activ_params <- list(types = c('input', 'tanh', 'softmax'), H = 5, k = 100)
optim_params <- list(type = 'sgd', lambda = 0.2, m = 0.9, L1 = 0, L2 = 0)

a <- new(ANN, c(2,100,2), optim_params, loss_params, activ_params)
n <- 500
sd_noise = 0.6
r  <- seq(0.05, 0.8,   length.out = n)
t1 <- seq(0,    2.6, length.out = n) + stats::rnorm(n, sd = sd_noise)
c1 <- cbind(r*sin(t1)-0.1 , r*cos(t1)+0.1)
t2 <- seq(9.4,  12, length.out = n) + stats::rnorm(n, sd = sd_noise)
c2 <- cbind(r*sin(t2)+0.1 , r*cos(t2)-0.1)
X  <- rbind(c1, c2)
y  <- as.numeric(1:(2*n)>n)+1
plot(X, col = y)

Y <- t(sapply(y, function(yy) as.numeric(yy == c(1,2))))


for(i in 0:1000) {
  a$forwardPass(X)
  a$backwardPass(Y)
  if (i %% 100 == 0){
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

