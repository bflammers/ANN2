// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
#include "loss_functions.h"
#include "activation_functions.h"
using namespace Rcpp;
using namespace arma;

typedef mat (*funcPtrA)(mat& X, int& H, int& k);
typedef mat (*funcPtrL)(mat& y, mat& y_fit, double& dHuber);
typedef mat (*funcPtrO)(mat& W, vec& b, double& lambda, double& m, double& L1, 
                        double& L2);

class loss {
private:
  double dHuber;
  funcPtrL L, dL;
  
public:
  loss(String loss_, double dHuber_) : dHuber(dHuber_) {
    
    // Assign activation function based on string
    XPtr<funcPtrL> L_pointer = assignLoss(loss_);
    L = *L_pointer;
    
    // Assign derivative function based on string
    XPtr<funcPtrL> dL_pointer = assignLossDeriv(loss_);
    dL = *dL_pointer;
  }
  
  mat eval (mat y, mat y_fit) {
    return L(y, y_fit, dHuber);
  }
  
  mat grad (mat y, mat y_fit) {
    return dL(y, y_fit, dHuber);
  }
  
};

class activation {
private:
  int H, k;
  funcPtrA g, dg;
  
public:
  activation(String activation_, int H_, int k_) : H(H_), k(k_) {

    // Assign activation function based on string
    XPtr<funcPtrA> g_pointer = assignActiv(activation_);
    g = *g_pointer;
    
    // Assign derivative function based on string
    XPtr<funcPtrA> dg_pointer = assignActivDeriv(activation_);
    dg = *dg_pointer;
  }
  
  mat eval (mat X) {
    return g(X, H, k);
  }
  
  mat grad (mat X) {
    return dg(X, H, k);
  }
  
};

class layer {
private:
  mat A, Z, D;
  activation g;
  
public:
  mat W;
  vec b;
  layer(int nodes_in_, int nodes_out_, String activation_, int H_, int k_) : 
        g(activation_, H_, k_) {
    
    // Initialize weight matrix and biasvector
    W = randn<mat>(nodes_out_, nodes_in_) / sqrt(nodes_in_);
    b = zeros<vec>(nodes_out_);
    
  }
  
  mat forward (mat X){
    int batch_size = X.n_cols;
    Z = W * X + repColVec(b, batch_size);
    // Consider not storing A if not used in backward() and update()
    A = g.eval(Z);
    return A;
  }
  
  mat backward (mat E){
    D = E % g.grad(Z).t();
    return D * W;
  }
};

RCPP_MODULE(mod_layer) {
  class_<layer>( "layer" )
  .constructor<int, int, String, int, int>()
  .method( "forward", &layer::forward )
  .method( "backward", &layer::backward )
  ;
}

class ANN {
private:
  std::list<layer>::iterator it;
  std::list<layer>::reverse_iterator rit;
  loss L;
  mat y_fit;
  
  
public:
  std::list<layer> layers;
  ANN(ivec num_nodes_, StringVector layer_activations_, String loss_, int H_, 
      int k_, double dHuber_) : L(loss_, dHuber_) {

    int n_layers = num_nodes_.size();
    for(int i = 1; i!=n_layers; i++){
      Rcout << "\n Layer: " << i << "/" << n_layers << " --> " << 
        layer_activations_(i);
      layer l(num_nodes_(i-1), num_nodes_(i), layer_activations_(i), H_, k_);
      layers.push_back(l);
    }
    
    
  }
  
  mat forwardPass (mat X) {
    X = X.t();
    for(it = layers.begin(); it != layers.end(); ++it) {
      X = it->forward(X);
    }
    y_fit = X.t();
    return y_fit;
  }
  
  mat backwardPass (mat y) {
    mat E = L.grad(y, y_fit);
    for(rit = layers.rbegin(); rit != layers.rend(); ++rit) {
      E = rit->backward(E);
    }
    // Remove when finished
    return E; 
  }
  
};

RCPP_MODULE(mod_ANN) {
  class_<ANN>( "ANN" )
  .constructor<ivec, StringVector, String, int, int, double>()
  .method( "forwardPass", &ANN::forwardPass)
  .method( "backwardPass", &ANN::backwardPass)
  ;
}


/*** R
#l <- new(layer, 5, 5, 'relu', 4, 5)
#m <- matrix(rnorm(10,1,2),5,2)
#l$forward(m)

a <- new(ANN, c(2,5,4,3,2), c('linear', 'tanh', 'relu', 'tanh', 'linear'), 'log', 0, 0, 0.6)
x <- matrix(rnorm(30), 10, 2)
e <- a$forwardPass(x)
e
a$backwardPass(e)



*/

