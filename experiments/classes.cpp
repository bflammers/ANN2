// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
#include "nn_functions.h"
using namespace Rcpp;
using namespace arma;

typedef mat (*funcPtr)(mat& x, int& H, int& k);

class activation {
private:
  int H, k;
  funcPtr g, dg;
  
public:
  activation(String activation_, int H_, int k_) {
    H = H_;
    k = k_;
    
    // Assign activation function based on string
    XPtr<funcPtr> g_pointer = assignActivation(activation_);
    g = *g_pointer;
    
    // Assign derivative function based on string
    XPtr<funcPtr> dg_pointer = assignDerivative(activation_);
    dg = *dg_pointer;
  }
  
  mat activ (mat X) {
    return g(X, H, k);
  }
  
  mat grad (mat X) {
    return dg(X, H, k);
  }
  
};

class layer {
private:
  mat A, W, Z, dW;
  vec b;
  activation g;
  
public:
  layer(int nodes_in_, int nodes_out_, String activation_, int H_, int k_) : 
  g(activation_, H_, k_) {
    
    // Initialize weight matrix and biasvector
    W = randn<mat>(nodes_out_, nodes_in_) / sqrt(nodes_in_);
    b = zeros<vec>(nodes_out_);
    
  }
  
  mat forward (mat X){
    int batch_size = X.n_cols;
    Z = W * X + repColVec(b, batch_size);
    A = g.activ(Z);
    return A;
  }
  
  mat backward (mat E){
    int batch_size = E.n_cols;
    mat dW = E * A.t();
    return W * E % g.grad(Z);
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
  std::list<layer> layers;
  
  
public:
  ANN(ivec num_nodes_, StringVector layer_activations_, int H_, int k_) {
    Rcout << "\n Layers: \n" << num_nodes_ << "\n Activations: \n" << layer_activations_;
    
    int n_layers = num_nodes_.size();
    for(int i = 1; i!=n_layers; i++){
      Rcout << "\n i: " << i << "/" << n_layers << " a: " << layer_activations_(i);
      layer l(num_nodes_(i-1), num_nodes_(i), layer_activations_(i), H_, k_);
      layers.push_back(l);
    }
    Rcout << "\n\n" << layers.size();
  }
};

RCPP_MODULE(mod_ANN) {
  class_<ANN>( "ANN" )
  .constructor<ivec, StringVector, int, int>()
  ;
}


/*** R
#l <- new(layer, 5, 5, 'relu', 4, 5)
#m <- matrix(rnorm(10,1,2),5,2)
#l$forward(m)

a <- new(ANN, c(3,5,5,5,1), c('linear', 'tanh', 'relu', 'tanh', 'linear'), 0, 0)

*/

