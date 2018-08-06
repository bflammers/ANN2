// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
#include "nn_functions.h"
using namespace Rcpp;
using namespace arma;

//typedef mat (*funcPtr)(mat& x);

class layer {
private:
  mat A, W, Z, dW;
  vec b;
  funcPtr g;
  
public:
  layer(int nodes_in_, int nodes_out_, String activation_) {
    
    // Initialize weight matrix and biasvector
    W = randn<mat>(nodes_out_, nodes_in_) / sqrt(nodes_in_);
    b = ones<vec>(nodes_out_);
    
    // Assign activation function based on string
    XPtr<funcPtr> g_pointer = assignActivation(activation_);
    g = *g_pointer;
  }
  
  mat test (mat X) {
    Rcout << "\n X: \n" << X << "\n W: \n" << W;
    return X;
  }
  
  mat activationFunction (mat X) {
    return g(X);
  }
  
  mat forward (mat X){
    int batch_size = X.n_cols;
    Z = W * X + repColVec(b, batch_size);
    A = g(Z);
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
  .constructor<int, int, String>()
  .method( "activationFunction", &layer::activationFunction )
  .method( "forward", &layer::forward )
  .method( "test", &layer::test )
  ;
}


/*** R
#l <- new(layer, 10, 5, 'relu')
#m <- matrix(rnorm(10,1,2),5,2)

#l$test(m)
#l$activationFunction(m)
#l$forward(m)

*/

