// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <iostream>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
mat tanhActivation(mat x){
  return 1.725*tanh(2*x/3);
}

// [[Rcpp::export]]
mat reluActivation(mat& x){
  return clamp(x, 0, x.max()); 
}

typedef mat (*funcPtr)(mat& x);

// [[Rcpp::export]]
XPtr<funcPtr> assignActivation(String activation_) {
  if (activation_ == "tanh")
    return(XPtr<funcPtr>(new funcPtr(&tanhActivation)));
  else if (activation_ == "relu")
    return(XPtr<funcPtr>(new funcPtr(&reluActivation)));
  else
    return XPtr<funcPtr>(R_NilValue); // runtime error as NULL no XPtr
}


class layer {
private:
  mat A, W, Z;
  funcPtr g;

public:
  layer(int nodes_in_, int nodes_out_, int batch_size_, String activation_) {
    W = randn<mat>(nodes_out_, nodes_in_);
    //mat A(nodes_in_, batch_size_);
    //mat Z(nodes_out_, batch_size_);
    
    XPtr<funcPtr> g_pointer = assignActivation(activation_);
    funcPtr g = *g_pointer;
    
    //Rcout << "\n W:\n"<< W << "\n A:\n" << A << "\n Z:\n"<< Z;
  }
  
  mat test (mat X) {
    Rcout << "\n X: \n" << X << "\n W: \n" << W;
    return X;
  }
  
  mat activationFunction (mat X) {
    return g(X);
  }
  
  mat forward (mat X){
    Rcout << W;
    Z = X;
    A = g(W * Z);
    return Z;
  }
};


class ANN {
public:
  ANN(ivec hidden_layers_, StringVector layer_activations_, 
      int nodes_in_, int nodes_out_) {
    Rcout << hidden_layers_ << layer_activations_ << nodes_in_ << nodes_out_;
    
    int n_layers_ = hidden_layers_.size();
    ivec seq_hidden = linspace<ivec>(1, n_layers_, n_layers_);
    ivec structure_nn = {nodes_in_, nodes_out_};
    Rcout << structure_nn << n_layers_;
    
    layer l1(5,4,2,"relu");
  }
private:
};

RCPP_MODULE(mod_ANN) {
  class_<ANN>( "ANN" )
  .constructor<ivec, StringVector, int, int>()
  ;
}

RCPP_MODULE(mod_layer) {
  class_<layer>( "layer" )
  .constructor<int, int, int, String>()
  .method( "activationFunction", &layer::activationFunction )
  .method( "forward", &layer::forward )
  .method( "test", &layer::test )
  ;
}


/*** R
#b <- new(ANN, c(5, 10), c('tanh', 'relu'), 3, 4)
#b

#x <- matrix(runif(1000, -5, 5), 200, 5)
#y <- tanhActivation(x)
#z <- reluActivation(x)

l <- new(layer, 5,5,10,'relu')
m <- matrix(rnorm(10,1,2),5,2)

l$test(m)
#l$activationFunction(m)
#l$forward(m)

*/
