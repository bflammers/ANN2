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
  mat A, Z;
  activation g;
  optimizerFactory oFact;
  
public:
  mat W, D;
  vec b;
  layer(int nodes_in_, int nodes_out_, String activation_, int H_, int k_, 
        double lambda_, double m_, double L1_, double L2_) : 
        g(activation_, H_, k_) {
    
    // Initialize weight matrix and biasvector
    W = randn<mat>(nodes_out_, nodes_in_) / sqrt(nodes_in_);
    b = zeros<vec>(nodes_out_);
    
    optimizerFactory oFact(W, b, lambda_, m_, L1_, L2_);
    // optimizer *O = NULL;
    // O = fact.createOptimizer(type);
    
    
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
  
  void printDim() {
    Rcout << "W: rows " << W.n_rows << " cols " << W.n_cols << "\n";
    Rcout << "b: size " << b.size() << "\n";
    Rcout << "D: rows " << D.n_rows << " cols " << D.n_cols << "\n";
    Rcout << "A: rows " << A.n_rows << " cols " << A.n_cols << "\n";
    Rcout << "Z: rows " << Z.n_rows << " cols " << Z.n_cols << "\n";
  }
};

// RCPP_MODULE(mod_layer) {
//   class_<layer>( "layer" )
//   .constructor<int, int, String, int, int>()
//   .method( "forward", &layer::forward )
//   .method( "backward", &layer::backward )
//   ;
// }

class ANN {
private:
  std::list<layer>::iterator it;
  std::list<layer>::reverse_iterator rit;
  loss L;
  mat y_fit;
  
  
public:
  std::list<layer> layers;
  ANN(ivec num_nodes_, StringVector layer_activations_, String loss_, int H_, 
      int k_, double dHuber_, double lambda_, double m_, double L1_, double L2_)
    : L(loss_, dHuber_) {
  
    int n_layers = num_nodes_.size();
    for(int i = 1; i!=n_layers; i++){
      Rcout << "\n Layer: " << i << "/" << n_layers << " --> " << 
        layer_activations_(i);
      layer l(num_nodes_(i-1), num_nodes_(i), layer_activations_(i), H_, k_, 
              lambda_, m_, L1_, L2_);
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
      //Rcout << rit->D;
    }
    // Remove when finished
    return E; 
  }
  
  void printLayers () {
    for(it = layers.begin(); it != layers.end(); ++it) {
      it->printDim();
      Rcout << "\n\n";
    }
  }
  
};

RCPP_MODULE(mod_ANN) {
  class_<ANN>( "ANN" )
  .constructor<ivec, StringVector, String, int, int, double, double, double, double, double>()
  .method( "forwardPass", &ANN::forwardPass)
  .method( "backwardPass", &ANN::backwardPass)
  .method( "printLayers", &ANN::printLayers)
  ;
}


/*** R
#l <- new(layer, 5, 5, 'relu', 4, 5)
#m <- matrix(rnorm(10,1,2),5,2)
#l$forward(m)

b_s <- 10

a <- new(ANN, c(2,5,4,3,2), c('linear', 'tanh', 'relu', 'tanh', 'linear'), 'log', 
         0, 0, 0.6, 0.1, 0.3, 0.1, 0.8)
x <- matrix(rnorm(2 * b_s), b_s, 2)
e <- a$forwardPass(x)
e
a$backwardPass(e)

#a$printLayers()


*/

