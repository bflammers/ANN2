// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

#include "Layer.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Layer class
// ---------------------------------------------------------------------------//
Layer::Layer () {}

Layer::Layer(int nodes_in_, int nodes_out_, List activ_param_, List optim_param_)
  : n_nodes(nodes_out_)
{
  
  // Initialize weight matrix and biasvector
  W = randn<mat>(nodes_out_, nodes_in_) / std::sqrt(nodes_in_);
  b = zeros<vec>(nodes_out_);
  
  // Set activation function
  g = ActivationFactory (activ_param_);
  
  // Set optimizer
  O = OptimizerFactory (W, b, optim_param_);
  
}
  
mat Layer::forward (mat X) 
{
  /* This method applies the transformation A = g( W * A_prev + b * iota )
   * where iota is rowvector of ones with X.n_col elements
   * For the first hidden layer, A_prev is the transposed input matrix  
   */
  
  // Store previous activation (or input matrix, in case of first hidden layer)
  // for use in backward()
  A_prev = X; 
  
  // Multiply by weight matrix and add bias vector to each column
  Z = W * X;
  Z.each_col() += b;
  
  // Apply activation function
  return g->eval(Z);
}

mat Layer::backward (mat E) 
{
  /* This method propagated the errors backward through the network
   * Furthermore, the gradients wrt. the weights and biases are calculated
   * and used for updating the weight matrix and bias vector using the optimizer
   */
  
  // Determine batch size for calculating average gradient over batch
  int batch_size = A_prev.n_cols;
  
  // Determine Error matrix (gradient with respect to Z = W * A_prev + b * iota)
  // This matrix is used to determine both the gradient wrt. W and b
  // L = g( ... g(Z) ... ) --> dL/dZ = E % g'(Z) where E is error propagated
  // from the previous layer - this is due to recursive application of the
  // chain rule of calculus
  mat D = E % g->grad(Z).t();
  
  // Determine average gradient matrix wrt. the weights by matrix multiplication with 
  // the previous activation (or input matrix, in case of the first hidden layer)
  // A = g(W * A_prev + b * iota) --> dL/dW = A_prev * g'(W * A_prev + b * iota)
  mat dW = A_prev * D / batch_size;
  
  // Update W
  W = O->updateW(W, dW);
  
  // Determine average gradient matrix wrt. the biases 
  // A = g(W * A_prev + b * iota) --> dL/db = iota * g'(W * A_prev + b * iota)
  vec db = sum(D, 0).t() / batch_size;
  
  // Update b
  b = O->updateb(b, db);
  
  // Propagate errors further backwards
  return D * W;
}

// Print methods - currently only used by ANN::print()
std::string Layer::print() {
  std::stringstream out;
  out << "  Layer - " << n_nodes << " nodes - " << g->type << " \n";
  return out.str();
}

