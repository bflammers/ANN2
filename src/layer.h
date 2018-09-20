#ifndef LAYER_H
#define LAYER_H

#include <RcppArmadillo.h>
#include "utils.h"
#include "activations.h"
#include "optimizer.h"
using namespace Rcpp;
using namespace arma;

// Class layer
class layer 
{
private:
  mat W, A_prev, Z;
  vec b;
  activation *g;
  optimizer *O;
  
public:
  int n_nodes;
  std::string activ_type;
  layer(int nodes_in_, int nodes_out_, List activ_param_, List optim_param_);
  mat forward (mat X);
  mat backward (mat E); 
};

#endif