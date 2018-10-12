#ifndef LAYER_H
#define LAYER_H

#include <RcppArmadillo.h>
#include "utils.h"
#include "Activations.h"
#include "Optimizer.h"
using namespace Rcpp;
using namespace arma;

// Class Layer
class Layer 
{
private:
  mat W, A_prev, Z;
  vec b;
  Activation *g;
  Optimizer *O;
  
public:
  int n_nodes;
  std::string activ_type;
  
  Layer ();
  Layer(int nodes_in_, int nodes_out_, List activ_param_, List optim_param_);
  mat forward (mat X);
  mat backward (mat E); 
  
  template<class Archive>
  void save(Archive & archive) const;
  
  template<class Archive>
  void load(Archive & archive);
};

#endif