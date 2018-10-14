#ifndef LAYER_H
#define LAYER_H

#include <RcppArmadillo.h>
#include "utils.h"
#include "Activations.h"
#include "Optimizer.h"

// Class Layer
class Layer 
{
private:
  arma::mat W, A_prev, Z;
  arma::vec b;
  Activation *g;
  Optimizer *O;
  
public:
  int n_nodes;
  std::string activ_type;
  
  Layer ();
  Layer(int nodes_in_, int nodes_out_, Rcpp::List activ_param_, Rcpp::List optim_param_);
  arma::mat forward (arma::mat X);
  arma::mat backward (arma::mat E); 
  
  // template<class Archive>
  // void save(Archive & archive) const;
  // 
  // template<class Archive>
  // void load(Archive & archive);
};

#endif