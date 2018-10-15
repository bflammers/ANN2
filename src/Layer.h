#ifndef LAYER_H
#define LAYER_H

#include <RcppArmadillo.h>
#include "utils.h"
#include "Activations.h"
#include "Optimizer.h"

#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>

// Class Layer
class Layer 
{
private:
  arma::mat W, A_prev, Z;
  arma::vec b;
  Optimizer *O;
  
public:
  int n_nodes;
  std::shared_ptr<Activation> g;
  
  Layer ();
  Layer(int nodes_in_, int nodes_out_, Rcpp::List activ_param_, Rcpp::List optim_param_);
  arma::mat forward (arma::mat X);
  arma::mat backward (arma::mat E); 
  
  // Serialize
  template<class Archive>
  void save(Archive & archive) const
  {
    MatSerializer ser_W(W);
    VecSerializer ser_b(b);
    archive( ser_W, ser_b, g, n_nodes );
  }

  // Deserialze
  template<class Archive>
  void load(Archive & archive)
  {
    MatSerializer ser_W;
    VecSerializer ser_b;
    archive( serW, ser_b, g, n_nodes );
    W = ser_W.getMat();
    b = ser_b.getVec();
  }
};

#endif