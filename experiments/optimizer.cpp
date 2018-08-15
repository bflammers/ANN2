// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "optimizer.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Optimizer parameter object
// ---------------------------------------------------------------------------//

struct sgdParams : optimParams {
  double lambda, m, L1, L2;
  sgdParams (double lambda_, double m_, double L1_, double L2_)
    : lambda(lambda_), m(m_), L1(L1_), L2(L2_) {}
};
struct RMSpropParams : optimParams {
  double lambda;
};

// ---------------------------------------------------------------------------//
// Methods of base class optimizer
// ---------------------------------------------------------------------------//

mat optimizer::updateW(mat W, mat D, mat A_prev) { return W.zeros(); }
vec optimizer::updateb(vec b, mat D) { return b.zeros(); }
void optimizer::setParams(optimParams P) {}

// ---------------------------------------------------------------------------//
// Optimizers
// ---------------------------------------------------------------------------//

// SGD 
class SGD : public optimizer
{
private:
  sgdParams P;
  int batch_size;
  mat mW;
  vec mb;
public:
  SGD (mat W_templ_, vec b_templ_, sgdParams P_) : P(P_) {
    
    // Initialize momentum matrices
    mW = zeros<mat>(size(W_templ_));
    mb = zeros<vec>(size(b_templ_));
    
  }
  mat updateW(mat W, mat D, mat A_prev) {
    batch_size = A_prev.n_cols;
    mat gW = A_prev * D / batch_size;
    mW = P.m * mW - P.lambda * gW.t();
    return (1 - P.lambda - P.L2) * W - P.lambda * P.L1 * sign(W) + mW;
  }
  
  vec updateb(vec b, mat D) {
    mb = P.m * mb - P.lambda * sum(D, 0).t() / batch_size;
    return b + mb;
  }
};

// RMSprop 
class RMSprop : public optimizer
{
private:
  double m;
  mat dW;
  vec db;
public:
  RMSprop (double m_, mat W_) : m(m_), dW(W_) {
    m = m * 100;
    dW.ones();
  }
  
  mat updateW(mat W, mat D, mat A_prev) {
    return W.zeros();
  }
  
  vec updateb(vec b, mat D) {
    return b.zeros();
  }
  
};

// ---------------------------------------------------------------------------//
// Methods for class optimizer factory 
// ---------------------------------------------------------------------------//

// Constructor
optimizerFactory::optimizerFactory (mat W_, vec b_, double lambda_, double m_, 
                                    double L1_, double L2_) : lambda(lambda_), 
                                    m(m_), L1(L1_), L2(L2_) {
  // Store templates of W and b for initialization purposes
  W_templ = zeros<mat>(size(W_));
  b_templ = zeros<vec>(size(b_));
}

// Method for creating optimizers
optimizer *optimizerFactory::createOptimizer (String type) {
  if      (type == "SGD")     return new SGD(W_templ, b_templ, lambda, m, L1, L2);
  else if (type == "RMSprop") return new RMSprop(m, W_templ);
  else                        return NULL;
}


