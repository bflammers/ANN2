// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

class optimizer {
public: 
  virtual mat updateW(mat W) { return W.zeros(); }
  virtual vec updateb(vec b) { return b.zeros(); }
};

class SGD : public optimizer
{
private:
  double lambda, m, L1, L2;
  mat gW, mW;
  vec gb, mb;
public:
  SGD (mat W_, vec b_, double lambda_, double m_, double L1_, double L2_) : 
       lambda(lambda_), m(m_), L1(L1_), L2(L2_) {
    
    // Initialize gradient and momentum matrices
    gW = zeros<mat>(size(W_));
    mW = zeros<mat>(size(W_));
    gb = zeros<vec>(size(b_));
    mb = zeros<vec>(size(b_));
    
  }
  double getValue() {return m;}
};

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
  double  getValue() {return m;}
};

class optimizerFactory
{
public:
  double lambda, m, L1, L2;
  mat W_templ;
  vec b_templ;
  optimizerFactory (mat W_, vec b_, double lambda_, double m_, double L1_, 
                    double L2_) : lambda(lambda_), m(m_), L1(L1_), L2(L2_) {
    // Store templates of W and b for initialization purposes
    W_templ = zeros<mat>(size(W_));
    b_templ = zeros<vec>(size(b_));
  }
  
  optimizer *createOptimizer (String type) {
    if      (type == "SGD")       return new SGD(W_templ, b_templ, lambda, m, L1, L2);
    else if (type == "RMSprop")   return new RMSprop(m, W_templ);
    else                          return NULL;
  }

};

// [[Rcpp::export]]
double testFactory(String type, double m_, mat W_, vec b_) {
  optimizerFactory fact(m_, W_, b_);
  optimizer *O = NULL;
  O = fact.createOptimizer(type);
  return O->getValue();
};



/*** R
testFactory("RMSprop", 0.7, diag(1:4), 5:7)

*/
