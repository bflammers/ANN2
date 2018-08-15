// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

class optimizer {
public: 
  virtual mat updateW(mat W, mat D, mat A_prev) { return W.zeros(); }
  virtual vec updateb(vec b, mat D) { return b.zeros(); }
};

class SGD : public optimizer
{
private:
  double lambda, m, L1, L2;
  int batch_size;
  mat mW;
  vec mb;
public:
  SGD (mat W_templ_, vec b_templ_, double lambda_, double m_, double L1_, 
       double L2_) : lambda(lambda_), m(m_), L1(L1_), L2(L2_) {
    
    // Initialize momentum matrices
    mW = zeros<mat>(size(W_templ_));
    mb = zeros<vec>(size(b_templ_));
    
  }
  mat updateW(mat W, mat D, mat A_prev) {
    //Rcout << "1 ";
    batch_size = A_prev.n_cols;
    mat gW = A_prev * D / batch_size;
    //Rcout << "2 ";
    mW = m * mW - lambda * gW.t();
    //Rcout << "3 ";
    return (1 - lambda - L2) * W - lambda * L1 * sign(W) + mW;
  }
  
  vec updateb(vec b, mat D) {
    //Rcout << " b: " << b << "\n mb: " << mb << "\n D:" << D; 
    //Rcout << "batch_size: " << batch_size << "\n ---------";
    mb = m * mb - lambda * sum(D, 0).t() / batch_size;
    return b + mb;
  }
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
mat testFactory(String type, mat W_, vec b_, double lambda_, double m_,
                   double L1_, double L2_, mat D_, mat A_prev_) {
  optimizerFactory fact(W_, b_, lambda_, m_, L1_, L2_);
  optimizer *O = NULL; 
  O = fact.createOptimizer(type);
  mat uW = O->updateW(W_, D_, A_prev_);
  return O->updateb(b_, D_);
};



/*** R
# c(2,5,4,3,2)
n_in <- 5
n_out <- 4
b_s <- 15
W <- matrix(rnorm(n_in * n_out), n_out, n_in)
b <- rnorm(n_out)
D <- matrix(rnorm(n_out * b_s), b_s, n_out)
A_prev <- matrix(rnorm(n_in * b_s), n_in, b_s)
testFactory("SGD", W, b, 0.0001, 0.8, 0.5, 0.3, D, A_prev)
*/
