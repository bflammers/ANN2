
#include <RcppArmadillo.h>
#include "test_utils.h"

using namespace Rcpp;
using namespace arma;

// This is needed for random number generation since Armadillo's RNG does not 
// seem to work nicely with testthat. Not sure what is going wrong. The first 
// time the test runs, the number are generated as expected by randu or randn
// but the second time, it generates all values close to zero
std::mt19937 RNG_engine;  // Mersenne twister random number engine

// Gaussian Random Matrix Generator
mat RNG_gaussian(int n_rows, int n_cols, double mu, double sd) {
  std::normal_distribution<double> distr_gaussian(mu, sd);
  mat X(n_rows, n_cols);
  X.imbue( [&]() { return distr_gaussian(RNG_engine); } );
  return X;
}

// Uniform Random Matrix Generator
mat RNG_uniform(int n_rows, int n_cols, double min_val, double max_val) {
  std::uniform_real_distribution<double> distr_uniform(min_val, max_val);
  mat X(n_rows, n_cols);
  X.imbue( [&]() { return distr_uniform(RNG_engine); } );
  return X;
}

// Binomial Random Matrix Generator
mat RNG_bernoulli(int n_rows, int n_cols, double p) {
  std::bernoulli_distribution distr_bernoulli(p);
  mat X(n_rows, n_cols);
  X.imbue( [&]() { return distr_bernoulli(RNG_engine); } );
  return X;
}

// ---------------------------------------------------------------------------//
// OPTIMIZER TESTING
// ---------------------------------------------------------------------------//

double OptimizerTester::rosenbrock_eval(mat params) {
  double x = params(0,0), y = params(0,1);
  return std::pow(1 - x, 2) + 100 * std::pow(y - std::pow(x, 2), 2);
}

double OptimizerTester::rosenbrock_eval(vec params) {
  double x = params(0), y = params(1);
  return std::pow(1 - x, 2) + 100 * std::pow(y - std::pow(x, 2), 2);
}

mat OptimizerTester::rosenbrock_grad(mat params) {
  double x = params(0,0), y = params(0,1);
  double dx = -400 * x * (y - std::pow(x, 2)) - 2 * (1 - x);
  double dy = 200 * (y - std::pow(x, 2));
  vec grad = {{dx, dy}};
  return grad;
}

vec OptimizerTester::rosenbrock_grad(vec params) {
  double x = params(0), y = params(1);
  double dx = -400 * x * (y - std::pow(x, 2)) - 2 * (1 - x);
  double dy = 200 * (y - std::pow(x, 2));
  vec grad = {dx, dy};
  return grad;
}

OptimizerTester::OptimizerTester (std::string optim_type) {
  
  W = {{1.5, 1.5}};
  b = {1.5, 1.5};
  
  List optim_param = List::create(Named("type") = optim_type,
                                  Named("learn_rate") = 1e-4,
                                  Named("L1") = 0,
                                  Named("L2") = 0,
                                  Named("adam_beta1") = 0.9,
                                  Named("adam_beta2") = 0.999,
                                  Named("rmsprop_decay") = 0.9,
                                  Named("sgd_momentum") = 0.99);
  
  O = OptimizerFactory(W, b, optim_param);
  O->n_train = 1;
}

void OptimizerTester::step_W () {
  mat dW = rosenbrock_grad(W);
  W = O->updateW(W, dW, 1);
}

void OptimizerTester::step_b () {
  b = O->updateb(b, rosenbrock_grad(b));
}


// ---------------------------------------------------------------------------//
// ACTIVATION FUNCTION TESTING
// ---------------------------------------------------------------------------//

ActivationTester::ActivationTester (std::string activ_type, double rel_tol_, 
                                    double abs_tol_) 
  : abs_tol(abs_tol_), rel_tol(rel_tol_) {
  
  List activ_param = List::create(Named("type") = activ_type, 
                                  Named("step_H") = 5, 
                                  Named("step_k") = 60);
  g = ActivationFactory(activ_param);
}

// Gradient checking
// See: http://cs231n.github.io/neural-networks-3/#gradcheck
bool ActivationTester::grad_check (mat X, bool obs_wise) {
  
  bool grad_match;
  
  if ( obs_wise ) {
    // OBSERVATIONWISE check (so columnwise, since function input is t(X))
    
    int n_obs = X.n_cols, n_class = X.n_rows;
    IntegerVector class_idx = sample(n_class, n_obs, true) - 1;
    vec num_grad_vec(n_obs), ana_grad_vec(n_obs);
    
    // Numerical gradient
    mat A_min(X), A_max(X);
    for (int i = 0; i < n_obs; i++) {
      A_min(class_idx[i], i) -= 1e-5;
      A_max(class_idx[i], i) += 1e-5;
    }
    mat num_grad = (g->eval(A_max) - g->eval(A_min)) / 2e-5;
    
    // Analytical gradient
    mat _ = g->eval(X); // Needed because grad() reuses A from eval()
    mat ana_grad = g->grad(X);
    
    for (int i = 0; i < n_obs; i++) {
      num_grad_vec[i] = num_grad(class_idx[i], i);
      ana_grad_vec[i] = ana_grad(class_idx[i], i);
    }
    
    // Do they match?
    grad_match = approx_equal(num_grad_vec, ana_grad_vec, "reldiff", rel_tol);
    
  } else {
    // ELEMENTWISE check (each element of matrix X)
    
    // Numerical gradient
    mat num_grad = (g->eval(X + 1e-5) - g->eval(X - 1e-5)) / 2e-5;
    
    // Analytical gradient
    mat _ = g->eval(X); // Needed because grad() reuses A from eval()
    mat ana_grad = g->grad(X);
    
    // Do they match?
    grad_match = approx_equal(num_grad, ana_grad, "reldiff", rel_tol);
  }
  
  return grad_match;
}

// Eval Function function: input, output check
bool ActivationTester::eval_check (double in_value, double out_value) {
  
  mat A(1,1); A.fill(in_value);
  mat B = g->eval(A);
  mat C(1,1); C.fill(out_value);
  
  return approx_equal(B, C, "both", abs_tol, rel_tol) ;
}

// ---------------------------------------------------------------------------//
// LOSS FUNCTION TESTING
// ---------------------------------------------------------------------------//

LossTester::LossTester (std::string loss_type, double rel_tol_, double abs_tol_)
  : abs_tol(abs_tol_), rel_tol(rel_tol_) {
  
  List loss_param = List::create(Named("type") = loss_type, 
                                 Named("huber_delta") = 1);
  L = LossFactory(loss_param);
}

// Gradient checking
// See: http://cs231n.github.io/neural-networks-3/#gradcheck
bool LossTester::grad_check (arma::mat y, arma::mat y_fit) {
  
  // Numerical gradient
  mat num_grad = (L->eval(y, y_fit + 1e-5) - L->eval(y, y_fit - 1e-5)) / 2e-5;
  
  // Analytical gradient
  mat _ = L->eval(y, y_fit); // Needed because grad() reuses A from eval()
  mat ana_grad = L->grad(y, y_fit);
  
  return approx_equal(num_grad, ana_grad, "reldiff", rel_tol);
}

// Eval Function function: input, output check
bool LossTester::eval_check (double in_y, double in_y_fit, double out_value) {
  
  mat y(1,1), y_fit(1,1); 
  y.fill(in_y); y_fit.fill(in_y_fit);
  mat B = L->eval(y, y_fit);
  mat C(1,1); C.fill(out_value);
  
  return approx_equal(B, C, "both", abs_tol, rel_tol) ;
}


