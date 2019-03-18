#ifndef TEST_UTILS_H 
#define TEST_UTILS_H

#include <RcppArmadillo.h>

// Include needed header files
#include "Optimizer.h"
#include "Activations.h"
#include "Loss.h"

// ---------------------------------------------------------------------------//
// RANDOM NUMBER GENERATION
// ---------------------------------------------------------------------------//

// This is needed for random number generation since Armadillo's RNG does not 
// seem to work nicely with testthat. Not sure what is going wrong. The first 
// time the test runs, the number are generated as expected by randu or randn
// but the second time, it generates all values close to zero

// Gaussian Random Matrix Generator
arma::mat RNG_gaussian(int n_rows, int n_cols, double mu = 0.0, 
                       double sd = 1.0);

// Uniform Random Matrix Generator
arma::mat RNG_uniform(int n_rows, int n_cols, double min_val = 0.0, 
                      double max_val = 1.0);

// Binomial Random Matrix Generator
arma::mat RNG_bernoulli(int n_rows, int n_cols, double p = 0.5);

// ---------------------------------------------------------------------------//
// OPTIMIZER TESTING
// ---------------------------------------------------------------------------//

class OptimizerTester 
{
private:
  double abs_tol;
  
public:
  arma::mat W;
  arma::vec b;
  std::unique_ptr<Optimizer> O;
  
  OptimizerTester (std::string optim_type, double abs_tol_);
  
  double rosenbrock_eval(arma::mat params);
  double rosenbrock_eval(arma::vec params);
  arma::mat rosenbrock_grad(arma::mat params);
  arma::vec rosenbrock_grad(arma::vec params);
  
  void step_W ();
  void step_b ();
  
};

// ---------------------------------------------------------------------------//
// FUNCTION TESTING
// ---------------------------------------------------------------------------//

class ActivationTester 
{
private:
  double abs_tol, rel_tol;
  
public:
  std::unique_ptr<Activation> g;
  ActivationTester (std::string activ_type, double rel_tol_, double abs_tol_);
  bool grad_check (arma::mat X, bool obs_wise = false);
  bool eval_check (double in_value, double out_value);
  
};

class LossTester 
{
private:
  double abs_tol, rel_tol;
  
public:
  std::unique_ptr<Loss> L;
  LossTester (std::string loss_type, double rel_tol_, double abs_tol_);
  bool grad_check (arma::mat y, arma::mat y_fit);
  bool eval_check (double in_y, double in_y_fit, double out_value);
  
};

#endif
