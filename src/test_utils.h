#ifndef TEST_UTILS_H 
#define TEST_UTILS_H

#include <RcppArmadillo.h>

// Include needed header files
#include "Activations.h"


// ---------------------------------------------------------------------------//
// ACTIVATION FUNCTIONS TESTING
// ---------------------------------------------------------------------------//

class ActivationTester 
{
private:
  double rel_tol, abs_tol;
  
public:
  std::unique_ptr<Activation> g;
  ActivationTester (std::string activ_type, double rel_tol_, double abs_tol_);
  bool gradient_check (arma::mat X);
  bool eval_check (double in_value, double out_value);
  
};


#endif
