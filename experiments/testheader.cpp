// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "testFactory.h"

// [[Rcpp::export]]
mat testFactory(List activ_param_, mat X) {
  // Set activation function
  activationFactory aFact(activ_param_); 
  activation *g = NULL;
  g = aFact.createActivation();
  return g->eval(X);
};

class ANN {
private:
  activation *g;
public:
  ANN(List activ_param_) {
    
    activationFactory aFact(activ_param_); 
    //*g = NULL;
    g = aFact.createActivation();
    
  }
  
  mat test (mat X) {
    return g->eval(X);
  }
  
};

RCPP_MODULE(mod_ANN) {
  class_<ANN>( "ANN" )
  .constructor<List>()
  .method( "test", &ANN::test)
  ;
}

/*** R
activ_params <- list(type = 'tanh', H = 5, k = 100)
X <- matrix(rnorm(100)*3, 50, 2)
y <- testFactory(activ_params, X)
plot(c(X), c(y))

a <- new(ANN, activ_params)
z <- a$test(X)
plot(c(X), c(z))
*/