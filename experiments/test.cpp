// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

class stepActivationOptimized
{
private:
  double H, k;
  vec seqH;
  mat::iterator mit;
public:
  stepActivationOptimized (List activ_param_) 
    : H(activ_param_["H"]), k(activ_param_["k"]) 
  {
    seqH = linspace(1, (H - 1), (H - 1))/ H;
    Rcout << seqH;
  }
  
  mat eval(mat X) 
  {
    mat A = X;
    for(mit = A.begin(); mit!=A.end(); mit++){
      (*mit) = sum( tanh( k * ((*mit) - seqH) ) );
    }
    return 0.5 + A / ( 2 * (H - 1) );
  }
  
  mat grad(mat X) 
  {
    mat gA = X;
    for(mit = gA.begin(); mit!=gA.end(); mit++){
      (*mit) = sum(1 - pow( tanh( k * ((*mit) - seqH) ), 2) );
    }
    return k * gA / ( 2 * (H-1) );
  }
  
};

RCPP_MODULE(mod_SFO) {
  class_<stepActivationOptimized>( "stepActivationOptimized" )
  .constructor<List>()
  .method( "eval", &stepActivationOptimized::eval)
  .method( "grad", &stepActivationOptimized::grad)
  ;
}

class stepActivation
{
private:
  double H, k;
public:
  stepActivation (List activ_param_) 
    : H(activ_param_["H"]), k(activ_param_["k"]) {}
  
  mat eval(mat X) 
  {
    uword N = X.n_elem;
    mat A(size(X));
    vec seqH = linspace(1, (H-1), (H-1))/ H;
    for(arma::uword i = 0; i!=N; i++){
      A(i) = sum( tanh( k * ( X(i) - seqH ) ) );
    }
    return 0.5 + A / ( 2 * (H-1) );
  }
  
  mat grad(mat X) 
  { 
    uword N = X.n_elem;
    mat gA(size(X));
    vec seqH = arma::linspace(1, (H-1), (H-1))/ H;
    for(uword i = 0; i!=N; i++){
      gA(i) = sum(1 - pow( tanh( k * (X(i) - seqH) ), 2) );
    }
    return k * gA / ( 2 * (H-1) );
  }
  
};

RCPP_MODULE(mod_SF) {
  class_<stepActivation>( "stepActivation" )
  .constructor<List>()
  .method( "eval", &stepActivation::eval)
  .method( "grad", &stepActivation::grad)
  ;
}

// [[Rcpp::export]]
mat eval(mat X, double H, double k) {
  uword N = X.n_elem;
  mat A(size(X));
  vec seqH = linspace(1, (H-1), (H-1))/ H;
  for(arma::uword i = 0; i!=N; i++){
    A(i) = sum( tanh( k * ( X(i) - seqH ) ) );
  }
  return 0.5 + A / ( 2 * (H-1) );
}

// [[Rcpp::export]]
mat eval2(mat X, double H, double k) {
  mat A = X;
  mat::iterator mit;
  vec seqH = linspace(1, (H-1), (H-1))/ H;
  for(mit = A.begin(); mit!=A.end(); mit++){
    (*mit) = sum( tanh( k * ( (*mit) - seqH ) ) );
  }
  return 0.5 + A / ( 2 * (H-1) );
}

// [[Rcpp::export]]
mat grad(mat X, double H, double k) { 
  uword N = X.n_elem;
  mat gA(size(X));
  vec seqH = arma::linspace(1, (H-1), (H-1))/ H; // Ok not ()H-1)??
  for(uword i = 0; i!=N; i++){
    gA(i) = sum(1 - pow( tanh( k * (X(i) - seqH) ), 2) );
  }
  return k * gA / ( 2 * (H-1) );
}

/*** R
x <- matrix(rnorm(3000), 1000, 3)

E <- eval(x, 5, 100)
E2 <- eval2(x, 5, 100)
G <- grad(x, 5, 100)

pX <- c(x)
pE <- c(E)
pG <- c(G)
plot(pX[order(pX)], pE[order(pX)], type = 'l', xlim = c(-0.2,1.2), col = 'blue')
lines(pX[order(pX)], pG[order(pX)], col = 'red')

activ_params <- list(H = 5, k = 100)
sfO <- new(stepActivationOptimized, activ_params)
sf <- new(stepActivation, activ_params)

E <- sfO$eval(x)
E2 <- sf$eval(x)
G <- sfO$grad(x)
G2 <- sf$grad(x)

all.equal(E, E2)
all.equal(G, G2)

pX <- c(x)
pE <- c(E2)
pG <- c(G2)
plot(pX[order(pX)], pE[order(pX)], type = 'l', xlim = c(-0.2,1.2), col = 'blue')
lines(pX[order(pX)], pG[order(pX)], col = 'red')

#library('rbenchmark')
#benchmark(sfO$eval(x), sf$eval(x), replications = 100000)
#benchmark(sfO$grad(x), sf$grad(x), replications = 100000)
*/