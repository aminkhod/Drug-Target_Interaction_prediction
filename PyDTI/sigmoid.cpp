

#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]
#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;

//[[Rcpp::export]]
NumericMatrix sigmoid(NumericMatrix mat) {
  
  
  
  int m = mat.nrow();
  int n = mat.ncol();
  
  NumericMatrix res(m, n);
  double x;
  double z;
  
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      x = mat(i, j);
      if (x >= 0) {
        z = exp(-x);
        res(i, j) = 1 / (1 + z);
      } else {
        z = exp(x);
        res(i, j) = z / (1 + z);
      }
    }
  
  return res;
} 




