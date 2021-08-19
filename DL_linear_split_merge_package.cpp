#include <RcppArmadillo.h>
#include <RcppGSL.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include<omp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]

using namespace arma;
#define tolr 1e-5
#define crossprod(x) symmatu((x).t() * (x))
#define tcrossprod(x) symmatu((x) * (x).t())

struct claunch_list{
  uvec inds,claunch,nj_val;
  double log_q;
  vec marginal_density;
  cube chol_marginal_delta, marginal_delta;
};

struct eta_list{
  vec eta,sum_of_samples;
};

eta_list gen_eta(int nj,double niw_kap,const vec &sum_of_samples, vec Meta,
                 const mat &Delta_inv, const mat &chol_Delta,const mat &chol_precmat_eta ){
  unsigned k=sum_of_samples.n_elem;
  
  double niw_kap_post=nj-1+niw_kap;
  // sum_of_samples-=eta;
  vec marg_mu= sum_of_samples/niw_kap_post;
  
  vec z1=vec(k, fill::randn),z2=vec(k, fill::randn);
  
  Meta+=( Delta_inv*  marg_mu  +solve( trimatu(chol_Delta ) ,z2)/sqrt(niw_kap_post) ) ; ///since chol_deta is generated to be lower-triangular
  
  eta_list L;
  L.eta = solve(trimatu( chol_precmat_eta), z1 +solve(trimatl((chol_precmat_eta).t()),  Meta ) ) ;
  
  L.sum_of_samples= sum_of_samples+L.eta;
  return L;
}


// [[Rcpp::export]]
double c_lmvgamma (double x, int p) {
  int i;
  double ans = 0;
  if (p < 1)
    Rcpp::stop("p must be greater than or equal to 1.");
  if (x <= 0)
    Rcpp::stop("x must be greater than 0.");
  ans =(p * (p - 1)/4.0) * M_LNPI;
  for (i = 0; i < p; ++i){
    ans +=  (lgamma(x  - (i/2.0) ));
  }
  return ans;
}


double log_marg_dens (const mat &chol_marginal_delta,int n,double t1ppp
                        ,double  niw_nu,double  niw_kap,double  diag_psi_iw   ){
  int k=chol_marginal_delta.n_cols;
  double dens;
  dens=-(k/2)*(n* M_LNPI+log1p(n/niw_kap)-niw_nu*log(diag_psi_iw)- (niw_nu+n)*log(t1ppp) ) 
    +c_lmvgamma ( (niw_nu+n)/2 ,  k)- c_lmvgamma ( niw_nu/2 ,  k)- (niw_nu+n)*sum(log(chol_marginal_delta.diag()) );
  return dens;
}

// [[Rcpp::export]]
double rig(double mu){
  double y = randn<double>();
  y *= y;
  double mu2 = gsl_pow_2(mu);
  double quad = 4 * mu * y + mu2 * gsl_pow_2(y);
  // double x = mu + y * mu2 / 2 - mu / 2  * sqrt(quad);
  double  x_div_mu=(2+mu*y - sqrt(quad) )/2;
  double  x = (mu* x_div_mu);
  if(x<=0){
    // cout<<"mu= "<<mu<<" y= "<<y<<endl;
    // Rcpp::stop("Error at rig!!");
    // x=0;
    // cout<<"manual return at rig!!"<<endl;
    return(1e-5);
  }
  
  double u = log (randu<double>());
  // if(u <= (mu / (x + mu))) return x;
  if(u <=  -log1p(x_div_mu) ) return x;
  else return mu / x_div_mu;
}

double    _unur_bessel_k_nuasympt (double x, double nu, int islog, int expon_scaled)    {
#define M_LNPI     1.14472988584940017414342735135      /* ln(pi) */
  
  double z;                   /* rescaled argument for K_nu() */
  double sz, t, t2, eta;      /* auxiliary variables */
  double d, u1t,u2t,u3t,u4t;  /* (auxiliary) results for Debye polynomials */
  double res;                 /* value of log(K_nu(x)) [= result] */
  
  /* rescale: we comute K_nu(z * nu) */
  z = x / nu;
  
  /* auxiliary variables */
  sz = hypot(1,z);   /* = sqrt(1+z^2) */
  t = 1. / sz;
  t2 = t*t;
  
  eta = (expon_scaled) ? (1./(z + sz)) : sz;
  eta += log(z) - log1p(sz);                  /* = log(z/(1+sz)) */
  
  /* evaluate Debye polynomials u_j(t) */
  u1t = (t * (3. - 5.*t2))/24.;
  u2t = t2 * (81. + t2*(-462. + t2 * 385.))/1152.;
  u3t = t*t2 * (30375. + t2 * (-369603. + t2 * (765765. - t2 * 425425.)))/414720.;
  u4t = t2*t2 * (4465125. 
                   + t2 * (-94121676.
                   + t2 * (349922430. 
                   + t2 * (-446185740. 
                   + t2 * 185910725.)))) / 39813120.;
                   d = (-u1t + (u2t + (-u3t + u4t/nu)/nu)/nu)/nu;
                   
                   /* log(K_nu(x)) */
                   res = log(1.+d) - nu*eta - 0.5*(log(2.*nu*sz) - M_LNPI);
                   
                   return (islog ? res : exp(res));
}

double _gig_mode(double lambda, double omega)
  /*---------------------------------------------------------------------------*/
  /* Compute mode of GIG distribution.                                         */
  /*                                                                           */
  /* Parameters:                                                               */
  /*   lambda .. parameter for distribution                                    */
  /*   omega ... parameter for distribution                                    */
  /*                                                                           */
  /* Return:                                                                   */
  /*   mode                                                                    */
  /*---------------------------------------------------------------------------*/
{
  if (lambda >= 1.)
    /* mode of fgig(x) */
    return (sqrt((lambda-1.)*(lambda-1.) + omega*omega)+(lambda-1.))/omega;
  else
    /* 0 <= lambda < 1: use mode of f(1/x) */
    return omega / (sqrt((1.-lambda)*(1.-lambda) + omega*omega)+(1.-lambda));
} /* end of _gig_mode() */
    
    void
  _rgig_ROU_shift_alt (double *res, int n, double lambda, double lambda_old, double omega, double alpha)
  /*---------------------------------------------------------------------------*/
  /* Type 8:                                                                   */
  /* Ratio-of-uniforms with shift by 'mode', alternative implementation.       */
  /*   Dagpunar (1989)                                                         */
  /*   Lehner (1989)                                                           */
  /*---------------------------------------------------------------------------*/
  {
    double xm, nc;     /* location of mode; c=log(f(xm)) normalization constant */
  double s, t;       /* auxiliary variables */
  double U, V, X;    /* random variables */
  
  int i;             /* loop variable (number of generated random variables) */
  int count = 0;     /* counter for total number of iterations */
  
  double a, b, c;    /* coefficent of cubic */
  double p, q;       /* coefficents of depressed cubic */
  double fi, fak;    /* auxiliary results for Cardano's rule */
  
  double y1, y2;     /* roots of (1/x)*sqrt(f((1/x)+m)) */
  
  double uplus, uminus;  /* maximum and minimum of x*sqrt(f(x+m)) */
  
  /* -- Setup -------------------------------------------------------------- */
  
  /* shortcuts */
  t = 0.5 * (lambda-1.);
  s = 0.25 * omega;
  
  /* mode = location of maximum of sqrt(f(x)) */
  xm = _gig_mode(lambda, omega);
  
  /* normalization constant: c = log(sqrt(f(xm))) */
  nc = t*log(xm) - s*(xm + 1./xm);
  
  /* location of minimum and maximum of (1/x)*sqrt(f(1/x+m)):  */
  
  /* compute coeffients of cubic equation y^3+a*y^2+b*y+c=0 */
  a = -(2.*(lambda+1.)/omega + xm);       /* < 0 */
  b = (2.*(lambda-1.)*xm/omega - 1.);
  c = xm;
  
  /* we need the roots in (0,xm) and (xm,inf) */
  
  /* substitute y=z-a/3 for depressed cubic equation z^3+p*z+q=0 */
  p = b - a*a/3.;
  q = (2.*a*a*a)/27. - (a*b)/3. + c;
  
  /* use Cardano's rule */
  fi = acos(-q/(2.*sqrt(-(p*p*p)/27.)));
  fak = 2.*sqrt(-p/3.);
  y1 = fak * cos(fi/3.) - a/3.;
  y2 = fak * cos(fi/3. + 4./3.*M_PI) - a/3.;
  
  /* boundaries of minmal bounding rectangle:                  */
  /* we us the "normalized" density f(x) / f(xm). hence        */
  /* upper boundary: vmax = 1.                                 */
  /* left hand boundary: uminus = (y2-xm) * sqrt(f(y2)) / sqrt(f(xm)) */
  /* right hand boundary: uplus = (y1-xm) * sqrt(f(y1)) / sqrt(f(xm)) */
  uplus  = (y1-xm) * exp(t*log(y1) - s*(y1 + 1./y1) - nc);
  uminus = (y2-xm) * exp(t*log(y2) - s*(y2 + 1./y2) - nc);
  
  /* -- Generate sample ---------------------------------------------------- */
  
  for (i=0; i<n; i++) {
    do {
      ++count;
      U = uminus + unif_rand() * (uplus - uminus);    /* U(u-,u+)  */
  V = unif_rand();                                /* U(0,vmax) */
  X = U/V + xm;
    }                                         /* Acceptance/Rejection */
  while ((X <= 0.) || ((log(V)) > (t*log(X) - s*(X + 1./X) - nc)));
    
    /* store random point */
    res[i] = (lambda_old < 0.) ? (alpha / X) : (alpha * X);
  }
  
  /* -- End ---------------------------------------------------------------- */
  
  return;
  } /* end of _rgig_ROU_shift_alt() */
  
  void _rgig_newapproach1 (double *res, int n, double lambda, double lambda_old, double omega, double alpha)
  /*---------------------------------------------------------------------------*/
  /* Type 4:                                                                   */
  /* New approach, constant hat in log-concave part.                           */
  /* Draw sample from GIG distribution.                                        */
  /*                                                                           */
  /* Case: 0 < lambda < 1, 0 < omega < 1                                       */
  /*                                                                           */
  /* Parameters:                                                               */
  /*   n ....... sample size (positive integer)                                */
  /*   lambda .. parameter for distribution                                    */
  /*   omega ... parameter for distribution                                    */
  /*                                                                           */
  /* Return:                                                                   */
  /*   random sample of size 'n'                                               */
  /*---------------------------------------------------------------------------*/
  {
    /* parameters for hat function */
    double A[3], Atot;  /* area below hat */
    double k0;          /* maximum of PDF */
    double k1, k2;      /* multiplicative constant */
    
    double xm;          /* location of mode */
    double x0;          /* splitting point T-concave / T-convex */
    double a;           /* auxiliary variable */
    
    double U, V, X;     /* random numbers */
    double hx;          /* hat at X */
    
    int i;              /* loop variable (number of generated random variables) */
    int count = 0;      /* counter for total number of iterations */
    
    /* -- Check arguments ---------------------------------------------------- */
    
    if (lambda >= 1. || omega >1.)
      Rcpp::stop ("invalid parameters");
    
    /* -- Setup -------------------------------------------------------------- */
    
    /* mode = location of maximum of sqrt(f(x)) */
    xm = _gig_mode(lambda, omega);
    
    /* splitting point */
    x0 = omega/(1.-lambda);
    
    /* domain [0, x_0] */
    k0 = exp((lambda-1.)*log(xm) - 0.5*omega*(xm + 1./xm));     /* = f(xm) */
    A[0] = k0 * x0;
    
    /* domain [x_0, Infinity] */
    if (x0 >= 2./omega) {
      k1 = 0.;
      A[1] = 0.;
      k2 = pow(x0, lambda-1.);
      A[2] = k2 * 2. * exp(-omega*x0/2.)/omega;
    }
    
    else {
      /* domain [x_0, 2/omega] */
      k1 = exp(-omega);
      A[1] = (lambda == 0.) 
        ? k1 * log(2./(omega*omega))
          : k1 / lambda * ( pow(2./omega, lambda) - pow(x0, lambda) );
      
      /* domain [2/omega, Infinity] */
      k2 = pow(2/omega, lambda-1.);
      A[2] = k2 * 2 * exp(-1.)/omega;
    }
    
    /* total area */
    Atot = A[0] + A[1] + A[2];
    
    /* -- Generate sample ---------------------------------------------------- */
    
    for (i=0; i<n; i++) {
      do {
        ++count;
        
        /* get uniform random number */
        V = Atot * unif_rand();
        
        do {
          
          /* domain [0, x_0] */
          if (V <= A[0]) {
            X = x0 * V / A[0];
            hx = k0;
            break;
          }
          
          /* domain [x_0, 2/omega] */
          V -= A[0];
          if (V <= A[1]) {
            if (lambda == 0.) {
              X = omega * exp(exp(omega)*V);
              hx = k1 / X;
            }
            else {
              X = pow(pow(x0, lambda) + (lambda / k1 * V), 1./lambda);
              hx = k1 * pow(X, lambda-1.);
            }
            break;
          }
          
          /* domain [max(x0,2/omega), Infinity] */
          V -= A[1];
          a = (x0 > 2./omega) ? x0 : 2./omega;
          X = -2./omega * log(exp(-omega/2. * a) - omega/(2.*k2) * V);
          hx = k2 * exp(-omega/2. * X);
          break;
          
        } while(0);
        
        /* accept or reject */
        U = unif_rand() * hx;
        
        if (log(U) <= (lambda-1.) * log(X) - omega/2. * (X+1./X)) {
          /* store random point */
          res[i] = (lambda_old < 0.) ? (alpha / X) : (alpha * X);
          break;
        }
      } while(1);
      
    }
    
    /* -- End ---------------------------------------------------------------- */
    
    return;
  } /* end of _rgig_newapproach1() */
    
    void _rgig_ROU_noshift (double *res, int n, double lambda, double lambda_old, double omega, double alpha)
  /*---------------------------------------------------------------------------*/
  /* Tpye 1:                                                                   */
  /* Ratio-of-uniforms without shift.                                          */
  /*   Dagpunar (1988), Sect.~4.6.2                                            */
  /*   Lehner (1989)                                                           */
  /*---------------------------------------------------------------------------*/
    {
      double xm, nc;     /* location of mode; c=log(f(xm)) normalization constant */
  double ym, um;     /* location of maximum of x*sqrt(f(x)); umax of MBR */
  double s, t;       /* auxiliary variables */
  double U, V, X;    /* random variables */
  
  int i;             /* loop variable (number of generated random variables) */
  int count = 0;     /* counter for total number of iterations */
  
  /* -- Setup -------------------------------------------------------------- */
  
  /* shortcuts */
  t = 0.5 * (lambda-1.);
  s = 0.25 * omega;
  
  /* mode = location of maximum of sqrt(f(x)) */
  xm = _gig_mode(lambda, omega);
  
  /* normalization constant: c = log(sqrt(f(xm))) */
  nc = t*log(xm) - s*(xm + 1./xm);
  
  /* location of maximum of x*sqrt(f(x)):           */
  /* we need the positive root of                   */
  /*    omega/2*y^2 - (lambda+1)*y - omega/2 = 0    */
  ym = ((lambda+1.) + sqrt((lambda+1.)*(lambda+1.) + omega*omega))/omega;
  
  /* boundaries of minmal bounding rectangle:                   */
  /* we us the "normalized" density f(x) / f(xm). hence         */
  /* upper boundary: vmax = 1.                                  */
  /* left hand boundary: umin = 0.                              */
  /* right hand boundary: umax = ym * sqrt(f(ym)) / sqrt(f(xm)) */
  um = exp(0.5*(lambda+1.)*log(ym) - s*(ym + 1./ym) - nc);
  
  /* -- Generate sample ---------------------------------------------------- */
  
  for (i=0; i<n; i++) {
    do {
      ++count;
      U = um * unif_rand();        /* U(0,umax) */
  V = unif_rand();             /* U(0,vmax) */
  X = U/V;
    }                              /* Acceptance/Rejection */
  while (((log(V)) > (t*log(X) - s*(X + 1./X) - nc)));
    
    /* store random point */
    res[i] = (lambda_old < 0.) ? (alpha / X) : (alpha * X);
  }
  
  /* -- End ---------------------------------------------------------------- */
  
  return;
    } /* end of _rgig_ROU_noshift() */
  
  
  
#define ZTOL (DOUBLE_EPS*10.0)
  // [[Rcpp::export]]
  /*---------------------------------------------------------------------------*/
  double rgig( double lambda, double chi, double psi)
  {
    double omega, alpha;     /* parameters of standard distribution */
  // SEXP sexp_res;           /* results */
  double *res;
  int i;
  int n=1;
  
  /* check sample size */
  if (n<=0) {
    Rcpp::stop("sample size 'n' must be positive integer.");
  }
  
  /* check GIG parameters: */
  if ( !(R_FINITE(lambda) && R_FINITE(chi) && R_FINITE(psi)) ||
  (chi <  0. || psi < 0)      || 
  (chi == 0. && lambda <= 0.) ||
  (psi == 0. && lambda >= 0.) ) {
    // cout<<"lambda="<<lambda<<", chi="<<chi<<", psi"=psi<<endl;
    printf("lambda= %lf, chi=%lf, psi= %lf\n",lambda, chi,psi);
    Rcpp::stop("invalid parameters for GIG distribution!!");
  }
  
  /* allocate array for random sample */
  // PROTECT(sexp_res = NEW_NUMERIC(n));
  res = (double *)malloc(n*sizeof(double)) ;
  
  if (chi < ZTOL) { 
    /* special cases which are basically Gamma and Inverse Gamma distribution */
    if (lambda > 0.0) {
      for (i=0; i<n; i++) res[i] = R::rgamma(lambda, 2.0/psi); 
    }
    else {
      for (i=0; i<n; i++) res[i] = 1.0/R::rgamma(-lambda, 2.0/psi); 
    }    
  }
  
  else if (psi < ZTOL) {
    /* special cases which are basically Gamma and Inverse Gamma distribution */
    if (lambda > 0.0) {
      for (i=0; i<n; i++) res[i] = 1.0/R::rgamma(lambda, 2.0/chi); 
    }
    else {
      for (i=0; i<n; i++) res[i] = R::rgamma(-lambda, 2.0/chi); 
    }    
    
  }
  
  else {
    double lambda_old = lambda;
    if (lambda < 0.) lambda = -lambda;
    alpha = sqrt(chi/psi);
    omega = sqrt(psi*chi);
    
    /* run generator */
    do {
      if (lambda > 2. || omega > 3.) {
        /* Ratio-of-uniforms with shift by 'mode', alternative implementation */
        _rgig_ROU_shift_alt(res, n, lambda, lambda_old, omega, alpha);
        break;
      }
      
      if (lambda >= 1.-2.25*omega*omega || omega > 0.2) {
        /* Ratio-of-uniforms without shift */
        _rgig_ROU_noshift(res, n, lambda, lambda_old, omega, alpha);
        break;
      }
      
      if (lambda >= 0. && omega > 0.) {
        /* New approach, constant hat in log-concave part. */
        _rgig_newapproach1(res, n, lambda, lambda_old, omega, alpha);
        break;
      }
      
      /* else */
      Rcpp::stop("parameters must satisfy lambda>=0 and omega>0.");
      
    } while (0);
  }
  
  /* return result */
  // UNPROTECT(1);
  double ret=res[0];
  free(res);
  return ret;
  } /* end of do_rgig() */
  
  
  double log_sum_exp(const arma::vec &x) {
    double maxVal= x.max();
    
    double sum_exp=sum(exp(x-maxVal));
    return log(sum_exp)+maxVal ;
  }
  
  inline double multiplier_fn(double niw_nu, double niw_kap, int k){
    return (niw_kap +1) / ( niw_kap* (niw_nu-k+1 )  );
  }  
  
  // [[Rcpp::export]]
  inline double log_t_density(const double nu,const  vec &x,const vec &mu,const mat &lower_chol){
    int k=x.n_elem;
    double det_sig_half= sum(log(lower_chol.diag() ) );
    vec resid=solve(trimatl ( lower_chol ) , x - mu );
    
    double quad_form=dot(resid,resid)/nu;
    // gsl_sf_lnpoch(a,b)=log (gamma(a+b)/gamma(a)  )
    double density =gsl_sf_lnpoch(nu/2 , ((double)k)/2)-// lgamma((nu+k)/2 ) -lgamma(nu/2) - 
      (k* log( (datum::pi)*nu) + (nu+k)*log1p(quad_form) )/2 -det_sig_half;
    
    return density;
  }
  
  // [[Rcpp::export]]
  inline double log_t_density_empty(const double nu,const double kap,const double diag_psi_iw, const vec &x){
    int k=x.n_elem;
    double t1ppp=(kap+1)/(kap*nu);
    /* adjustment_factor.diag()+= (diag_psi_iw* t1ppp);
     mat lower_chol=chol(adjustment_factor,"lower");*/
    
    double adjustment_factor = (diag_psi_iw* t1ppp);
    
    double det_sig_half= k* log(adjustment_factor)/2;  //sum(log(lower_chol.diag() ) );
    // vec resid= solve(trimatl ( lower_chol ) , x  );
    
    double quad_form= dot(x,x)/  (adjustment_factor*nu); 
    // gsl_sf_lnpoch(a,b)=log (gamma(a+b)/gamma(a)  )
    double density =gsl_sf_lnpoch(nu/2 , ((double)k)/2)-  //lgamma((nu+k)/2 ) -lgamma(nu/2) - 
      (k* log((datum::pi)*nu) + (nu+k)*log1p(quad_form ) )/2 -det_sig_half;
    // cout<<"t1ppp="<<t1ppp<<" det_sig_half"<<det_sig_half<<" quad_form="<<quad_form<<" density="<<density;
    return density;
  }
  
  uvec std_setdiff(arma::uvec a, arma::uvec b) {
    
    // std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
    // std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
    a=sort(a);b=sort(b);
    
    std::vector<unsigned> out;
    
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::inserter(out, out.end()));
    
    return arma::conv_to<arma::uvec>::from(out);
  }
  
  claunch_list  gibbs_scan(int nscan, uvec claunch, const mat &eta,
                           const double niw_nu, const double niw_kap, const double diag_psi_iw){
    int k=eta.n_cols, n=eta.n_rows,nmix=2,j;
    double log_q=0,t1ppp;
    
    /////set the values of claunch to 0 and 1 ///
    uvec original_labels= claunch(span(0,1)); //unique(claunch);
    if(original_labels(0)==original_labels(1)){
      original_labels.print("original_labels");
      Rcpp::stop("original_labels(0)=original_labels(1)");
    }
    
    claunch.transform( [&original_labels](uword val) { return  (val==original_labels(0) )? 0:1 ; } );
    /////////////////////////////////////////////
    uvec nj_val(nmix);  //y number corresponding to each cluster
    vec log_nj_val(nmix),niw_kap_post(nmix), df_post(nmix) ,niw_nu_post(nmix) ;
    mat  sum_of_samples=zeros<mat>(k,nmix),marginal_mu=zeros<mat>(k,nmix);
    cube sum_of_squares=zeros<cube>(k,k,nmix),   chol_marginal_delta=zeros<cube>(k,k,nmix), marginal_delta=zeros<cube>(k,k,nmix);
    uvec *inds_eq_j;inds_eq_j=new uvec[nmix ];
    ivec d(nmix); ////multinomial indicator for each sample
    vec probs,log_probs(nmix), log_probs_gibbs(nmix); ////assignment probability for each sample
    
    
    //////Initialize the Gibbs scan /////
    for(int j=0;j<nmix;j++){
      inds_eq_j[j] =find(claunch==claunch(j));
      // cout<<"n_inds="<<j<<" is "<<(inds_eq_j[j]) .n_elem<<endl;
      nj_val(j)=(inds_eq_j[j]) .n_elem;
      log_nj_val(j)=log(nj_val(j));
      
      niw_nu_post(j)=niw_nu+nj_val(j); niw_kap_post(j)=niw_kap+nj_val(j);
      df_post(j)=niw_nu_post(j)-k+1;
      
      sum_of_squares.slice(j)= crossprod(eta.rows(inds_eq_j[j]));
      sum_of_samples .col(j)=sum ( eta.rows(inds_eq_j[j]) ,0).t(); //1 in sum/mean implies rowmeans
      
      marginal_mu.col(j) =sum_of_samples.col(j)/ niw_kap_post(j);
      t1ppp=multiplier_fn(niw_nu_post(j), niw_kap_post(j),k);
      
      marginal_delta.slice(j) =symmatu( sum_of_squares.slice(j) - tcrossprod(sum_of_samples .col(j))/niw_kap_post(j) );
      (marginal_delta.slice(j)).diag() += diag_psi_iw;
      chol_marginal_delta.slice(j)=sqrt(t1ppp )*chol(marginal_delta.slice(j), "lower");
    }
    /////////////////////////////////////
    
    double  log_probs_max;//,normal_exp; 
    
    for(int scan_iter=0;scan_iter<nscan;++scan_iter)
      for(int jj=2;jj<n;++jj){
        double dens,cluster_prob;
        
        uword current_ind=claunch(jj); //find the current index of data point jj
        for(j=0;j<nmix;++j){
          if( j!=current_ind){ //checking if jj-th data point is in j-th cluster
            dens= log_t_density(df_post(j) ,  (eta.row(jj)).t()
                                  ,marginal_mu.col(j), chol_marginal_delta.slice(j));
            // cout<<"at j="<<j<<" log_density="<<dens<<"\t clust prob="<<cluster_prob<<endl;
            cluster_prob=log_nj_val(j); 
            
            log_probs(j)=  dens +cluster_prob;
          }
        }
        //calculating allocation probabilities
        
        j=current_ind;
        mat cross_prod=crossprod(eta.row(jj));
        vec tmp_marginal_mu(k), tmp_sum_of_samples(k);
        mat tmp_sum_of_sq=zeros<mat>(k,k), tmp_chol_marginal_delta=zeros<mat>(k,k), tmp_marginal_delta=zeros<mat>(k,k);
        
        if(nj_val(j)<2 ){
          cout<<"current_ind= "<<j<<"nj_val(j)= "<<nj_val(j)<<endl;
          Rcpp::stop("nj_val(current_ind)<2 in launch fn ");
        }
        tmp_sum_of_samples=sum_of_samples .col(j) - (eta.row(jj)).t() ;
        tmp_marginal_mu=(tmp_sum_of_samples ) /(niw_kap_post(j)-1) ;
        
        tmp_sum_of_sq=symmatu( sum_of_squares.slice(j)- cross_prod );
        
        
        tmp_marginal_delta=symmatu( tmp_sum_of_sq -  tcrossprod(tmp_sum_of_samples)/(niw_kap_post(j)-1 ) );
        // cout <<"tmp_marginal_delta.diag()="<< tmp_marginal_delta.diag() << endl;
        tmp_marginal_delta.diag() += diag_psi_iw;  
        
        t1ppp= multiplier_fn( niw_nu_post(j)-1 , niw_kap_post(j)-1, k);
        tmp_chol_marginal_delta=  chol(t1ppp* tmp_marginal_delta ,"lower") ;
        
        dens= log_t_density(df_post(j)-1 ,  (eta.row(jj)).t(),tmp_marginal_mu, tmp_chol_marginal_delta );
        cluster_prob=log( nj_val(j)-1 );
        
        log_probs(j)=  dens +cluster_prob;
        
        // cout<<"at current_ind="<<j<<" log_density="<<dens<<"\t clust prob="<<cluster_prob<<endl;
        // log_probs.print("log_probs=");
        
        double log_DEN=log_sum_exp(log_probs);
        
        log_probs_gibbs=log_probs-log_DEN;
        probs= normalise(exp(log_probs-log_DEN) ,1);
        
        // log_probs.print("log_probs: ");
        // probs.print("probs: ");
        
        if(   gsl_fcmp(sum(probs),1.0,1e-5) ){
          // cout<< "At jj="<<jj<<"sum_prob is 0"<<endl;
          log_probs_max= max(log_probs);
          log_probs-=log_probs_max;
          
          probs=normalise(exp(log_probs) ,1);
        }
        
        ////check if sum of the allocation probbilities is zero
        if(sum(probs)==0)  {
          Rcpp::stop("sum(probs)=0 inside launch fn");
        }
        ///////////////////////////////////////////////////////
        
        R::rmultinom(1, probs.begin(), nmix, d.begin());
        uvec dd=find(d==1,1,"first");
        claunch(jj)=dd(  0);
        if(scan_iter== nscan-1)
          log_q+= log_probs_gibbs(claunch(jj) ) ;
        
        //updating cluster occupancies
        if(claunch(jj)!=current_ind){
          // cout<<"yes"<<endl;
          //**** update the source cluster parameters ****//
          j=current_ind;
          log_nj_val(j ) = log( --nj_val(j) );
          niw_kap_post(j)--; niw_nu_post(j)--; df_post(j)--;
          
          sum_of_samples.col(j)=tmp_sum_of_samples;
          sum_of_squares.slice(j)=tmp_sum_of_sq;
          marginal_mu.col(j)=tmp_marginal_mu;
          
          marginal_delta.slice(j)=tmp_marginal_delta;
          chol_marginal_delta.slice(j)=tmp_chol_marginal_delta;
          // tmpcube.slice(j)=tmp_chol_marginal_delta;
          
          ///////////////////////////////////////////////////////
          
          //**** update the destination cluster parameters ****//
          j=claunch(jj);
          log_nj_val(j ) = log( ++nj_val(j) );
          niw_kap_post(j)++; niw_nu_post(j)++; df_post(j)++;
          
          sum_of_samples.col(j)+=  (eta.row(jj)).t() ;
          sum_of_squares.slice(j) += cross_prod ;
          
          marginal_mu.col(j) =(sum_of_samples.col(j) ) /(niw_kap_post(j)) ;
          
          marginal_delta.slice(j) =symmatu( sum_of_squares.slice(j) - tcrossprod(sum_of_samples .col(j))/niw_kap_post(j) );
          (marginal_delta.slice(j)).diag() += diag_psi_iw;
          t1ppp= multiplier_fn( niw_nu_post(j) , niw_kap_post(j), k);
          
          chol_marginal_delta.slice(j) =  chol(t1ppp*marginal_delta.slice(j) ,"lower") ;
          // tmpcube.slice(j)=chol_marginal_delta.slice(j);
          ///////////////////////////////////////////////////////
        }
      }
      
      
      /////set the values of claunch to original labs ///
      claunch.transform( [&original_labels](uword val) { return  (val==0 )? original_labels(0):original_labels(1) ; } );
    /////////////////////////////////////////////
    
    claunch_list out;
    (out.marginal_density).set_size(nmix);
    out.nj_val=nj_val;
    out.marginal_delta=marginal_delta;
    out.chol_marginal_delta  =chol_marginal_delta;
    
    for(j=0;j<nmix;++j){
      t1ppp= multiplier_fn( niw_nu_post(j) , niw_kap_post(j), k);
      
      (out.marginal_density)(j)= log_marg_dens ( (out.chol_marginal_delta).slice(j),out.nj_val(j), t1ppp
                                                   , niw_nu, niw_kap, diag_psi_iw   );
    }
    
    out.log_q=log_q;
    out.claunch=claunch;
    
    delete[] inds_eq_j;
    return out;
  }
  
  claunch_list  get_cmerge(const claunch_list &out, uvec c_orig, const mat &eta,
                           const double niw_nu, const double niw_kap, const double diag_psi_iw){
    int j;
    for(j=0;j<2;++j)
      if(c_orig(j)!= out.claunch(j) ){
        cout<<"At j="<<j<< "c_orig(j)="<<c_orig(j)<< "out.claunch(j)="<< out.claunch(j);
        Rcpp::stop("In get_cmerge fn!");
      }
      
      
      int k=eta.n_cols, n=eta.n_rows,nmix=2;
      double log_q=0,t1ppp;
      
      uvec claunch=out.claunch;
      
      /////set the values of claunch to 0 and 1 ///
      uvec original_labels= c_orig(span(0,1)); // unique(claunch);
      if(original_labels(0)==original_labels(1)){
        original_labels.print("original_labels in get_cmerge");
        Rcpp::stop("original_labels(0)=original_labels(1)");
      }
      
      
      claunch.transform( [&original_labels](uword val) { return  (val==original_labels(0) )? 0:1 ; } );
      c_orig.transform( [&original_labels](uword val) { return  (val==original_labels(0) )? 0:1 ; } );
      
      // cout<<"Flag get_cmerge 2"<<endl;
      /////////////////////////////////////////////
      uvec nj_val=out.nj_val;  //occupancu number corresponding to each cluster
      vec log_nj_val(nmix),niw_kap_post(nmix), df_post(nmix) ,niw_nu_post(nmix) ;
      mat  sum_of_samples=zeros<mat>(k,nmix),marginal_mu=zeros<mat>(k,nmix);
      cube sum_of_squares=zeros<cube>(k,k,nmix),   chol_marginal_delta=zeros<cube>(k,k,nmix), marginal_delta=zeros<cube>(k,k,nmix);
      uvec *inds_eq_j;inds_eq_j=new uvec[nmix ];
      vec log_probs(nmix); ////assignment probability for each sample
      
      //////Initialize the Gibbs scan /////
      for( j=0;j<nmix;j++){
        inds_eq_j[j] =find(claunch==claunch(j));
        if(nj_val(j)!=(inds_eq_j[j]) .n_elem){
          cout<<" nj_val(j)!=(inds_eq_j[j]) .n_elem at j="<<j<<"in get_cmerge function"<<endl;
          Rcpp::stop("");
        }
        log_nj_val(j)=log(nj_val(j));
        
        niw_nu_post(j)=niw_nu+nj_val(j); niw_kap_post(j)=niw_kap+nj_val(j);
        df_post(j)=niw_nu_post(j)-k+1;
        
        sum_of_squares.slice(j)= crossprod(eta.rows(inds_eq_j[j]));
        sum_of_samples .col(j)=sum ( eta.rows(inds_eq_j[j]) ,0).t(); //1 in sum/mean implies rowmeans
        
        marginal_mu.col(j) =sum_of_samples.col(j)/ niw_kap_post(j);
        t1ppp=multiplier_fn(niw_nu_post(j), niw_kap_post(j),k);
      }
      marginal_delta =out.marginal_delta;
      chol_marginal_delta=out.chol_marginal_delta;
      /////////////////////////////////////
      
      for(int jj=2;jj<n;++jj){
        double dens,cluster_prob;
        
        uword current_ind=claunch(jj); //find the current cluster id of data point jj
        for(j=0;j<nmix;++j){
          if( j!=current_ind){ //checking if jj-th data point is in j-th cluster
            dens= log_t_density(df_post(j) ,  (eta.row(jj)).t(), marginal_mu.col(j), chol_marginal_delta.slice(j));
            // cout<<"at j="<<j<<" log_density="<<dens<<"\t clust prob="<<cluster_prob<<endl;
            cluster_prob=log_nj_val(j); 
            
            log_probs(j)=  dens +cluster_prob;
          }
        }
        //calculating allocation probabilities
        
        j=current_ind;
        mat cross_prod=crossprod(eta.row(jj));
        vec tmp_marginal_mu(k), tmp_sum_of_samples(k);
        mat tmp_sum_of_sq=zeros<mat>(k,k), tmp_chol_marginal_delta=zeros<mat>(k,k), tmp_marginal_delta=zeros<mat>(k,k);
        
        if(nj_val(j)<2 ){
          cout<<"current_ind= "<<j<<"nj_val(j)= "<<nj_val(j)<<endl;
          Rcpp::stop("nj_val(current_ind)<2 in get_cmerge ");
        }   
        tmp_sum_of_samples=sum_of_samples .col(j) - (eta.row(jj)).t() ;
        tmp_marginal_mu=(tmp_sum_of_samples ) /(niw_kap_post(j)-1) ;
        
        tmp_sum_of_sq=symmatu( sum_of_squares.slice(j)- cross_prod );
        
        // cout<<"eig_sym( tmp_sum_of_sq )=";
        // eig_sym( tmp_sum_of_sq ).print();
        
        // cout<<"jj="<<jj<<endl;
        // cout<<"Flag 1 current_ind="<<current_ind<<"nj_val-1="<<nj_val(current_ind) -1<<endl;
        
        
        tmp_marginal_delta=symmatu( tmp_sum_of_sq -  tcrossprod(tmp_sum_of_samples)/(niw_kap_post(j)-1 ) );
        // cout <<"tmp_marginal_delta.diag()="<< tmp_marginal_delta.diag() << endl;
        tmp_marginal_delta.diag() += diag_psi_iw;  
        
        t1ppp= multiplier_fn( niw_nu_post(j)-1 , niw_kap_post(j)-1, k);
        tmp_chol_marginal_delta=  chol(t1ppp* tmp_marginal_delta ,"lower") ;
        
        dens= log_t_density(df_post(j)-1 ,  (eta.row(jj)).t(),tmp_marginal_mu, tmp_chol_marginal_delta );
        cluster_prob=log( nj_val(j)-1 );
        
        log_probs(j)=  dens +cluster_prob;
        
        // cout<<"at current_ind="<<j<<" log_density="<<dens<<"\t clust prob="<<cluster_prob<<endl;
        // log_probs.print("log_probs=");
        
        long double log_DEN=log_sum_exp(log_probs);
        
        log_probs-=log_DEN;
        
        claunch(jj)=c_orig(jj);
        
        log_q+= log_probs(claunch(jj) ) ;
        
        //updating cluster occupancies
        if(claunch(jj)!=current_ind){
          //**** update the source cluster parameters ****//
          j=current_ind;
          log_nj_val(j ) = log( --nj_val(j) );
          niw_kap_post(j)--; niw_nu_post(j)--; df_post(j)--;
          
          sum_of_samples.col(j)=tmp_sum_of_samples;
          sum_of_squares.slice(j)=tmp_sum_of_sq;
          marginal_mu.col(j)=tmp_marginal_mu;
          
          marginal_delta.slice(j)=tmp_marginal_delta;
          chol_marginal_delta.slice(j)=tmp_chol_marginal_delta;
          // tmpcube.slice(j)=tmp_chol_marginal_delta;
          
          ///////////////////////////////////////////////////////
          
          //**** update the destination cluster parameters ****//
          j=claunch(jj);
          log_nj_val(j ) = log( ++nj_val(j) );
          niw_kap_post(j)++; niw_nu_post(j)++; df_post(j)++;
          
          sum_of_samples.col(j)+=  (eta.row(jj)).t() ;
          sum_of_squares.slice(j) += cross_prod ;
          
          marginal_mu.col(j) =(sum_of_samples.col(j) ) /(niw_kap_post(j)) ;
          
          marginal_delta.slice(j) =symmatu( sum_of_squares.slice(j) - tcrossprod(sum_of_samples .col(j))/niw_kap_post(j) );
          (marginal_delta.slice(j)).diag() += diag_psi_iw;
          t1ppp= multiplier_fn( niw_nu_post(j) , niw_kap_post(j), k);
          
          chol_marginal_delta.slice(j) =  chol(t1ppp*marginal_delta.slice(j) ,"lower") ;
          // tmpcube.slice(j)=chol_marginal_delta.slice(j);
          ///////////////////////////////////////////////////////
        }
      }
      
      
      /////out.c^launch=c^merge. set the values of claunch to original c_j ///
      claunch.fill(original_labels( 1) );
      /////////////////////////////////////////////
      
      claunch_list out_merge;
      (out_merge.nj_val).set_size(1);
      out_merge.nj_val(0) =sum(nj_val);
      
      
      out_merge.log_q=log_q;
      out_merge.claunch=claunch;
      
      delete[] inds_eq_j;
      
      return out_merge;
  }
  
  
  claunch_list get_claunch(const uvec &clust_id,const uvec &nj_val_full,const mat &eta_all, int nscan, const double niw_nu, const double niw_kap, const double diag_psi_iw){
    unsigned nsamp=clust_id.n_elem;
    uvec rand_samps= randperm( nsamp, 2 ); //indices of the randomly chosen i,j
    // rand_samps.print("rand_samps");
    
    // (clust_id.elem(rand_samps)).print("clust_id.elem(rand_samps)");
    
    uvec S,all_samps;
    uvec claunch_S, claunch_ij(2); // cluster ids of the samples in S
    
    if( clust_id( rand_samps(0) ) == clust_id( rand_samps(1) )){// c_i = c_j
      all_samps= find(clust_id== clust_id(rand_samps(1)) ); // S \union {i,j}
      S=std_setdiff(all_samps,rand_samps); // the set S
      
      // cout<<"split S.n_elem ="<<S.n_elem<<" all_samps.n_elem= "<<all_samps.n_elem<<endl;
      uvec empty_inds=find(nj_val_full==0,1,"first");
      claunch_ij(0)=empty_inds(0);// assign c_i^launch= some empty cluster
      claunch_ij(1)=clust_id( rand_samps(0) ); //c_j^launch=c_j
      ++nscan;
    }
    
    else{//c_i \neq c_j
      all_samps=vectorise(join_cols(find(clust_id== clust_id(rand_samps(0)) ), 
                                    find(clust_id== clust_id(rand_samps(1)) ))  ); // S \union {i,j}
      
      // cout<<"all_samps.n_elem= "<<all_samps.n_elem<<endl;
      S=std_setdiff(all_samps,rand_samps); // the set S
      
      // cout<<"merge S.n_elem ="<<S.n_elem<<" all_samps.n_elem= "<<all_samps.n_elem<<endl;
      claunch_ij=clust_id( rand_samps);
    }
    
    claunch_S.set_size(S.n_elem);
    if(S.n_elem+2 != all_samps.n_elem){
      cout<<"S.n_elem ="<<S.n_elem<<" all_samps.n_elem= "<<all_samps.n_elem<<endl;
      Rcpp::stop("S.n_elem+2 != all_samps.n_elem in get_claunch function!");
    }
    
    claunch_S .transform( [&claunch_ij](uword val) { return  (randu()<0.5)? claunch_ij(0):claunch_ij(1); } ); //randomly allocate c_k=  c_i or c_j
    uvec claunch=join_cols(claunch_ij, claunch_S ), all_inds=join_cols(rand_samps,S);
    
    // (claunch(span(0,1))).print("claunch(span(0,1))");
    
    mat eta= eta_all.rows(all_inds);// join_cols( eta_all.rows(rand_samps), eta_all.rows(S) );
    claunch_list out= gibbs_scan(nscan, claunch, eta,niw_nu, niw_kap, diag_psi_iw);
    
    out.inds=all_inds;
    
    return out;
  }
  
  void update_loading_params( mat &Lambda,mat &T_phi, mat &Plam,const double a){
    const unsigned p=Lambda.n_rows, k=Lambda.n_cols;
    // --- UPDATE PSI --- //
    mat abs_theta=abs(Lambda);
    mat psi_inv= (T_phi/   abs_theta) ;
    psi_inv.transform( [](double val) { return ( rig(val) ); } );
    
    
    // --- UPDATE PHI --- //
    T_phi=abs_theta;
    T_phi.transform( [&a](double val) { double tmp=rgig(a-1, 2* val, 1); return(  GSL_MAX_DBL(tmp,9e-6)) ; } );
    
    if(T_phi.has_nan()){
      T_phi.print("T_phi :");
      // abs_theta.print("abs_theta :");
      // uvec tmppp=abs_theta.elem( find_nonfinite( T_phi ) );
      // cout<<"corresponding abs_theta"<<  tmppp <<"at that rgig ";
      cout<<"corresponding abs_theta"<<  abs_theta.elem( find_nonfinite( T_phi ) ) <<endl;
      // for(int h=0; h<tmppp.size();++h )
      //   cout<<rgig(a-1, 2* tmppp(h), 1)<<endl;
      
      cout<<"has inf in phi "<<T_phi.has_inf()<<endl;
      Rcpp::stop("phi has nan!!");
    }
    
    
    // --- UPDATE PLAM --- //
    Plam= ( psi_inv/square(T_phi));
    Plam.transform( [](double val) { return ( val/(1+tolr*val) ); } );
    // cout<<"Plam updated; accu_plam"<<accu(Plam) <<endl;
    if(Plam.has_nan()){
      cout<<"corresponding psi_inv"<<  psi_inv.elem( find_nonfinite( Plam ) ) <<endl;
      // cout<<"corresponding T_phi"<<  T_phi.elem( find_nonfinite( Plam ) ) <<endl;
      Rcpp::stop("plam has nan!!");
    }
  }
    
    double update_dir_prec(double alpha, double a, double b,unsigned n, unsigned k){//n=total number of data points; k= # clusters
      double phi=R::rbeta(alpha+1,(double) n);
      double gam1=a+k, gam2=b- log(phi);
      double pi=(gam1-1)/(n* gam2 );
      return (log(randu()) < log(pi)-log1p(pi)  ) ? (randg(  distr_param(gam1,1/gam2) )) : (randg(  distr_param(gam1-1,1/gam2) )) ;
    }
    
// [[Rcpp::export]]
umat DL_mixture(const double a_dir,const double b_dir,const double diag_psi_iw,const double niw_kap, const double niw_nu,
                       double as, double bs, double a,
                       int nrun, int burn, int thin, unsigned nstep, double prob,
                       arma::mat Lambda, arma::mat eta,
                       arma::mat Y,
                       uvec del,
                       bool dofactor=1){
  // arma_rng::set_seed(100);
  int approx=0;
  int acceptance=0,mh_count=0;
  int n=Y.n_rows,p=Y.n_cols,k=eta.n_cols;
  unsigned nmix=n;
  
  double t1ppp;
  
  arma::mat psijh(p,k);
  arma::mat Plam;
  arma::vec ps(p,fill::ones); //Diagonal variance parameters 
  
  mat psi = randg<mat>( p, k, distr_param(1.0,0.5) );
  mat psi_inv=pow(psi,-1.0);
  
  mat phi=randg<mat>( p, k, distr_param(a,1.0) );
  // double T_sum=accu(phi);
  mat T_phi=phi;
  // phi/=T_sum;
  // double tau=randg<double>(distr_param(p*k*a ,0.5)   );
  
  // --- Initiate PLAM --- //
  Plam=1/( 1+ square (T_phi) );
  
  mat theta=randn<mat>( p, k);
  theta /= sqrt(Plam);
  
  
  int j;
  
  umat alloc_var_mat(std::floor((nrun+burn)/thin), n);
  
  // --- initialise loop objects --- //
  mat Lmsg, noise,  eta2, Llamt,  Ytil, Meta;
  
  // mat adjustment_factor=zeros<mat>(k,k);
  cube Delta(k,k,nmix),Delta_inv(k,k,nmix),chol_Delta(k,k,nmix);//each slice is the cov matix of a mixture
  
  ivec d; ////multinomial indicator for each sample
  vec probs(nmix),log_probs(nmix); ////assignment probability for each sample
  
  bool thincheck, printcheck;
  
  uvec *inds_eq_j;
  
  
  ///// Setting-up GSL random number generator for sampling from dirichlet
  /*const gsl_rng_type * T;
  gsl_rng * r;
  
   // create a generator chosen by theenvironment variable GSL_RNG_TYPE 
  
  gsl_rng_env_setup();
  
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set(r,500);*/
  
  ///////////////////////////////////////////////////////////////////
  double alpha=1.0;//randg(distr_param(a_dir,b_dir));
  double prob_empty=log(alpha);
  ///////initialize parameters coresponding to clusters//////
  uvec nj_val(nmix);  //occupancy number corresponding to each cluster
  vec log_nj_val(nmix), niw_kap_post(nmix), df_post(nmix) ,niw_nu_post(nmix) ;
  mat  sum_of_samples=zeros<mat>(k,nmix),marginal_mu=zeros<mat>(k,nmix);
  cube sum_of_squares=zeros<cube>(k,k,nmix),   chol_marginal_delta=zeros<cube>(k,k,nmix),
    marginal_delta=zeros<cube>(k,k,nmix);
  inds_eq_j=new uvec[nmix ];
  
  if(niw_nu<=(double) (k-1)) //wiki notation- prior DF for niw distn
    Rcpp::stop("DF too small in prior");
  
  // Rcpp::Rcout<<"cluster occupancy finding starts!!"<<endl;
  for(j=0;j<nmix;++j){
    inds_eq_j[j] =find(del==j);
    nj_val(j)=(inds_eq_j[j]) .n_elem;
    
    if(nj_val(j)){
      // cout<<"n_inds="<<j<<" is "<<(inds_eq_j[j]) .n_elem<<endl;
      log_nj_val(j)=log(nj_val(j));
      
      niw_nu_post(j)=niw_nu+nj_val(j); niw_kap_post(j)=niw_kap+nj_val(j);
      df_post(j)=niw_nu_post(j)-k+1;
      sum_of_squares.slice(j)= crossprod(eta.rows(inds_eq_j[j]));
      sum_of_samples .col(j)=sum ( eta.rows(inds_eq_j[j]) ,0).t(); //1 in sum/mean implies rowmeans
      
      marginal_mu.col(j) =sum_of_samples.col(j)/ niw_kap_post(j);
      t1ppp=multiplier_fn(niw_nu_post(j), niw_kap_post(j),k);
      
      marginal_delta.slice(j) =symmatu( sum_of_squares.slice(j) - tcrossprod(sum_of_samples .col(j))/niw_kap_post(j) );
      (marginal_delta.slice(j)).diag() += diag_psi_iw;
      chol_marginal_delta.slice(j)=sqrt(t1ppp )*chol(marginal_delta.slice(j), "lower");
      // tmpcube.slice(j)=chol_marginal_delta.slice(j);
    }
  }
  //////////////////////////////////////////////////////////
  
  ////prior NIW matrix param (psi in wiki notation)
  //----------------------------------------------//
  
  // cout<<"loop starts"<<endl;
  // --- loop --- //
  Progress progrr(nrun+burn, 1 );
  for(int i=0; i<nrun+burn; ++i){
    if (Progress::check_abort() )
      return umat(1,1, fill::value(0));
    progrr.increment(); 
    if(dofactor){
      // --- UPDATE mixture parameters --- //
      // --- UPDATE mu & Delta --- //
      for(j=0;j<nmix;++j){
        if(nj_val(j)){
          /*if(nj_val(j)==1)
           cout<<"nj=1 at j= "<<j<<endl;*/
          Delta.slice(j)=iwishrnd(marginal_delta.slice(j), niw_nu_post(j) ); ////This is the posterior cov mt of eta in j-th cluster
          Delta_inv.slice(j)=inv_sympd(Delta.slice(j));
          
          // diagvec(inv_sympd(Delta.slice(j))).print("inv mat_diag:");
          chol( chol_Delta.slice(j), Delta.slice(j) ); //chol_Delta is upper-triangular
          // cout<<"cluster mean and disp matrix updated at j="<<j<<endl;
        }
      }
      // cout<<"cluster mean and disp matrix updated"<<endl;
      
      // --- UPDATE FACTOR MODEL parameters --- //
      // --- UPDATE PSI --- //
      update_loading_params( Lambda, T_phi,  Plam, a);
      
      // --- UPDATE LAMBDA --- //
      eta2 = crossprod( eta);    // prepare eta crossproduct before the loop
      // #pragma omp parallel for
      for(int j = 0; j < p; ++j) {
        // cout<<"Plam.row("<<j<<")"<<Plam.row(j)<<"ps(j)"<<ps(j)<<endl;
        // psi.print("psi:");
        Llamt = chol(diagmat(Plam.row(j)) + ps(j)*eta2);
        // cout<<"Flag"<<endl;
        Lambda.row(j) = (solve(trimatu(Llamt ), randn<vec>(k) +solve(trimatl((Llamt ).t()), ps(j) * (eta.t() * Y.col(j)) ) ) ).t();
      }
      // cout<<"Lambda updated"<<endl;
      if(Lambda.has_nan())
        Rcpp::stop("Lambda has nan!!");
      
      // --- UPDATE ETA --- //
      //---code chunk for updating alloc_prob directly through Y---//
      cube chol_precmat_eta(k,k,nmix);
      //-----------------------------------------------------------//
      
      Lmsg = Lambda.each_col() % ps;
      mat Lmsg2 =symmatu( Lmsg.t() * Lambda);
      // mat inv_Lmsg2 =inv_sympd(Lmsg2);
      // mat eta(n,k,fill::zeros);
      noise = mat(k, n,fill::randn);
      Meta=(Y * Lmsg).t(); //of order k x n              
      
      // #pragma omp parallel for
      for(j=0;j<nmix;++j){
        /////////////////////////////////////////////////////////////////////
        if(nj_val(j)){
          chol_precmat_eta.slice(j) = (chol(Delta_inv.slice(j) + Lmsg2));//Cholesky of posterior prec mt of eta
          
          for( auto it : inds_eq_j[j]){
            eta_list L=gen_eta(nj_val(j),niw_kap, sum_of_samples .col(j)-(eta.row(it)).t(), (Meta.col(it)),
                               Delta_inv.slice(j), chol_Delta.slice(j), chol_precmat_eta.slice(j) );
            eta.row(it)=(L.eta).t();
            sum_of_samples.col(j)=L.sum_of_samples;
          }
          
          
          sum_of_squares.slice(j)= crossprod(eta.rows(inds_eq_j[j]));
          // sum_of_samples .col(j)=sum ( eta.rows(inds_eq_j[j]) ,0).t(); //1 in sum/mean implies rowmeans
          
          marginal_mu.col(j) =sum_of_samples.col(j)/ niw_kap_post(j);
          t1ppp=multiplier_fn(niw_nu_post(j), niw_kap_post(j),k);
          
          marginal_delta.slice(j) =symmatu( sum_of_squares.slice(j) - tcrossprod(sum_of_samples .col(j))/niw_kap_post(j) );
          (marginal_delta.slice(j)).diag() += diag_psi_iw;
          // cout<<"prev flag j= "<<j<<endl;
          chol_marginal_delta.slice(j)=chol(( t1ppp*marginal_delta.slice(j)), "lower");
          // cout<<"flag j= "<<j<<endl;
          // tmpcube.slice(j)=chol_marginal_delta.slice(j);
        }
      }
      // cout<<"eta updated"<<endl;
    }
    
    // --- UPDATE ALLOCATION VARIABLES --- //
    if(randu()<prob){
      // if(0){
      // cout<<"split-merge!"<<endl;
      ++mh_count;
      claunch_list claun= get_claunch(del, nj_val, eta,nstep, niw_nu, niw_kap, diag_psi_iw) ;
      uword fn_i=claun.inds(0), fn_j=claun.inds(1); // values of originally sampled i & j
      
      // (claun.inds).print("claun.inds");
      // (del(claun.inds(span(0,1)) ) ).print("del(claun.inds(span(0,1)) )");
      // 
      // (indmat.t()).print("c_orig, claunch ");
      
      // ((claun.claunch).t()) .print("claun.claunch");
      
      // if( any((claun.claunch)!=(del(claun.inds))  ) && del( fn_i)!=del( fn_j )){
      //   (del(claun.inds(span(0,1)) ) ).print("del(claun.inds(span(0,1)) )");
      //   cout<<"yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes "<<endl;
      //   umat indmat=join_vert((claun.claunch).t(),(del(claun.inds)).t() );
      //   uvec intmp = (  find ( claun.claunch !=del(claun.inds ) ));
      //   
      //   ((indmat.cols(intmp)).t()).print("indmat.cols(intmp)");
      // }
      
      
      double marginal_dens_old, log_mh_prob;
      if(del( fn_i)==del( fn_j )){//checking if i & j belong to same cluster originally-split proposal
        // cout<<"Flag split"<<endl;
        if(del( fn_i ) != claun.claunch(1) || del( fn_i) == claun.claunch(0))// checking consitency of the labels
          Rcpp::stop("Error in get_claunch output-split!");
        if(nj_val(del(fn_j)) != sum(claun.nj_val) ){// checking if n_{c_i^launch}+n_{c_j^launch}=n_{c_i}
          cout<<"nj_val(fn_j)= "<<nj_val(del(fn_j))<<" sum(claun.nj_val)="<<sum(claun.nj_val)<<endl;
          Rcpp::stop("Cluster size mismatch in split proposal!");
        } 
        
        t1ppp=multiplier_fn(niw_nu_post(del(fn_j)), niw_kap_post(del(fn_j)),k);
        marginal_dens_old=log_marg_dens (chol_marginal_delta.slice(del(fn_j)), nj_val(del(fn_j)), t1ppp
                                           ,niw_nu,niw_kap,  diag_psi_iw   );
        
        log_mh_prob=- claun.log_q +log(alpha)+ gsl_sf_lnbeta((double) claun.nj_val(0), (double) claun.nj_val(1))
          +sum(claun.marginal_density)-marginal_dens_old;
        
        // cout<<"log_mh_prob= "<<log_mh_prob<<endl;
        if(log(randu())<log_mh_prob){
          ++acceptance;
          //**** update the source cluster parameters ****//
          j=del(fn_j);
          inds_eq_j[j] =  (claun.inds).elem(find(claun.claunch== j) );
          
          (del.elem(inds_eq_j[j])).fill(j);
          if((inds_eq_j[j]) .n_elem!=claun.nj_val(1))
            Rcpp::stop("(inds_eq_j[j]).n_elem!=claun.nj_val(1) in source cluster!-split");
          
          nj_val(j)=(inds_eq_j[j]) .n_elem;;
          log_nj_val(j ) = log( nj_val(j));
          niw_kap_post(j)=niw_kap+nj_val(j); niw_nu_post(j)=niw_nu+nj_val(j); df_post(j)=niw_nu_post(j)-k+1;
          
          sum_of_squares.slice(j)= crossprod(eta.rows(inds_eq_j[j] ));
          sum_of_samples .col(j)=sum ( eta.rows(inds_eq_j[j]) ,0).t(); //1 in sum/mean implies rowmeans
          marginal_mu.col(j) =sum_of_samples.col(j)/ niw_kap_post(j);
          
          marginal_delta.slice(j)=(claun.marginal_delta).slice(1) ;
          chol_marginal_delta.slice(j)=(claun.chol_marginal_delta).slice(1);
          // tmpcube.slice(j)=chol_marginal_delta.slice(j);
          
          ///////////////////////////////////////////////////////
          
          //**** update the destination cluster parameters ****//
          j=claun.claunch(0);
          inds_eq_j[j] =  (claun.inds).elem(find(claun.claunch== j ) );
          (del.elem(inds_eq_j[j])).fill(j);
          if((inds_eq_j[j]) .n_elem!=claun.nj_val(0))
            Rcpp::stop("samp.inds.n_elem!=claun.nj_val(0) in destination cluster!");
          
          nj_val(j)=(inds_eq_j[j]) .n_elem;
          log_nj_val(j ) = log( nj_val(j));
          niw_kap_post(j)=niw_kap+nj_val(j); niw_nu_post(j)=niw_nu+nj_val(j); df_post(j)=niw_nu_post(j)-k+1;
          
          sum_of_squares.slice(j)= crossprod(eta.rows(inds_eq_j[j] ));
          sum_of_samples .col(j)=sum ( eta.rows(inds_eq_j[j]) ,0).t(); //1 in sum/mean implies rowmeans
          marginal_mu.col(j) =sum_of_samples.col(j)/ niw_kap_post(j);
          
          marginal_delta.slice(j)=(claun.marginal_delta).slice(0) ;
          chol_marginal_delta.slice(j)=(claun.chol_marginal_delta).slice(0);
          // tmpcube.slice(j)=chol_marginal_delta.slice(j);
          ///////////////////////////////////////////////////////
        }
      }
      else{// merge proposal
        // cout<<"Flag merge"<<endl;
        if(del( fn_i ) != claun.claunch(0) || del( fn_j) != claun.claunch(1) ){// checking consitency of the labels
          (del(claun.inds(span(0,1)) ) ).print("del(claun.inds(span(0,1)) )");
          
          (claun.claunch(span(0,1))  ).print("claun.claunch(span(0,1))");
          Rcpp::stop("Error in get_claunch output-merge!");
        } 
        
        uword tmp_nj_val=sum(claun.nj_val);
        
        // cout<<"tmp_nj_val= "<<tmp_nj_val<<endl;
        
        if(nj_val(del(fn_i))+nj_val(del(fn_j)) !=tmp_nj_val  ){// checking if n_{c_i^launch}+n_{c_j^launch}=n_{c_i}+n_{c_j}
          cout<<"nj_val(del(fn_i))="<<nj_val(del(fn_i))<<" nj_val(del(fn_j))="<<nj_val(del(fn_j))<<" tmp_nj_val="<<tmp_nj_val<<endl;
          Rcpp::stop("Cluster size mismatchin merge proposal!");
        }
        
        ///////////////////compute L(c^merge|y) ////////////////////////////////
        vec tmp_marginal_mu, tmp_sum_of_samples;
        mat tmp_sum_of_sq, tmp_chol_marginal_delta, tmp_marginal_delta;
        
        
        uword tmp_niw_nu_post=niw_nu+tmp_nj_val, tmp_niw_kap_post=niw_kap+tmp_nj_val, tmp_df_post=tmp_niw_nu_post-k+1;
        
        tmp_sum_of_sq= sum_of_squares.slice(del(fn_j))+sum_of_squares.slice(del(fn_i));
        tmp_sum_of_samples=sum_of_samples.col(del(fn_j)) + sum_of_samples.col(del(fn_i));
        
        tmp_marginal_mu =tmp_sum_of_samples/ tmp_niw_kap_post;
        t1ppp=multiplier_fn(tmp_niw_nu_post, tmp_niw_kap_post,k);
        
        tmp_marginal_delta =symmatu( tmp_sum_of_sq - tcrossprod(tmp_sum_of_samples)/tmp_niw_kap_post );
        (tmp_marginal_delta).diag() += diag_psi_iw;
        tmp_chol_marginal_delta=sqrt(t1ppp )*chol(tmp_marginal_delta, "lower");
        
        
        double marginal_dens_new=log_marg_dens (tmp_chol_marginal_delta, tmp_nj_val, t1ppp
                                                  ,niw_nu,niw_kap,  diag_psi_iw   );
        ///////////////////////////////////////////////////////////////////////////
        
        
        ///////////////////compute L(c|y) ////////////////////////////////
        marginal_dens_old=0.0;
        for(j=0;j<2;++j){
          t1ppp=multiplier_fn(niw_nu_post(del(claun.inds(j))), niw_kap_post(del(claun.inds(j))),k);
          marginal_dens_old+=log_marg_dens (chol_marginal_delta.slice(del(claun.inds(j))), nj_val(del(claun.inds(j))), t1ppp
                                              ,niw_nu,niw_kap,  diag_psi_iw   );
        }
        
        // cout<<"marginal_dens_old:merge proposal : "<<marginal_dens_old<<endl;
        ////////////////////////////////////////////////////////////////////////////
        claunch_list out_merge= get_cmerge(claun, del(claun.inds) , eta.rows(claun.inds ),
                                           niw_nu, niw_kap, diag_psi_iw);
        
        log_mh_prob= claun.log_q -log(alpha)- gsl_sf_lnbeta((double) nj_val(del(fn_i)), (double) nj_val(del(fn_j)))
          +marginal_dens_new-marginal_dens_old;
        // cout<<"log_mh_prob= "<<log_mh_prob<<endl;
        
        if(log(randu())<log_mh_prob){
          ++acceptance;
          //**** update the source cluster parameters ****//
          j=del(fn_i);
          inds_eq_j[j].reset();
          if((inds_eq_j[j]) .n_elem!=0)
            Rcpp::stop("inds_eq_j[fn_i].n_elem!=0 in source cluster!-merge");
          
          nj_val(j)=inds_eq_j[j].n_elem;
          log_nj_val(j ) = datum::log_min;
          ///////////////////////////////////////////////////////
          
          //**** update the destination cluster parameters ****//
          j=del(fn_j);
          inds_eq_j[j] =  (claun.inds);
          if((inds_eq_j[j]) .n_elem!= out_merge.nj_val(0))
            Rcpp::stop("inds_eq_j[j]) .n_elem!= out_merge.nj_val(0) in destination cluster!-merge");
          
          (del.elem(inds_eq_j[j])).fill(j);
          nj_val(j)=(inds_eq_j[j]) .n_elem;
          log_nj_val(j ) = log( nj_val(j));
          niw_kap_post(j)=niw_kap+nj_val(j); niw_nu_post(j)=niw_nu+nj_val(j); df_post(j)=niw_nu_post(j)-k+1;
          
          sum_of_samples.col(j)=tmp_sum_of_samples;
          sum_of_squares.slice(j)=tmp_sum_of_sq;
          marginal_mu.col(j)=tmp_marginal_mu;
          
          marginal_delta.slice(j)=tmp_marginal_delta;
          chol_marginal_delta.slice(j)=tmp_chol_marginal_delta;
          // tmpcube.slice(j)=tmp_chol_marginal_delta;
          ///////////////////////////////////////////////////////
        }
      }
      
      // cout<<"log_mh_prob= "<<log_mh_prob<<endl;
      // mh_prob_file<< (double) acceptance/(double) mh_count<<endl;
      
    }
    else{
      // cout<<"gibbs!"<<endl;
      double  log_probs_max;//,normal_exp; 
      
      for(int jj=0;jj<n;++jj){
        double dens,cluster_prob, pdf_empty;
        pdf_empty= log_t_density_empty((double) (niw_nu-k+1), niw_kap,diag_psi_iw, (eta.row(jj)).t() );
        
        // cout<<"log_pdf_empty="<<pdf_empty<<endl;
        uvec non_empty_clusts= find(nj_val>0);//ids of non-empty clusters
        uword current_ind=del(jj); //find the current index of data point jj
        
        ///If nj_vl(current_ind)==1 no need to create an empty cluster....if not we have to////
        unsigned nclust;
        if(nj_val(current_ind)>1){
          nclust =   non_empty_clusts.n_elem+1;
          
          log_probs.set_size(nclust);
          
          uvec empty_inds=find(nj_val==0,1,"first");
          non_empty_clusts.resize(non_empty_clusts.n_elem+1);  non_empty_clusts.tail(1)=empty_inds(0);
        }
        else {
          nclust =   non_empty_clusts.n_elem;
          log_probs.set_size(nclust);
        }
        ///////////////////////////////////////////////////////////////////////////////////////
        
        
        
        for(unsigned j1=0;j1<nclust;++j1){
          j=non_empty_clusts(j1);
          // cout<<"mcmc_iter="<<mcmc_iter<< " jj= "<<jj<<"clust id= "<<j<<" n_j= "<<nj_val(j)<< endl;
          if(j!=current_ind){
            if(nj_val(j)<1 ){
              dens=pdf_empty;
              cluster_prob=prob_empty;
            }
            
            else{
              dens= log_t_density(df_post(j) ,  (eta.row(jj)).t()
                                    ,marginal_mu.col(j), chol_marginal_delta.slice(j));
              // cout<<"at j="<<j<<" log_density="<<dens<<"\t clust prob="<<cluster_prob<<endl;
              cluster_prob=log_nj_val(j);
            }
            log_probs(j1)= dens +cluster_prob;
          }
        }
        //calculating allocation probabilities
        
        j=current_ind;
        mat cross_prod=crossprod(eta.row(jj));
        vec tmp_marginal_mu(k), tmp_sum_of_samples(k);
        mat tmp_sum_of_sq=zeros<mat>(k,k), tmp_chol_marginal_delta=zeros<mat>(k,k), tmp_marginal_delta=zeros<mat>(k,k);
        
        if(nj_val(j)==1){
          dens= pdf_empty;
          cluster_prob=prob_empty;
        }
        
        else{
          if(nj_val(j)==0 )   Rcpp::stop("nj_val(current_ind)=0 ");
          tmp_sum_of_samples=sum_of_samples .col(j) - (eta.row(jj)).t() ;
          tmp_marginal_mu=(tmp_sum_of_samples ) /(niw_kap_post(j)-1) ;
          
          tmp_sum_of_sq=symmatu( sum_of_squares.slice(j)- cross_prod );
          
          // cout<<"eig_sym( tmp_sum_of_sq )=";
          // eig_sym( tmp_sum_of_sq ).print();
          
          // cout<<"jj="<<jj<<endl;
          // cout<<"Flag 1 current_ind="<<current_ind<<"nj_val-1="<<nj_val(current_ind) -1<<endl;
          
          
          tmp_marginal_delta=symmatu( tmp_sum_of_sq -  tcrossprod(tmp_sum_of_samples)/(niw_kap_post(j)-1 ) );
          // cout <<"tmp_marginal_delta.diag()="<< tmp_marginal_delta.diag() << endl;
          tmp_marginal_delta.diag() += diag_psi_iw;  
          
          t1ppp= multiplier_fn( niw_nu_post(j)-1 , niw_kap_post(j)-1, k);
          tmp_chol_marginal_delta=  chol(t1ppp* tmp_marginal_delta ,"lower") ;
          
          dens= log_t_density(df_post(j)-1 ,  (eta.row(jj)).t(),tmp_marginal_mu, tmp_chol_marginal_delta );
          cluster_prob=log( nj_val(j)-1 );
        }
        
        uvec indddd=find( non_empty_clusts==j,1,"first");
        log_probs(indddd(0) )= dens +cluster_prob;
        
        
        // cout<<"at current_ind="<<j<<" log_density="<<dens<<"\t clust prob="<<cluster_prob<<endl;
        // log_probs.print("log_probs=");
        
        double log_DEN=log_sum_exp(log_probs);
        probs= normalise(exp(log_probs-log_DEN) ,1);
        
        // log_probs.print("log_probs: ");
        // (probs.t()).print("probs: ");
        
        if(   gsl_fcmp(sum(probs),1.0,1e-5) ){
          // cout<< "At jj="<<jj<<"sum_prob is 0"<<endl;
          log_probs_max= max(log_probs);
          log_probs-=log_probs_max;
          
          probs=normalise(exp(log_probs) ,1);
        }
        
        ////check if sum of the allocation probbilities is zero
        if(sum(probs)==0)  {
          cout<<"sum_prob=0 after adjustment at nsamp= "<<jj<<" Iteration= "<<i<<endl;
          Rcpp::stop("");
        }
        ///////////////////////////////////////////////////////
        
        d.set_size(probs.n_elem);
        R::rmultinom(1, probs.begin(), probs.n_elem, d.begin());
        uvec dd=find(d==1,1,"first");
        del(jj)=non_empty_clusts( dd(  0));
        
        //updating cluster occupancies
        if(del(jj)!=current_ind){
          // j=del(jj);
          // cout<<"cluster change mcmc_iter="<<mcmc_iter<< " sample id= "<<jj<<"prev clust id= "<<current_ind<<" n_j= "<<nj_val(current_ind)<< endl;
          // cout<<"new clust id= "<<j<<" n_j="<<nj_val(j)<<endl;
          
          //**** update the source cluster parameters ****//
          j=current_ind;
          --nj_val(j);
          
          if(nj_val(j) ){
            log_nj_val(j ) = log(nj_val(j)  );
            niw_kap_post(j)--; niw_nu_post(j)--; df_post(j)--;
            
            sum_of_samples.col(j)=tmp_sum_of_samples;
            sum_of_squares.slice(j)=tmp_sum_of_sq;
            marginal_mu.col(j)=tmp_marginal_mu;
            
            marginal_delta.slice(j)=tmp_marginal_delta;
            chol_marginal_delta.slice(j)=tmp_chol_marginal_delta;
            // tmpcube.slice(j)=tmp_chol_marginal_delta;
          }
          
          ///////////////////////////////////////////////////////
          
          //**** update the destination cluster parameters ****//
          j=del(jj);
          log_nj_val(j ) = log( ++nj_val(j) );
          niw_nu_post(j)=niw_nu+nj_val(j); niw_kap_post(j)=niw_kap+nj_val(j); df_post(j)=niw_nu_post(j)-k+1;
          
          if(nj_val(j)==1){
            sum_of_samples.col(j)=  (eta.row(jj)).t() ;
            sum_of_squares.slice(j) = cross_prod ;
          }
          else{
            sum_of_samples.col(j)+=  (eta.row(jj)).t() ;
            sum_of_squares.slice(j) += cross_prod ;
          }
          
          marginal_mu.col(j) =(sum_of_samples.col(j) ) /(niw_kap_post(j)) ;
          
          marginal_delta.slice(j) =symmatu( sum_of_squares.slice(j) - tcrossprod(sum_of_samples .col(j))/niw_kap_post(j) );
          (marginal_delta.slice(j)).diag() += diag_psi_iw;
          t1ppp= multiplier_fn( niw_nu_post(j) , niw_kap_post(j), k);
          
          chol_marginal_delta.slice(j) =  chol(t1ppp*marginal_delta.slice(j) ,"lower") ;
          // tmpcube.slice(j)=chol_marginal_delta.slice(j);
          ///////////////////////////////////////////////////////
        }
      }
    }
    // cout<<"allocation variables updated"<<endl;
    
    // --- UPDATE MIXTURE PROBABILITY --- //
    unsigned counttttt=0;
    for(j=0;j<nmix;++j){
      if(nj_val(j)){
        ++counttttt;
        inds_eq_j[j] =find(del==j);
      }
      else (inds_eq_j[j]).reset();
      
      if((inds_eq_j[j]).n_elem != nj_val(j)){
        cout<<"inds_eq_j[j]).n_elem="<<(inds_eq_j[j]).n_elem<<"\t"<<"nj_val("<<j<<")="<<nj_val(j)<<endl;
        Rcpp::stop("Occupancy mismatch!!!");
      }
    }
    // cout<< "# clusters= "<< counttttt<<endl;
    
    
    //////UPDATE dir_prec
    alpha=update_dir_prec(alpha,  a_dir,  b_dir,n, counttttt);
    prob_empty=log(alpha);
    
    
    // --- UPDATE SIGMA --- //
    mat Y_hat=eta * Lambda.t();
    Ytil = Y - Y_hat;
    ps =  bs + 0.5 * sum(square(Ytil), 0).t();
    ps.transform( [&as,&n](double val) { return  randg(distr_param(as + 0.5*n, 1 / val ) ); ; } );
    // cout<<"Sigma updated"<<endl;
    
    /*thincheck = i - std::floor(i/thin) * thin; // % operator stolen by arma
    if(!thincheck ) {
      cout<<"Iteration: "<<i<< "# clusters= "<< counttttt<<" "<<msg<<endl;
    }*/
    
    
    int remainder= (i+1 );
    int quotient= (int) std::floor(remainder/thin);
    remainder-= (quotient*thin) ;
    
    if(remainder==0){
      alloc_var_mat.row(quotient-1)=del.t();
    }
  }
  
  delete[] inds_eq_j;
  // gsl_rng_free (r);
  
   
  return alloc_var_mat;
}
