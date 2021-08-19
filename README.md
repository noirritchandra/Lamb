LAtent Mixture of Bayesian Cluetring (Lamb) - User Manual
================
Noirrit Kiran Chandra
8/18/2021

## \*\*For ***`Microsoft Windows`***Users

There might be some issues with `Windows` users while compiling the
`C++` file. Please refer to [this stackoverflow
link](https://stackoverflow.com/questions/55976547/linking-gsl-libraries-to-rcppgsl-in-windows-10).

## Compiling the Source `C++` File and Loading Requisite Libraries

## Simulate Data from the Lamb Model

#### Simulate Data

``` r
set.seed(1)
p= 2500 # Observed dimension
n=2000 # Sample size
d=20 # Latent dimension
k= 15 # Number of true clusters

#cluster memberships probabilities 
# 2/3 of the clusters have same probs, the remaining 1/3 together have the same prob of a single big bluster
pis <- c(rep(1/(round(k*2/3)+1), (round(k*2/3))), 
rep((1/(round(k*2/3)+1))/(k - (round(k*2/3))), k - (round(k*2/3))))
pis <- pis/sum(pis) ##cluster weights

lambda = simulate_lambda(d,p,1) # Generate the loading matrix lambda
sc=quantile(diag(tcrossprod(lambda) ) ,.75); lambda=lambda/sqrt(sc) # Scaling lambda 
s = sample.int(n=k,size=n, prob=pis, replace = TRUE) ##Cluster membership indicators
eta = matrix(rnorm(n*d), nrow=n, ncol=d) # Latent factors
shifts=2*seq(-k,k,length.out = k) 

for(i in 1:k){
  inds = which(s==i)
  vars = sqrt(rchisq(d,1))
  m = pracma::randortho(d) %*% diag(vars)
  eta[inds,]= eta[inds,] %*% t(m)+shifts[i] # The i-th cluster is centered around rep(shifts[i],d) in the latend eta space
}
y <- tcrossprod(eta,lambda) + matrix(rnorm(p*n, sd=sqrt(.5)), nrow=n, ncol=p) # Observed data
```

#### UMAP Plot of the Simulated Data

``` r
umap_data=uwot::umap(y, min_dist=.8, n_threads = parallel::detectCores()/2) #2-dimensional UMAP scores of the original data
dat.true=data.frame(umap_data, Cluster=as.factor(s)); colnames(dat.true)= c("UMAP1","UMAP2", "Cluster")
p.true<-ggplot(data = dat.true, mapping = aes(x = UMAP1, y = UMAP2 ,colour=Cluster)) + geom_point()
p.true
```

![](README_files/figure-gfm/UMAP%20plot-1.png)<!-- -->

------------------------------------------------------------------------

> ***For any other dataset replace the simlated `y` with that and follow
> the proceeding steps*.**

## Pipeline of Fitting the Lamb Model

#### Pre-processing Data

``` r
y_original=y
y=y_original
centering.var=median(colMeans(y))
scaling.var=median(apply(y,2,sd))
y=(y_original-centering.var)/scaling.var
```

#### Empirical Estimation the Latent Dimension `d` and Initializing `eta` & `lambda`

``` r
pca.results=irlba::irlba(y, nv=40)
cum_eigs= cumsum(pca.results$d)/sum(pca.results$d)
d=min(which(cum_eigs>.95)) # Smallest $d$ explaining at least 95% of the variability in the data
d=ifelse(d<=15,15,d) # Set d= at least 15
eta= pca.results$u %*% diag(pca.results$d)
eta=eta[,1:d] #Left singular values of y are used as to initialize eta
lambda=pca.results$v[,1:d] #Right singular values of y are used to initialize lambda
```

#### Initializing Cluster Allocations using `KMenas`

``` r
cluster.start = kmeans(y, 80)$cluster - 1
```

### Set Prior Parameter of `Lamb`

##### Parameter Description in the Chandra et al. (2021+) Notations:

| Parameters                                                                                                                                                                                                                                                        | Description                                                                                                                                                                                     |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `as`=![formula](https://render.githubusercontent.com/render/math?math=a_%5Csigma); `bs`=![formula](https://render.githubusercontent.com/render/math?math=b_%5Csigma)                                                                                              | The Gamma hyperparameters for residual precision ![formula](https://render.githubusercontent.com/render/math?math=%5Csigma_j%5E%7B-2%7D)’s for j=1,…,p                                          |
| `a`=<img src="https://render.githubusercontent.com/render/math?math=a "/>                                                                                                                                                                                         | Dirichlet-Laplace parameter on `lambda`                                                                                                                                                         |
| `diag_psi_iw`=![formula](https://render.githubusercontent.com/render/math?math=%5Cxi%5E2); `niw_kap`=![formula](https://render.githubusercontent.com/render/math?math=%5Ckappa_0); `nu`=![formula](https://render.githubusercontent.com/render/math?math=%5Cnu_0) | Hyperarameters for the Normal-Inverse Wishart base measure ![formula](https://render.githubusercontent.com/render/math?math=G_0) of the Dirichlet process prior on `eta`                        |
| `a_dir`=![formula](https://render.githubusercontent.com/render/math?math=a_%5Calpha); `b_dir`=![formula](https://render.githubusercontent.com/render/math?math=b_%5Calpha)                                                                                        | Gamma hyperparameters for the conjugate Gamma(![formula](https://render.githubusercontent.com/render/math?math=a_%5Calpha,b_%5Calpha)) prior on the Dirichlet-process concentration parameter 𝛼 |

##### Set Prior Hyperparameters

``` r
as = 1; bs = 0.3 
a=0.5
diag_psi_iw=20
niw_kap=1e-3
nu=d+50
a_dir=.1
b_dir=.1
```

#### Fit the `Lamb` Model

``` r
result.lamb <- DL_mixture(a_dir, b_dir, diag_psi_iw=diag_psi_iw, niw_kap=niw_kap, niw_nu=nu, 
                          as=as, bs=bs, a=a,
                          nrun=5e3, burn=1e3, thin=1, 
                          nstep = 5,prob=0.5, #With probability `prob` either Split-Merge sampler or Gibbs sampler in performed witn `nstep` Gibbs scans
                          lambda, eta, y,
                          del = cluster.start, 
                          dofactor=1 #If dofactor set to 0, clustering will be done on the initial input eta values only; eta will not be updated in MCMC
                          )
```

### Post-processing the MCMC Samples

``` r
burn=2e2 #burn/thin
post.samples=result.lamb[-(1:burn),]+1
sim.mat.lamb <- mcclust::comp.psm(post.samples) # Compute posterior similarity matrix
clust.lamb <-   minbinder(sim.mat.lamb)$cl # Minimizing Binder loss across MCMC estimates
```

### Adjusted Rand Index

``` r
adjustedRandIndex( clust.lamb, s)
```

    ## [1] 0.9990262

### UMAP Plot of Lamb Clustering

``` r
dat.lamb=data.frame(umap_data, Cluster=as.factor(clust.lamb)); colnames(dat.true)= c("UMAP1","UMAP2", "Cluster")
p.lamb<-ggplot(data = dat.true, mapping = aes(x = UMAP1, y = UMAP2 ,colour=Cluster)) + geom_point()
p.lamb
```

![](README_files/figure-gfm/UMAP%20Plot%20of%20Lamb-1.png)<!-- -->

### Comparison with the True Clustering

``` r
ggpubr:: ggarrange(p.true,p.lamb,nrow=1,ncol=2, labels = c("True", "Lamb"))
```

![](README_files/figure-gfm/Comparison%20with%20True%20UMAP-1.png)<!-- -->
