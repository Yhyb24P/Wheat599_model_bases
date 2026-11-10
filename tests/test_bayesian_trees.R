suppressPackageStartupMessages(library(Rcpp))
sourceCpp("src/wheat599/bayesian_trees/sampler.cpp",cacheDir=file.path(path.expand("~"),".cache/wheat599-rcpp"))
set.seed(1701)
z <- matrix(rnorm(12),4,3);y <- matrix(rnorm(8),4,2)
sigma <- matrix(c(1,.4,.4,2),2,2);v <- .3;draws <- 16000L
P <- kronecker(solve(sigma),crossprod(z))+diag(1/v,6)
C <- solve(P);m <- C%*%as.vector(crossprod(z,y)%*%solve(sigma))
a <- gaussian_block_check(z,y,sigma,v,draws)
stopifnot(max(abs(a$mean-m)/sqrt(diag(C)/draws))<5,
 max(abs(a$covariance-C))<.025)
set.seed(22)
x <- matrix(rbinom(240,1,.5),40,6);y <- matrix(rnorm(160),40,4)+x[,1]
a <- sparse_polygenic_bart(x,y,x[1:3,],5L,2,30L,20L,42L)
b <- sparse_polygenic_bart(x,y,x[1:3,],5L,2,30L,20L,42L)
stopifnot(identical(a,b),identical(dim(a$query),c(3L,4L,20L)),all(is.finite(a$query)))
# Query order affects only output order, not training or the posterior RNG path.
c <- sparse_polygenic_bart(x,y,x[3:1,],5L,2,30L,20L,42L)
stopifnot(identical(a$training,c$training),max(abs(a$query-c$query[3:1,,,drop=FALSE]))==0)
d <- sparse_polygenic_bart(x,y,1-x[1:3,],5L,2,30L,20L,42L)
stopifnot(identical(a$training,d$training),identical(a$sigma,d$sigma))
stopifnot(inherits(try(sparse_polygenic_bart(x+.1,y,x[1:3,],5L,2,30L,20L,42L),silent=TRUE),'try-error'))
cat("Gaussian posterior moments, deterministic initialization, binary-input validation and query isolation passed.\n")
k <- sampler_kernel_checks(80000L)
stopifnot(abs(k$split_frequency-k$split_exact)<.015,
 abs(k$dirichlet_mean-.5)<.025,abs(k$dirichlet_second_moment-.375)<.025)
cat("Tree birth/death detailed balance and DART legal-support correction passed.\n")
stopifnot(abs(k$concentration_prior_mean_u-1/3)<.04,k$covariance_max_error<.002)
cat("Concentration hyperprior and inverse-Wishart conditional moments passed.\n")
