args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L)
request <- jsonlite::fromJSON(args[[1]])
read_matrix <- function(name) as.matrix(read.csv(file.path(request$directory,paste0(name,".csv")),header=FALSE))
cache <- Sys.getenv("WHEAT599_RCPP_CACHE",file.path(path.expand("~"),".cache","wheat599-rcpp"))
if (isTRUE(request$compile_only)) {
 build <- Rcpp::sourceCpp(request$source,cacheDir=cache,showOutput=FALSE)
 jsonlite::write_json(list(loader=file.path(build$buildDirectory,"sampler.cpp.R")),
                     request$loader_manifest,auto_unbox=TRUE)
 quit(status=0)
}
if (is.null(request$loader)) stop("Compile the sampler once before starting chains")
suppressPackageStartupMessages(library(Rcpp))
source(request$loader)
fit <- sparse_polygenic_bart(read_matrix("x"),read_matrix("y"),read_matrix("query"),
 request$trees,request$kappa,request$warmup,request$draws,request$seed)
metadata <- list(dimensions=lapply(fit,dim),seed=request$seed,warmup=request$warmup,draws=request$draws,
 R=R.version.string,Rcpp=as.character(packageVersion("Rcpp")),RcppArmadillo=as.character(packageVersion("RcppArmadillo")))
for (name in names(fit)) {
 value <- fit[[name]]
 if (!all(is.finite(value))) stop("Nonfinite posterior output")
 writeBin(as.double(value),file.path(request$output,paste0(name,".f64")),size=8L,endian="little")
}
jsonlite::write_json(metadata,file.path(request$output,"metadata.json"),auto_unbox=TRUE,pretty=TRUE)
