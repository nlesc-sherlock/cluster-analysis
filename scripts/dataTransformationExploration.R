library(manipulate)
library(mixtools)


### This is the Ground truth
filelist <-  read.table("/home/anandgavai/Sherlock3/cluster-analysis/data/pentax/filelist.txt",sep=" ")


### list of edge list extracted for ncc, pce and pce0
edgeList_ncc <- read.table("/home/anandgavai/Sherlock3/cluster-analysis/data/pentax/edgelist-pentax-ncc.txt",sep=" ")
edgeList_pce <- read.table("/home/anandgavai/Sherlock3/cluster-analysis/data/pentax/edgelist-pentax-pce.txt",sep=" ")
edgeList_pce0 <- read.table("/home/anandgavai/Sherlock3/cluster-analysis/data/pentax/edgelist-pentax-pce0.txt",sep=" ")





## density plots with and witout transformation explorations
## ncc
ncc<-edgeList_ncc[,3]
plot(density(ncc))
plot(density(na.omit(sqrt(ncc))))
plot(density(na.omit(log(ncc))))


##pce
pce<-edgeList_pce[,3]
plot(density(pce))
plot(density(sqrt(pce)))
plot(density(log(pce)))


## pce0
pce0<-edgeList_pce0[,3]
plot(density(pce0))
plot(density(sqrt(pce0)))
plot(density(log(pce0)))


## summary of raw data
summary (ncc)
summary (pce)
summary (pce0)

# example only for pce values, log transformed
pce_log<-log(pce)
plot(hist(pce_log,breaks=c(1000)))

nV<-pce_log
#remove outliers from the distribution
nV<-pce_log[!pce_log %in% boxplot.stats(pce_log)$out]
boxplot(nV)

x<-nV
model <- normalmixEM(x=x, k=2)
index.lower <- which.min(model$mu)  # Index of component with lower mean

find.cutoff <- function(proba=0.5, i=index.lower) {
  ## Cutoff such that Pr[drawn from bad component] == proba
  f <- function(x) {
    proba - (model$lambda[i]*dnorm(x, model$mu[i], model$sigma[i]) /
               (model$lambda[1]*dnorm(x, model$mu[1], model$sigma[1]) + model$lambda[2]*dnorm(x, model$mu[2], model$sigma[2])))
  }
  return(uniroot(f=f, lower= as.numeric(quantile(x,0.05)), upper=as.numeric(quantile(x,0.95)))$root)  
}

cutoffs <- c(find.cutoff(proba=0.5), find.cutoff(proba=0.75)) 

hist(x)
abline(v=cutoffs, col=c("red", "blue"), lty=2)




