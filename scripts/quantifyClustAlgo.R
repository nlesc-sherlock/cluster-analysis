library(ClusterR)
library(reshape2)

### Reference material 
https://cran.r-project.org/web/packages/ClusterR/vignettes/the_clusterR_package.html

setwd("/home/anandgavai/Sherlock5/cluster-analysis/data/pentax")

mat<-read.csv("matrix-pentax-pce.txt",header = FALSE,sep=",")

# Remove last empty column
colnames(mat)<-NULL
rownames(mat)<-NULL

mat<-as.matrix(mat[,1:638])
g  <- graph.adjacency(mat,weighted=TRUE)

# vector of edgelist
#Edge list is not the right data format for benchmarking

# Get the feature adn cordinate matrix from Sonya and run the ClusterR algorithm on all the features

