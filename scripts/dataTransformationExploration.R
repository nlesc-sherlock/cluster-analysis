library(manipulate)

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


#scaling between 0 and 1 
nV<-(pce_log-min(pce_log))/(max(pce_log)-min(pce_log))

plot(density(nV))


idx<-which(nV < 0.4)
nV[idx]<-0.4

idx2<-which(nV>0.9)
nV[idx2]<-0.9

plot(density(nV))


x <- ((nV/0.35) - (0.4/0.35))^2
x <- x/2

plot(density(x))



summary(log(pce))

## Testing to check if values from adjacency matrix are same as values from edgeList
mat_pce <- read.table("/home/anandgavai/Sherlock3/cluster-analysis/data/pentax/matrix-pentax-pce.txt",sep=",")
mat_pce<-mat_pce[,1:638]


