
### Read binary file Examples of sincle file reading ###
# Fourier transformed data is located in temp
infile1 <- "/var/scratch/bwn200/temp/Agfa_Sensor505-x_0_1890.dat"
infile2 <- "/var/scratch/bwn200/temp/Kodak_M1063_4_12768.dat"
# Raw data
infile1 <- "/var/scratch/bwn200/patterns_set1/Agfa_Sensor505-x_0_1890.dat"




# read list of files from a directory
files<-list.files("/var/scratch/bwn200/temp/",pattern="*.dat")

# The dimention of a particular image noise is 2592*1944*4
# It is in total one big array of noise 2592 is the rows, 1944 are the columns and 4 represents the byte size
ss<-NULL
for(i in 1:length(files)){
  con <- file(paste("/var/scratch/bwn200/temp/",files[i],sep=""), "rb")
  dim <- readBin(con, "integer",20155392)
  fname<-cbind(files[i],min(dim),max(dim),mean(dim),median(dim),sd(dim),skewness(dim),kurtosis(dim))
  ss<-rbind(ss,fname)
  print(fname)
  close(con)
}


### Individual file exploration
con1 <- file(infile1, "rb")
dim1 <- readBin(con1, "integer",20155392)
close(con1)

con2 <- file(infile2, "rb")
dim2 <- readBin(con2, "integer",20155392)
close(con2)


med<-read.csv('/home/anandgavai/ClusterAnalysis/med.csv',header=TRUE)
med<-read.csv('/home/anandgavai/ClusterAnalysis/med_from_fourier_Kodak.csv',header=TRUE)
summaryNoise<- read.csv('/home/anandgavai/ClusterAnalysis/Summary_Noise_Analysis_Kodak.csv',header=TRUE)


## Hierarchical clustering
d <- dist(as.matrix(summaryNoise)) # distance matrix
fit <- hclust(d) 
plot(fit) # display dendogram



# K-Means Cluster Analysis
fit <- kmeans(summaryNoise[,2:8], 6) # 5 cluster solution
# get cluster means 
aggregate(summaryNoise[2:8],by=list(fit$cluster),FUN=median)
# append cluster assignment
mydata <- data.frame(summaryNoise[,2:8], fit$cluster)
table(mydata[,8])




