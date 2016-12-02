library(ClusterR)
library(reshape2)
library(igraph)
### Reference material 
##https://cran.r-project.org/web/packages/ClusterR/vignettes/the_clusterR_package.html

setwd("/home/anandgavai/Sherlock5/cluster-analysis/data/embedded_3D")

mat<-read.csv("coordinates_pentax_pce_log.txt",header = FALSE,sep="")



remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

mat[,2]<-remove_outliers(mat[,2])
mat[,3]<-remove_outliers(mat[,3])
mat[,4]<-remove_outliers(mat[,4])
mat<-na.omit(mat)



### start using GMM for this data now
X<-mat
dat = center_scale(X[,2:4], mean_center = T, sd_scale = T)  # centering and scaling the data

gmm = GMM(dat, 4, dist_mode = "maha_dist", seed_mode = "random_subset", km_iter = 10,
          
          em_iter = 10, verbose = F)

pr = predict_GMM(dat, gmm$centroids, gmm$covariance_matrices, gmm$weights)  


opt_gmm = Optimal_Clusters_GMM(dat, max_clusters = 4, criterion = "AIC", 
                               
                               dist_mode = "eucl_dist", seed_mode = "random_subset",
                               
                               km_iter = 10, em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)


gmm_data<-cbind(X,pr$log_likelihood,pr$cluster_proba,pr$cluster_labels)

colnames(gmm_data)<-c("Cameras","feature1","feature2","feature3","log_lik1","log_lik2","log_lik3","log_lik4","clus_prob1","clus_prob2","clus_prob3","clus_prob4","cluster_labels")

write.csv(gmm_data,file="GMM_WO_Log_Trans1.csv",row.names=FALSE)
