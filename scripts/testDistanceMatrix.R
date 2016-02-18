library (seewave)
library (quantmod)

#file1a <-"/home/anandgavai/Sherlock3/cluster-analysis/analysis/noise_data/Agfa_Sensor505-x_0_1890.dat"
#file2a <-"/home/anandgavai/Sherlock3/cluster-analysis/analysis/noise_data/Agfa_Sensor505-x_0_1891.dat"

## raw files from Agfa camera this are highly correlated files, show how they look like
file1 <-"/home/anandgavai/Sherlock3/cluster-analysis/analysis/noise_data/Agfa_Sensor505-x_0_1906.dat"
file2 <-"/home/anandgavai/Sherlock3/cluster-analysis/analysis/noise_data/Agfa_Sensor505-x_0_1904.dat"

### files from Canon camera
file3 <- "/home/anandgavai/Sherlock3/cluster-analysis/analysis/noise_data/Canon_Ixus55_0_2626.dat"
file4 <- "/home/anandgavai/Sherlock3/cluster-analysis/analysis/noise_data/Canon_Ixus55_0_2625.dat"


### Individual file exploration
con1 <- file(file1, "rb")
dim1 <- readBin(con1, endian="big",what="double",10077696)
close(con1)


### Individual file exploration
con2 <- file(file2, "rb")
dim2 <- readBin(con2, endian="big",what="double",10077696)
close(con2)


### Individual file exploration
con3 <- file(file3, "rb")
dim3 <- readBin(con3, endian="big",what="double",10077696)
close(con3)



### Individual file exploration
con4 <- file(file4, "rb")
dim4 <- readBin(con4, endian="big",what="double",10077696)
close(con4)


### Cross correlation for two different cameras 
ccf(dim1,dim4)

### Cross correlation of two similar cameras
ccf(dim1,dim2)


## Fourier transformation
fdim1 <- as.numeric(fft(dim1))
fdim2 <- as.numeric(fft(dim2))





