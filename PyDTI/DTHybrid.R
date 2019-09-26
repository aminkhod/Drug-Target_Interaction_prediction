### try http:// if https:// URLs are not supported
#source("https://bioconductor.org/biocLite.R")
#biocLite("BiocGenerics")

library(DTHybrid)
# Example using a Drug-Target Interaction dataset

#options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE)
print(args)

lambda <- as.numeric(args[2])
alpha <- as.numeric(args[3])
print(lambda)
print(alpha)
w=read.table("DTHybridW.txt")
Y= read.table('DTHybridintMat.txt')
sd = read.table('DTHybriddrugMat.txt')
st = read.table('DTHybridtargetMat.txt')

w <- as.matrix(w)
sd <- as.matrix(sd)
st <- as.matrix(st)
Y <- as.matrix(Y)

#setwd('../data/datasets')
#switch (db,
#        e = {
#          cat("en data\n")
#          flush.console()
#          sd <- read.table("e_simmat_dc.txt")
#          sd <- as.matrix(sd)
#          st <- read.table("e_simmat_dg.txt")
#          st <- as.matrix(st)
#          Y <- read.table("e_admat_dgc.txt")
#          Y <- as.matrix(Y) 
#          Y <- t(Y)         
#        },
#        ic = {
#          cat("ic data\n")
#          flush.console()
#          sd <- read.table("ic_simmat_dc.txt")
#          sd <- as.matrix(sd)
#          st <- read.table("ic_simmat_dg.txt")
#          st <- as.matrix(st)
#          Y <- read.table("ic_admat_dgc.txt")
#          Y <- as.matrix(Y)
#          Y <- t(Y)
#        },
#        gpcr = {
#          cat("gpcr data\n")
#          flush.console()
#          sd <- read.table("gpcr_simmat_dc.txt")
#          sd <- as.matrix(sd)
#          st <- read.table("gpcr_simmat_dg.txt")
#          st <- as.matrix(st)
#          Y <- read.table("gpcr_admat_dgc.txt")
#          Y <- as.matrix(Y)
#          Y <- t(Y)
#        },
#        nr = {
#          cat("nr data\n")
#          flush.console()
#          sd <- read.table("nr_simmat_dc.txt")
#          sd <- as.matrix(sd)
#          st <- read.table("nr_simmat_dg.txt")
#          st <- as.matrix(st)
#          Y <- read.table("nr_admat_dgc.txt")
#          Y <- as.matrix(Y)
#          Y <- t(Y)
#        },
#        srep = {
#          cat("Scientific Reports data\n")
#          flush.console()
#          sd <- read.table("drug ChemSimilarity.txt", sep = ",")
#          sd <- data.matrix(sd)
#          st <- read.table("target SeqSimilarity.txt", sep = ",")
#          st <- data.matrix(st)
#          Y <- read.table("adjacent Matrix.txt", sep = ",")
#          Y <- data.matrix(Y)
#        },
#        stop("db should be one of the follows: 
#             {e, ic, gpcr, nr or srep}\n")
#        )
#setwd('../')
#setwd('../')
#setwd('PyDTI/')


# Compute recommendation
#result <- computeRecommendation(da)
## Not run: print(result)
Y <-(w*Y)

# Compute recommendation using similarity informations
#score <- computeRecommendation( Y,S=sd, S1=st)
#write.table(score,"DTHybridscore.txt", col.names = , row.names = F)


#lambda=0.5, alpha=0.5

## Not run: print(result1)

# Speeds up the computation process through the use of multiple threads
#library(parallel)
#cl <- makeCluster(detectCores())
score<-computeRecommendation(Y,lambda=lambda,alpha=alpha,S=sd,S1=st)
write.table(score,"DTHybridscore.txt", col.names = , row.names = F)
#stopCluster(cl)
## Not run: print(result2)
